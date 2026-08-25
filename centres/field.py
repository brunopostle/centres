import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

#: 3x3 structuring element; dilating by it gives the local maximum of a field.
_RIDGE_KERNEL = np.ones((3, 3), np.uint8)


#: Weight exponent for the spacing estimate. Medial-axis samples are weighted by
#: d**-SPACING_WEIGHT so that one large empty area — whose medial axis is both
#: long and deep — cannot outvote the many small regions that make up the
#: artwork. See edge_spacing.
SPACING_WEIGHT = 1.5

#: The distance transform is capped at this many typical inter-edge spacings.
CAP_SPACINGS = 8.0


# --- Project invariant -------------------------------------------------------
#
# Every threshold in the pipeline is expressed in units of the artwork's own
# characteristic scale — edge_spacing — never in pixels, and never as a fraction
# of an observed maximum.
#
# Pixels are a property of the photograph: how much mount, wall or table happened
# to be in shot, and what the file was resized to. edge_spacing is a property of
# the thing photographed. Every frame-derived constant this pipeline has carried
# turned out to be a bug:
#
#   - Canny's fixed 50/150 thresholds, absolute in 8-bit gradient units, made a
#     vignette delete the corners of the image outright (#10).
#   - A cap of min(h, w) / 10 made cropping 15% off the Ardabil move it from 154
#     detected centres to 242 (#11).
#   - Dividing by field.max() let whichever pixel was furthest from an edge set
#     the amplitude of the whole image (#27).
#
# A global scale estimate is correct and intended: the characteristic scale of an
# artwork is a global property of it, so measures expressed in those units move
# when the artwork changes — in proportion, which is the point. What is not
# acceptable is a scale set by the frame, or by a single outlier pixel.
#
# build_structural_field still divides by field.max(), so a single outlier pixel
# still sets the field's amplitude. Two earlier fixes for that half of #27 were
# tried and reverted: dividing by the cap instead is near-identical wherever the
# cap bites and catastrophic wherever it does not (the field's amplitude then
# falls well short of what an absolute detection threshold expects), and the
# same failure mode blocks any other fixed normalisation.
#
# What actually decoupled the two: detect_centers no longer compares against an
# absolute threshold at all. blob_log's threshold_rel takes a fraction of each
# image's own peak LoG response — since the Laplacian is linear, that response
# scales in direct proportion to whatever field.max() is, so a relative cutoff
# selects the same detections whatever field.max() turns out to be. The
# coupling this note used to warn about is now moot: rescaling the field can no
# longer move the detection threshold, because there is no longer an absolute
# number for it to move past. See detect_centers in pipeline.py and #27.
#
# A SUBTLER VIOLATION, measured on #9: edge_spacing itself is not as
# resolution-invariant as this note assumes. It is pinned near ~4 px at every
# resolution (varamin 4.12 / 4.09 / 5.17 at 1024 / 512 / 256), because the
# percentile edge threshold in _detect_edges holds edge density roughly constant
# (~8-11% of pixels) whatever the resolution, and the medial-axis half-widths of a
# fixed-density edge map are a fixed number of pixels. So edge_spacing partly tracks
# edge density rather than the artwork's scale, and the detection count tracks pixel
# area. This does not reach the reported score (the #29 wholeness gate is
# count-invariant by construction), but it does move the raw discrimination axes
# under a resize. Fixing it needs resolution-adaptive edge density, a front-end
# redesign; see PLAN.md A2 (#9).
# -----------------------------------------------------------------------------


def edge_spacing(dist):
    """Estimate the typical spacing between edges from their distance transform.

    The local maxima of ``dist`` form the medial axis of the edge map, and the
    value at each such point is the half-width of the region it sits in. The
    estimate is the weighted geometric mean of those half-widths, doubled.

    Weighting is what makes this a property of the artwork rather than of the
    frame it is photographed in. Unweighted, a large blank area contributes a
    medial axis that is both long (many samples) and deep (large values), so
    adding a museum mount raises the estimate several-fold — on the Pazyryk
    carpet, whose ground is already plain, by 5x. Weighting each sample by
    ``d**-1.5`` counts regions rather than pixels, which brings that worst case
    down to 21% and the rest of the corpus to within 10%.

    Returns 0.0 when there is no usable medial axis: when every pixel is an edge,
    and when none is — an image with no edges at all has no structure to measure
    and distanceTransform fills it with FLT_MAX, which the diagonal bound rejects.
    """
    limit = float(np.hypot(*dist.shape))  # no distance within the frame exceeds it
    ridge = dist[
        (dist > 0)
        & (dist <= limit)
        & (dist >= cv2.dilate(dist, _RIDGE_KERNEL) - 1e-6)
    ]
    if ridge.size == 0:
        return 0.0
    weights = ridge**-SPACING_WEIGHT
    half_width = np.exp((weights * np.log1p(ridge)).sum() / weights.sum()) - 1.0
    return float(2.0 * half_width)


#: Width of the illumination estimate, as a fraction of the short image side.
#: Large enough to pass over the artwork's own motifs, small enough to follow a
#: lens falloff across the frame.
_FLATFIELD_SIGMA = 0.05

#: Mid-grey the normalised image is re-centred on.
_FLATFIELD_TARGET = 128.0

#: Contrast gain: multiplies the standardised deviation from the local mean.
#: Sets the output dynamic range; the percentile edge threshold downstream makes
#: the exact value non-critical.
_FLATFIELD_GAIN = 48.0

#: Std floor in grey levels. Below this a window counts as flat, so its noise is
#: not amplified into edges. An absolute constant, hence inversion-invariant.
_FLATFIELD_STD_FLOOR = 6.0

#: Percentile of gradient magnitude taken as Canny's upper hysteresis threshold,
#: and the lower/upper threshold ratio.
_EDGE_PERCENTILE = 92.0
_EDGE_RATIO = 0.4

#: Floor on the upper threshold, in Canny's own |dx| + |dy| units: the smallest
#: contrast step that counts as a boundary at all. A 3x3 Sobel turns a
#: one-grey-level step into a response of about 4, so this is a step of roughly
#: 20 levels, or 8% of the range. A percentile alone would declare a fixed
#: fraction of *any* image to be edges, including one that is pure noise; the
#: floor is what says "there is no structure here". It does not bind on real
#: images — the corpus sits at 96-208 across every transform in the audit — so
#: the threshold in normal use is the statistic, not this constant.
_EDGE_FLOOR = 80.0


def _flat_field(gray):
    """Normalise local contrast, so a boundary reads the same everywhere.

    Each pixel becomes ``128 + K * (g - local_mean) / local_std``, with the mean
    and standard deviation taken over a Gaussian window a twentieth of the short
    side. This removes a smooth illumination gain the same way dividing by a
    blurred copy did — a vignette scales both the local mean and the local
    contrast, and dividing by the local std restores the contrast in the dark
    corners — and it does two further things the division did not.

    It is **exactly equivariant under tone inversion in real arithmetic**: with
    ``g' = 255 - g``, linearity of the Gaussian blur gives ``mean' = 255 - mean``
    and ``var' = var`` (the cross term in ``(255-g)^2`` cancels against
    ``(255-mean)^2``), so ``normalized' = 256 - normalized`` exactly. Since
    ``_FLATFIELD_TARGET = 128`` makes that constant (256) even, quantising with
    ``floor`` — what ``.astype(np.uint8)`` does — gives
    ``floor(r) + floor(256 - r) = 255`` for *every* non-integer ``r``, with no
    special-casing needed (#13, #9's ladder work first surfaced this identity).

    That guarantee needs real-valued ``mean``/``var``, though, and this used to
    compute them in float32: two float32 roundings (the blur, then the variance's
    subtraction of two close numbers) broke the identity on about 0.02% of
    pixels, letting a photographic negative's edge map differ from the original's
    by a handful of pixels — small, but exactly the sort of order-sensitive
    difference the dihedral-vote in :func:`_canny_symmetrised` exists to guard
    against, and it wasn't guarding against this one. Computing in float64 (the
    quantisation step is still ``uint8``, so this costs one pass at double
    precision, not double the pipeline) measured exact — 0 differing pixels
    across a 608,256-pixel image and its inverse, versus 13 before.
    """
    h, w = gray.shape[:2]
    g = gray.astype(np.float64)
    sigma = _FLATFIELD_SIGMA * min(h, w)
    mean = cv2.GaussianBlur(g, (0, 0), sigmaX=sigma)
    var = cv2.GaussianBlur(g * g, (0, 0), sigmaX=sigma) - mean * mean
    # Floor the std so a near-flat region does not amplify its own quantisation
    # noise into spurious edges. The floor is an absolute grey-level constant, so
    # it is itself inversion-invariant.
    std = np.sqrt(np.maximum(var, _FLATFIELD_STD_FLOOR**2))
    normalized = _FLATFIELD_TARGET + _FLATFIELD_GAIN * (g - mean) / std
    return np.clip(normalized, 0, 255).astype(np.uint8)


#: How many of the eight dihedral orientations must agree for a pixel to be an
#: edge. Four is the majority. See _detect_edges.
_EDGE_VOTES = 4


def _canny_symmetrised(blurred, low, high):
    """Canny, made exactly equivariant under the symmetries of the square.

    ``cv2.Canny`` is not. Its non-maximum suppression bins the gradient direction
    and its hysteresis traces weak edges outward from strong ones in array
    traversal order, so which marginal edges survive depends on how the image
    happens to be oriented in memory. Measured on the Ardabil: greyscale
    conversion and Gaussian blur are exactly mirror-equivariant, Canny is not —
    2907 edge pixels differ under a mirror flip, 0.54% — and the distance
    transform amplifies that into 8.6% of the field. 24 of 154 centres failed to
    survive a mirror flip, and roughness moved 3.4 points out of 10, further than
    the spread across all six carpets.

    Running Canny under each of the eight orientations of the square, mapping the
    results back and taking a majority vote, gives an edge map that is equivariant
    by construction: the vote is over the same eight results whatever orientation
    the image arrives in, so the answer cannot depend on the arrival orientation.

    Costs eight edge detections. Edge detection is not the pipeline's bottleneck —
    ``reconstruct_field`` is O(n.h.w) — so this is affordable in exchange for a
    guarantee.
    """
    votes = np.zeros(blurred.shape, dtype=np.uint8)
    for k in range(4):
        rotated = np.rot90(blurred, k)
        for flipped in (False, True):
            oriented = rotated[:, ::-1] if flipped else rotated
            edges = cv2.Canny(np.ascontiguousarray(oriented), low, high)
            if flipped:
                edges = edges[:, ::-1]
            votes += (np.rot90(edges, -k) > 0).astype(np.uint8)
    return np.where(votes >= _EDGE_VOTES, 255, 0).astype(np.uint8)


def _detect_edges(gray):
    """Canny edges with thresholds set by image statistics, not by fiat.

    Illumination is flattened first, then the hysteresis thresholds are taken
    from a percentile of the gradient magnitude. See
    :func:`build_structural_field` for why. The detection itself is symmetrised
    over the dihedral group so that the result does not depend on which way up
    the image happens to be stored — see :func:`_canny_symmetrised`.

    ``_flat_field``'s output is now exactly antisymmetric under inversion (its
    own docstring), which took this pre-blur's own inversion residual from 13
    differing pixels (out of 608,256) to 1, and the worst per-property
    inversion delta on a 44-image corpus check from ~0.75-1.0 to 0.93 (#13).

    Recomputing this blur itself in float64 too, chasing that last pixel, was
    tried and reverted. It closed the corpus residual further (44-image worst
    0.93 -> 0.19) -- cv2's default ``uint8``-in/``uint8``-out blur is evidently
    not just less precise but numerically *different* here, since the kernel
    weights are dtype-independent so this is about intermediate rounding, not
    the kernel -- but it also shifted detection counts by dozens on some
    synthetic sweep stimuli, wrongly signing `echoes`' ground-truth tracking
    and collapsing `not_separateness`' (both confirmed to trace to this change
    alone, isolated from the ``_flat_field`` fix above, which is harmless to
    both). Threshold recalibration (`_DETECTION_THRESHOLD_REL` 0.1-0.35) could
    not recover both at once. Reverted rather than trade working ground-truth
    tracking for a bigger but still-incomplete equivariance gain; the residual
    this leaves is real and open, see #13.
    """
    blurred = cv2.GaussianBlur(_flat_field(gray), (0, 0), sigmaX=2)
    # Canny's default gradient is the L1 norm of a 3x3 Sobel; match it so the
    # percentile is taken over the same quantity the thresholds are compared to.
    dx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.abs(dx) + np.abs(dy)
    high = max(float(np.percentile(magnitude, _EDGE_PERCENTILE)), _EDGE_FLOOR)
    return _canny_symmetrised(blurred, high * _EDGE_RATIO, high)


def build_structural_field(image):
    """Build a structural field that peaks at the centres of spatial regions.

    Uses a distance transform from edges so interiors of bounded regions score
    highest. Two corrections are applied to the naive version:

    1. Illumination-normalised, statistically thresholded edges. Boundaries are
       found with Canny, but neither its input nor its thresholds are taken raw:

       - The illumination field is divided out first (``_flat_field``). A
         vignette, a raking light or an uneven scan multiplies the image by a
         smooth gain, and Canny compares an 8-bit gradient against absolute
         numbers. With the fixed 50/150 thresholds this function used to carry,
         a moderate vignette pushed the corners of the frame below the lower
         threshold and their edges vanished outright — which is what made the
         whole tool so sensitive to vignetting.
       - A pre-blur (sigma=2) suppresses fine texture so that only major
         structural boundaries are detected as edges. Without this, densely
         textured images (e.g. fine carpet weave) produce edges on ~30% of
         pixels, making every interior point close to an edge and shifting the
         field peaks to featureless background areas at the image periphery.
       - The hysteresis thresholds are then a percentile of the gradient
         magnitude rather than fixed constants, so the edge density stays
         roughly constant from image to image and the field is unchanged by
         any monotone change of overall contrast.

    2. Distance cap — the distance transform is capped at CAP_SPACINGS times the
       typical spacing between edges (see edge_spacing). Beyond that distance a
       region carries no more structural information than one at exactly that
       distance, so the values are flattened. This is what stops large smooth
       background areas (museum mounts, white borders) from accumulating
       arbitrarily high distance values that dominate the field and draw centres
       to the image boundary rather than the carpet interior.

       The cap was previously min(h,w)/10, a function of the image frame rather
       than of the artwork. Because the field is normalised by its maximum, and
       that maximum is the cap wherever the cap bites, a frame-derived cap
       rescaled the whole field whenever the image was recropped: cropping 15%
       off the Ardabil moved it from 154 detected centres to 242 and its
       gradients score from 6.9 to 1.8. A cap in units of the artwork's own
       edge spacing leaves the field on the retained region essentially
       unchanged (see AUDIT.md and the A4 notes in PLAN.md).

    3. Normalisation — the field is divided by its own maximum, which means a
       single pixel, whichever happens to be furthest from an edge, sets the
       amplitude of the whole image. That used to matter because detect_centers
       compared the result to an absolute LoG threshold; it no longer does, so
       rescaling the field can no longer change what gets detected. See the
       project invariant note above and #27.
    """
    field, _ = _field_and_distance(image)
    return field


def _field_and_distance(image):
    """Shared implementation behind build_structural_field.

    Returns ``(field, dist)`` where ``dist`` is the *raw*, uncapped Euclidean
    distance-to-edge transform the field is built from. ``build_structural_field``
    exposes only ``field``; ``detect_centers`` additionally needs ``dist`` itself
    to tell a real bounded centre from a local maximum on a structureless plateau
    (#8) -- the distinction the capped, normalised field has already discarded.
    Split out so both can share one (8x-orientation) edge detection pass rather
    than paying for it twice.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = _detect_edges(gray)
    # distanceTransform expects 0 = obstacle; Canny gives 255 on edges
    dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
    spacing = edge_spacing(dist)
    if spacing <= 0:
        # No usable medial axis: no edges at all, or every pixel an edge. There
        # is no structure to measure and no scale to express it in, so the field
        # is empty rather than an arbitrary constant.
        return np.zeros(gray.shape, dtype=float), dist
    capped = np.minimum(dist, CAP_SPACINGS * spacing)
    # gaussian_filter is linear, so blur(255-gray) = 255-blur(gray) exactly -- under
    # tone inversion this term doesn't shift by a small residual, it *inverts*,
    # everywhere in the image at once. That turned out to be the dominant driver
    # of #13's inversion-equivariance residual, well past the edge map itself
    # (which already differs by ~1 pixel in 135000 -- see _canny_symmetrised and
    # the #30/#31 fixes). (blur - 0.5)^2, rescaled back to [0, 1], folds both
    # polarities onto the same value, so the term no longer cares which tonal
    # direction the image arrived in. A first attempt used min(blur, 1-blur),
    # which works just as well on every real image tried but has a kink at
    # blur=0.5 that injected spurious field structure and collapsed one point
    # of the border_band sweep (539 -> 315 centres, neighbours untouched) --
    # exactly the kind of discontinuity-amplification #27 already burned time
    # on. The squared fold is smooth, so it doesn't create new field ridges.
    # Measured: detection churn under inversion on 4 corpus images (ardabil,
    # varamin, bidjar, pazyryk) drops from 15/25/20/7 candidates to 0/3/0/2,
    # same as the kinked version, with no sweep collapses anywhere.
    blur = gaussian_filter(gray.astype(float) / 255.0, sigma=3)
    blur = (blur - 0.5) ** 2 * 4.0
    field = capped + 0.1 * blur
    return field / (field.max() + 1e-8), dist


def reconstruct_field(shape, centers):
    """Reconstruct a continuous wholeness field as a sum of Gaussian kernels."""
    h, w = shape
    Y, X = np.mgrid[0:h, 0:w]
    field = np.zeros(shape)
    for c in centers:
        kernel = np.exp(-((X - c.x) ** 2 + (Y - c.y) ** 2) / (2 * c.scale**2))
        field += c.strength * kernel
    field = field / (field.max() + 1e-8)
    return field
