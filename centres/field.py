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
# ONE PLACE STILL VIOLATES THIS, knowingly: build_structural_field divides by
# field.max(), and detect_centers thresholds the result at an absolute 0.08.
# Dividing by the cap instead was tried and reverted — it is near-identical
# wherever the cap bites, and catastrophic wherever it does not, because the
# absolute threshold then rejects almost every detection. The field's scale and
# the detection threshold have to be fixed together, in the same units, or not at
# all. Tracked on #27.
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

#: Illumination floor, as a fraction of mean brightness. Caps the gain applied
#: to near-black regions so their quantisation noise is not amplified into edges.
_FLATFIELD_FLOOR = 0.1

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
    """Divide out a smoothly varying illumination field.

    A vignette, a raking light or an uneven scan multiplies the image by a
    smooth gain. Dividing by a heavily blurred copy of the image estimates that
    gain and removes it, leaving local contrast — which is what a boundary
    actually is — at the same amplitude everywhere in the frame.
    """
    h, w = gray.shape[:2]
    gray = gray.astype(np.float32)
    illumination = cv2.GaussianBlur(
        gray, (0, 0), sigmaX=_FLATFIELD_SIGMA * min(h, w)
    )
    floor = max(1.0, _FLATFIELD_FLOOR * float(gray.mean()))
    normalized = _FLATFIELD_TARGET * gray / np.maximum(illumination, floor)
    return np.clip(normalized, 0, 255).astype(np.uint8)


def _detect_edges(gray):
    """Canny edges with thresholds set by image statistics, not by fiat.

    Illumination is flattened first, then the hysteresis thresholds are taken
    from a percentile of the gradient magnitude. See
    :func:`build_structural_field` for why.
    """
    blurred = cv2.GaussianBlur(_flat_field(gray), (0, 0), sigmaX=2)
    # Canny's default gradient is the L1 norm of a 3x3 Sobel; match it so the
    # percentile is taken over the same quantity the thresholds are compared to.
    dx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.abs(dx) + np.abs(dy)
    high = max(float(np.percentile(magnitude, _EDGE_PERCENTILE)), _EDGE_FLOOR)
    return cv2.Canny(blurred, high * _EDGE_RATIO, high)


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

    3. Normalisation — the field is divided by its own maximum, which is a known
       weakness rather than a design choice.

       It means a single pixel, whichever happens to be furthest from an edge,
       sets the amplitude of the whole image, while detect_centers applies an
       absolute LoG threshold to the result. That coupling is how vignetting used
       to do its damage: corners lost edges, the distance transform there ran to
       the cap, field.max() jumped (varamin 32 to 69), and every centre count
       moved. Correction 2 defuses most of it — where the cap bites, field.max()
       *is* the cap, which is a property of the artwork.

       Dividing by the cap directly was tried and reverted (#27). It is
       near-identical on the corpus, where the cap always bites, but it breaks
       wherever the cap does not: on a sparse lattice CAP_SPACINGS * spacing is
       270 px against a largest actual distance of 74, so the field peaks at 0.27
       and the absolute 0.08 threshold rejects nearly every detection. Centre
       counts on the synthetic generators collapsed from hundreds to single
       figures.

       The lesson is that the field's scale and the detection threshold cannot be
       fixed independently: whatever sets the amplitude has to be the same
       quantity the threshold is expressed in. Left coupled and honest until both
       are addressed together.
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
        return np.zeros(gray.shape, dtype=float)
    dist = np.minimum(dist, CAP_SPACINGS * spacing)
    blur = gaussian_filter(gray.astype(float) / 255.0, sigma=3)
    field = dist + 0.1 * blur
    return field / (field.max() + 1e-8)


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
