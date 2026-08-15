import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


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

    2. Distance cap — the distance transform is capped at min(h,w)/10 pixels.
       This prevents large smooth background areas (museum mounts, white borders)
       from accumulating arbitrarily high distance values that dominate the field
       and draw centres to the image boundary rather than the carpet interior.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = _detect_edges(gray)
    # distanceTransform expects 0 = obstacle; Canny gives 255 on edges
    dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
    dist = np.minimum(dist, min(h, w) / 10.0)
    blur = gaussian_filter(gray.astype(float) / 255.0, sigma=3)
    field = dist + 0.1 * blur
    field = field / (field.max() + 1e-8)
    return field


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
