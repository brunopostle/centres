import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def build_structural_field(image):
    """Build a structural field that peaks at the centres of spatial regions.

    Uses a distance transform from edges so interiors of bounded regions score
    highest. Two corrections are applied to the naive version:

    1. Pre-blur (sigma=2) before Canny — suppresses fine texture so that only
       major structural boundaries are detected as edges. Without this, densely
       textured images (e.g. fine carpet weave) produce edges on ~30% of pixels,
       making every interior point close to an edge and shifting the field peaks
       to featureless background areas at the image periphery.

    2. Distance cap — the distance transform is capped at min(h,w)/10 pixels.
       This prevents large smooth background areas (museum mounts, white borders)
       from accumulating arbitrarily high distance values that dominate the field
       and draw centres to the image boundary rather than the carpet interior.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)
    edges = cv2.Canny(gray_blurred, 50, 150)
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
