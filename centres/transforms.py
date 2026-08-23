"""Image transformations grouped by how much they may legitimately change a score.

The grouping is the whole point. A mirror cannot change the structure of an
artwork, so any variation a measure shows under ``ISOMETRY`` is measurement noise
with no possible structural interpretation. ``BENIGN`` adds tonal and codec
changes, which likewise leave composition intact — it is the set the ``analyse
--robust`` mode scores over, reporting each measure as the median across the group
with the spread as an error bar (#23). ``PRACTICAL`` adds the things that happen to
real photographs of real objects — cropping, padding, lens vignetting, oblique
viewpoint — where a small change is defensible but a large one makes the measure
unusable in the field; the audit's invariance stage uses it.

This module is the single source of truth for the transforms; ``audit.transforms``
re-exports it so the harness and the product share exactly one definition.
"""

import cv2
import numpy as np


def identity(img):
    return img


def mirror(img):
    return img[:, ::-1].copy()


def rot90(img):
    return np.rot90(img).copy()


def gamma(img, g=1.3):
    lut = np.array([((v / 255.0) ** (1 / g)) * 255 for v in range(256)], np.uint8)
    return cv2.LUT(img, lut)


def jpeg(img, quality=50):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, 1)


def crop(img, frac=0.05):
    h, w = img.shape[:2]
    m = int(frac * min(h, w))
    return img[m:h - m, m:w - m].copy()


def pad(img, frac=0.10):
    h, w = img.shape[:2]
    m = int(frac * min(h, w))
    return cv2.copyMakeBorder(img, m, m, m, m, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def vignette(img, strength=0.6):
    h, w = img.shape[:2]
    Y, X = np.mgrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    r = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2) / np.sqrt(2)
    return np.clip(img * (1 - strength * r ** 2)[..., None], 0, 255).astype(np.uint8)


def perspective(img, amount=0.06):
    h, w = img.shape[:2]
    d = amount * w
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[d, 0], [w - d * 0.4, 0], [w, h], [0, h]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h))


def invert(img):
    """Tone inversion: a photographic negative. Cannot change structure (#30)."""
    return 255 - img


#: Exact isometries. Any variation here is pure measurement noise.
ISOMETRY = {"identity": identity, "mirror": mirror, "rot90": rot90}

#: Isometries plus tonal and codec changes that leave composition intact.
BENIGN = dict(ISOMETRY, **{
    "gamma1.3": gamma,
    "jpeg50": jpeg,
    "invert": invert,
})

#: What happens to real photographs of real objects.
PRACTICAL = dict(BENIGN, **{
    "crop5%": crop,
    "pad10%": pad,
    "vignette": vignette,
    "perspective": perspective,
})
