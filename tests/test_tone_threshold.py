"""Guards for the tone-robust figure/ground split (#31).

``boundaries`` and ``deep_interlock`` binarise the image with a symmetrised Otsu
threshold instead of a fixed grey 128, so they no longer drift when a gamma or JPEG
change remaps tone. These check the two properties that make that work: the
threshold is exactly inversion-symmetric (so tone inversion, #30, still leaves the
split unchanged), and the measures move little under a gamma change where the fixed
threshold moved a lot.
"""

import cv2
import numpy as np
import pytest

from centres.properties import _tone_threshold, boundaries, deep_interlock
from audit import stimuli, transforms


def test_tone_threshold_is_exactly_inversion_symmetric():
    """t(255 - g) == 255 - t(g), the identity that keeps the split inversion-invariant."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        g = rng.integers(0, 256, (128, 128), dtype=np.uint8)
        t = _tone_threshold(g)
        t_inv = _tone_threshold(255 - g)
        assert t_inv == pytest.approx(255.0 - t, abs=1e-9)


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def test_boundaries_and_interlock_are_gamma_stable():
    """A gamma remap should barely move these image-domain measures now."""
    img = stimuli.bordered_motifs(0.3)          # a clean figure/ground stimulus
    g0 = _gray(img)
    g1 = _gray(transforms.gamma(img, 1.4))

    b0, b1 = boundaries(None, [], None, gray=g0), boundaries(None, [], None, gray=g1)
    d0, d1 = deep_interlock([], None, gray=g0), deep_interlock([], None, gray=g1)
    # Raw values are in [0, 1]; a gamma remap must not move either by more than a
    # tenth. With the old fixed-128 threshold the gamma shift reached several tenths.
    assert abs(b0 - b1) < 0.1, (b0, b1)
    assert abs(d0 - d1) < 0.1, (d0, d1)
