"""Regression guard for the #29 discriminator (AUDIT.md §17).

The "redundancy OR spatial coherence" rule keeps every real artwork separated from
noise, at rest and under a resize. This exercises the real pipeline on two corpus
images chosen because they stress *different* axes — the Varamin is redundant but
spatially flat, the Egyptian geometric tile is spatially coherent but not redundant
— so it guards the complementarity that makes the OR rule work, which a single-axis
test could not. If a pipeline change quietly breaks the separation, this fails here
rather than a later audit noticing.

Kept deliberately cheap: it runs at 512 px (and a 256 px downscale for the resize
check) and uses one noise generator, so it adds only a few seconds. The project's
CI does not run pytest, so a "slow, CI-only" mark would never execute — the guard
has to be fast enough to live in the default suite.
"""

import os

import cv2
import pytest

from centres.pipeline import analyze
from audit import redundancy, stimuli

IMAGES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "images")


def _stats(img, size):
    """Strength entropy and spatial coherence after scaling the long side to ``size``."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    _, centers, _, _ = analyze(img)
    return redundancy.strength_entropy(centers), redundancy.spatial_coherence(centers)


def _read(name):
    img = cv2.imread(os.path.join(IMAGES, name))
    if img is None:
        pytest.skip(f"{name} not in the corpus")
    return img


def test_or_rule_separates_complementary_artworks_and_survives_resize():
    varamin = _read("varamin.jpg")
    egypt = _read("tile_geometric_egypt.jpg")

    for size in (512, 256):
        floor, ceiling = _stats(stimuli.white_noise(seed=0), size)  # noise bounds at this res
        v_se, v_mo = _stats(varamin, size)
        e_se, e_mo = _stats(egypt, size)

        # The OR rule: each artwork clears the noise cloud on at least one axis.
        assert (v_se < floor) or (v_mo > ceiling), ("varamin", size, v_se, v_mo, floor, ceiling)
        assert (e_se < floor) or (e_mo > ceiling), ("egypt", size, e_se, e_mo, floor, ceiling)

        if size == 512:
            # Complementarity at the working resolution: the Varamin is carried by
            # redundancy (low entropy), the Egyptian tile by spatial coherence. These
            # are the two axes; a discriminator that lost either would fail one here.
            assert v_se < floor, ("varamin redundancy", v_se, floor)
            assert e_mo > ceiling, ("egypt coherence", e_mo, ceiling)
