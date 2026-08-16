"""Figure/ground polarity (#26).

The structural field is a distance transform from edges and carries no tone, so
it cannot tell a motif from the gap between motifs. Polarity is the signed local
contrast that can.
"""

import cv2
import numpy as np
import pytest

from centres.field import build_structural_field
from centres.pipeline import assign_polarity, detect_centers

from audit.stimuli import jittered_lattice


def _polarised(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return assign_polarity(detect_centers(build_structural_field(img)), gray)


def test_separates_motifs_from_the_gaps_between_them():
    """The case that motivated the whole issue.

    225 identical circles on a lattice yield 481 centres: each circle exactly
    once, plus 256 in the gaps. The gap detections respond *more* strongly to the
    LoG than any motif does, so no threshold separates them. Polarity does.
    """
    img = jittered_lattice(0.0, n=1024, step=64, radius=16)
    centers = _polarised(img)
    truth = np.array(
        [(x, y) for y in range(64, 1024 - 63, 64) for x in range(64, 1024 - 63, 64)],
        dtype=float,
    )
    motif, gap = [], []
    for c in centers:
        nearest = np.hypot(truth[:, 0] - c.x, truth[:, 1] - c.y).min()
        (motif if nearest <= 32 else gap).append(c.polarity)
    assert len(motif) > 200 and len(gap) > 200, "expected both populations"

    correct = sum(p > 0 for p in motif) + sum(p <= 0 for p in gap)
    assert correct / len(centers) >= 0.95
    # in fact the two populations do not overlap at all
    assert min(motif) > max(gap)


def test_polarity_is_bounded():
    for c in _polarised(jittered_lattice(0.2)):
        assert -1.0 <= c.polarity <= 1.0


def test_inverting_the_image_exactly_negates_polarity():
    """Which sign means "figure" is a convention; the magnitude must not be.

    A light-on-dark artwork must not be treated as structurally different from
    the same design dark-on-light. Canny works on gradient magnitude, so
    inverting the image leaves the edges — and therefore the detected centres —
    identical; only the tone flips.

    Michelson contrast, (s - i) / (s + i), fails this: it flips sign but not
    magnitude, since the denominator is not invariant under inversion. Dividing
    by the local range instead is exactly antisymmetric.
    """
    img = jittered_lattice(0.0, n=512, step=64, radius=16)
    key = lambda c: (round(c.x, 3), round(c.y, 3))
    direct = sorted(_polarised(img), key=key)
    inverted = sorted(_polarised(255 - img), key=key)

    assert len(direct) == len(inverted), "inversion must not change the centre set"
    for a, b in zip(direct, inverted):
        assert key(a) == key(b), "same centres expected, in sorted order"
        assert a.polarity == pytest.approx(-b.polarity, abs=1e-9)


def test_uniform_surround_gives_no_polarity():
    """A centre sitting on no tonal boundary has no polarity to report."""
    from centres.centers import Center

    gray = np.full((200, 200), 128, np.uint8)
    c = Center(id=0, x=100.0, y=100.0, scale=20.0, strength=1.0)
    assign_polarity([c], gray)
    assert c.polarity == pytest.approx(0.0, abs=1e-9)


def test_default_is_zero_so_untouched_centres_are_neutral():
    from centres.centers import Center

    assert Center(id=0, x=0.0, y=0.0, scale=1.0, strength=1.0).polarity == 0.0
