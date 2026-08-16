"""Region descriptors (#22).

The audit found eleven of fifteen measures computing a different quantity from
the one named on them, because the information was never in the representation.
These tests assert the descriptors recover the quantities the source names.
"""

import cv2
import numpy as np
import pytest

from centres.field import build_structural_field
from centres.pipeline import assign_polarity, detect_centers
from centres.regions import Region, segment_regions, _describe


def _mask(draw, n=200):
    m = np.zeros((n, n), bool)
    canvas = np.zeros((n, n), np.uint8)
    draw(canvas)
    return canvas > 0


def _region_of(draw, n=200, tone=255):
    mask = _mask(draw, n)
    gray = np.where(mask, tone, 0).astype(np.uint8)
    return _describe(mask, gray)


def test_compactness_is_one_for_a_disc_and_less_for_a_star():
    disc = _region_of(lambda c: cv2.circle(c, (100, 100), 60, 255, -1))
    pts = []
    for k in range(16):
        a = k * np.pi / 8
        r = 60 if k % 2 == 0 else 22
        pts.append([100 + r * np.cos(a), 100 + r * np.sin(a)])
    star = _region_of(lambda c: cv2.fillPoly(c, [np.array(pts, np.int32)], 255))
    assert disc.compactness == pytest.approx(1.0, abs=0.12)
    assert star.compactness < disc.compactness - 0.3


def test_solidity_separates_convex_from_concave():
    disc = _region_of(lambda c: cv2.circle(c, (100, 100), 60, 255, -1))
    crescent = _mask(lambda c: (cv2.circle(c, (100, 100), 60, 255, -1),
                                cv2.circle(c, (130, 100), 45, 0, -1)))
    gray = np.where(crescent, 255, 0).astype(np.uint8)
    assert disc.solidity == pytest.approx(1.0, abs=0.05)
    assert _describe(crescent, gray).solidity < 0.85


def test_vertical_symmetry_is_one_for_a_symmetric_shape():
    disc = _region_of(lambda c: cv2.circle(c, (100, 100), 50, 255, -1))
    assert disc.vertical_symmetry == pytest.approx(1.0, abs=0.05)
    assert disc.horizontal_symmetry == pytest.approx(1.0, abs=0.05)


def test_vertical_and_horizontal_symmetry_are_distinguished():
    """The source is specific that the axis is the vertical one."""
    wedge = np.array([[40, 40], [160, 100], [40, 160]], np.int32)
    r = _region_of(lambda c: cv2.fillPoly(c, [wedge], 255))
    assert r.horizontal_symmetry > r.vertical_symmetry + 0.15


def test_tone_is_recorded_where_the_field_discards_it():
    dark = _region_of(lambda c: cv2.circle(c, (100, 100), 50, 255, -1), tone=40)
    light = _region_of(lambda c: cv2.circle(c, (100, 100), 50, 255, -1), tone=220)
    assert dark.tone < 0.25 < light.tone


def test_elongation_separates_a_disc_from_a_bar():
    disc = _region_of(lambda c: cv2.circle(c, (100, 100), 50, 255, -1))
    bar = _region_of(lambda c: cv2.rectangle(c, (20, 90), (180, 110), 255, -1))
    assert disc.elongation > 0.8
    assert bar.elongation < 0.3


def test_similar_shapes_have_similar_signatures():
    small = _region_of(lambda c: cv2.circle(c, (100, 100), 30, 255, -1))
    large = _region_of(lambda c: cv2.circle(c, (100, 100), 60, 255, -1))
    square = _region_of(lambda c: cv2.rectangle(c, (55, 55), (145, 145), 255, -1))
    d_scale = abs(small.shape_signature[0] - large.shape_signature[0])
    d_kind = abs(small.shape_signature[0] - square.shape_signature[0])
    assert d_scale < d_kind, "a circle should resemble a circle across scales"


def test_segmentation_gives_every_centre_a_region():
    from audit.stimuli import jittered_lattice

    img = jittered_lattice(0.15, n=512)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    field = build_structural_field(img)
    centers = segment_regions(field, assign_polarity(detect_centers(field), gray), gray)
    assert centers
    assert all(isinstance(c.region, Region) for c in centers)
    assert sum(c.region.area > 0 for c in centers) > 0.8 * len(centers)
