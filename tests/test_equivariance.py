"""Isometry equivariance of the front end (#13).

A mirror flip and a 90 degree rotation cannot change an artwork's structure, so
any variation they produce is measurement noise with no possible structural
interpretation. Before symmetrisation a mirror flip moved roughness by 3.4 points
out of 10 on the Ardabil -- further than the spread across all six carpets.
"""

import cv2
import numpy as np
import pytest

from centres.field import _detect_edges, build_structural_field

from audit.stimuli import jittered_lattice


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _rot(a, k=1):
    return np.ascontiguousarray(np.rot90(a, k))


def test_edge_map_is_equivariant_under_rotation():
    """cv2.Canny alone is not: its hysteresis traces in array traversal order."""
    gray = _gray(jittered_lattice(0.15, n=512))
    direct = _detect_edges(gray)
    viaRot = np.rot90(_detect_edges(_rot(gray)), -1)
    differing = int((direct != viaRot).sum())
    assert differing <= direct.size * 1e-5, f"{differing} edge pixels differ"


def test_edge_map_is_equivariant_under_mirror():
    gray = _gray(jittered_lattice(0.15, n=512))
    direct = _detect_edges(gray)
    viaMirror = _detect_edges(np.ascontiguousarray(gray[:, ::-1]))[:, ::-1]
    differing = int((direct != viaMirror).sum())
    assert differing <= direct.size * 1e-5, f"{differing} edge pixels differ"


def test_field_is_near_equivariant():
    """The residual is not zero, and the reason is known.

    ``cv2.GaussianBlur`` is separable and its row/column passes round differently
    under transposition, so the illumination estimate inside ``_flat_field``
    differs by a handful of pixels before any edge detection happens. A few of
    those survive into the edge map, and the distance transform amplifies them.
    Eliminating it means symmetrising the illumination estimate too, which costs
    eight large-sigma blurs. See #13.
    """
    img = jittered_lattice(0.15, n=512)
    direct = build_structural_field(img)
    viaRot = np.rot90(build_structural_field(_rot(img)), -1)
    assert np.abs(direct - viaRot).max() < 0.05
    assert np.abs(direct - viaRot).mean() < 1e-3
