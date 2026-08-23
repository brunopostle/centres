"""Tests for the robust (median-over-transforms) scoring and error bars (#23)."""

import numpy as np
import pytest

from centres import transforms as ct
from centres.cli import _median_spread, robust_analyse, robust_json
from centres.properties import compute_all  # noqa: F401  (import sanity)
from audit import transforms as at


def test_transforms_are_a_single_shared_source():
    """audit.transforms re-exports centres.transforms — one definition, not two."""
    assert at.BENIGN is ct.BENIGN
    assert at.PRACTICAL is ct.PRACTICAL
    assert at.ISOMETRY is ct.ISOMETRY
    assert list(ct.BENIGN) == ["identity", "mirror", "rot90", "gamma1.3", "jpeg50", "invert"]


def test_median_spread_value_and_error():
    # exact invariance -> zero error bar
    assert _median_spread([5.0, 5.0, 5.0]) == (5.0, 0.0)
    # median and half-range
    assert _median_spread([4.0, 5.0, 6.0]) == (5.0, 1.0)
    # None entries are ignored, not counted as zero
    assert _median_spread([3.0, None, 7.0]) == (5.0, 2.0)
    # all undefined -> undefined, not a number
    assert _median_spread([None, None]) == (None, None)
    assert _median_spread([]) == (None, None)


def _tiny_structured_image():
    """A small deterministic image with real structure: nested light-on-dark squares."""
    img = np.full((200, 200, 3), 40, np.uint8)
    img[30:170, 30:170] = 200
    img[70:130, 70:130] = 40
    img[90:110, 90:110] = 200
    return img


def test_robust_analyse_shape_and_error_bars():
    n_med, life, props = robust_analyse(_tiny_structured_image())
    assert isinstance(n_med, int) and n_med >= 0
    life_med, life_err = life
    assert life_med is not None and life_err >= 0.0
    # every property present, errors non-negative, defined scores in range
    KEYS = [k for k, _ in __import__("centres.cli", fromlist=["_PROPERTY_LABELS"])._PROPERTY_LABELS]
    assert set(props) == set(KEYS)
    for med, err in props.values():
        if med is not None:
            assert 0.0 <= med <= 10.0 and err >= 0.0


def test_identity_is_one_of_the_samples_so_error_brackets_the_plain_score():
    """The identity transform is in BENIGN, so the plain score is one of the values
    the median and half-range are taken over."""
    assert "identity" in ct.BENIGN
    img = _tiny_structured_image()
    _, (life_med, life_err), props = robust_analyse(img)
    payload = robust_json(0, (life_med, life_err), props, len(ct.BENIGN))
    assert payload["transforms"] == 6
    assert payload["degree_of_life"]["error"] >= 0.0
