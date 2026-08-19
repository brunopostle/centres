"""Tests for the candidate global-redundancy discriminators (audit/redundancy.py).

These check the properties the discrimination stage relies on: that the entropy
is lower for a redundant population than a diffuse one, that it is invariant to a
monotone rescaling of the inputs (the reason the histogram is ranged to the data),
and that the rank-separation helper counts pairs correctly. They do not re-run the
pipeline; the empirical carpet-vs-noise separation is measured by ``python -m audit``.
"""

import numpy as np
import pytest

from centres.centers import Center
from audit import redundancy


def _centers(scales, strengths):
    return [Center(id=i, x=0.0, y=0.0, scale=float(s), strength=float(w))
            for i, (s, w) in enumerate(zip(scales, strengths))]


def test_entropy_none_below_two_samples():
    assert redundancy._entropy([]) is None
    assert redundancy._entropy([0.5]) is None
    assert redundancy.strength_entropy([]) is None
    assert redundancy.strength_entropy(_centers([10], [0.5])) is None


def test_strength_entropy_lower_for_redundant_population():
    """A few recurring strengths -> low entropy; strengths spread across the range -> high."""
    rng = np.random.default_rng(0)
    redundant = _centers([10] * 200, rng.choice([0.2, 0.5, 0.8], size=200))
    diffuse = _centers([10] * 200, rng.uniform(0.0, 1.0, size=200))
    assert redundancy.strength_entropy(redundant) < redundancy.strength_entropy(diffuse)


def test_scale_entropy_lower_for_redundant_population():
    rng = np.random.default_rng(0)
    redundant = _centers(rng.choice([5.0, 15.0, 45.0], size=200), [0.5] * 200)
    diffuse = _centers(rng.uniform(2.0, 48.0, size=200), [0.5] * 200)
    assert redundancy.scale_entropy(redundant) < redundancy.scale_entropy(diffuse)


def test_entropy_invariant_to_affine_rescaling():
    """Ranging the histogram to the data makes entropy blind to a monotone rescale."""
    rng = np.random.default_rng(1)
    base = rng.uniform(0.0, 1.0, size=300)
    assert redundancy._entropy(base) == pytest.approx(
        redundancy._entropy(0.3 * base + 5.0), abs=1e-9)


def test_pairwise_order_counts_pairs():
    # every art strictly above every noise -> 1.0 when higher=life
    assert redundancy.pairwise_order([3, 4, 5], [0, 1, 2], True) == 1.0
    # fully inverted
    assert redundancy.pairwise_order([0, 1, 2], [3, 4, 5], True) == 0.0
    # lower=life flips the sense
    assert redundancy.pairwise_order([0, 1], [3, 4], False) == 1.0
    # ties count as unordered (neither > nor <)
    assert redundancy.pairwise_order([1, 1], [1, 1], True) == 0.0


def test_pairwise_order_none_when_a_side_empty():
    assert redundancy.pairwise_order([], [1, 2], True) is None
    assert redundancy.pairwise_order([1, 2], [None, None], True) is None
