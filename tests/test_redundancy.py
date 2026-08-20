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


def _placed(xy, strengths):
    return [Center(id=i, x=float(x), y=float(y), scale=10.0, strength=float(s))
            for i, ((x, y), s) in enumerate(zip(xy, strengths))]


def test_spatial_coherence_none_below_k_plus_two():
    rng = np.random.default_rng(0)
    xy = rng.integers(0, 1000, (5, 2))
    assert redundancy.spatial_coherence(_placed(xy, [0.5] * 5), k=6) is None


def test_spatial_coherence_high_when_strength_tracks_position():
    """Strength = x/1000: neighbours in space have near-equal strength -> Moran's I ~ 1."""
    rng = np.random.default_rng(0)
    xy = rng.integers(0, 1000, (200, 2))
    ordered = _placed(xy, [x / 1000.0 for x, _ in xy])
    assert redundancy.spatial_coherence(ordered) > 0.8


def test_spatial_coherence_near_zero_for_random_strengths():
    rng = np.random.default_rng(1)
    xy = rng.integers(0, 1000, (200, 2))
    rand = _placed(xy, rng.random(200))
    assert abs(redundancy.spatial_coherence(rand)) < 0.15


def test_spatial_coherence_zero_when_all_strengths_equal():
    rng = np.random.default_rng(2)
    xy = rng.integers(0, 1000, (50, 2))
    assert redundancy.spatial_coherence(_placed(xy, [0.7] * 50)) == 0.0


def test_combined_separation_or_rule():
    # noise: high entropy, low coherence
    noise_se = [2.6, 2.7, 2.55]
    noise_mo = [0.05, 0.07, 0.03]
    # art A separated by entropy only, art B by coherence only, art C by neither
    art_se = [2.0, 2.65, 2.65]   # A low, B/C high
    art_mo = [0.02, 0.30, 0.04]  # B high, A/C low
    frac, margin = redundancy.combined_separation(art_se, art_mo, noise_se, noise_mo)
    # A (entropy) and B (coherence) separate; C crosses both -> 2/3
    assert frac == pytest.approx(2 / 3)
    # C's margin is negative (fails both), so the tightest margin is < 0
    assert margin < 0

    # all separated -> fraction 1.0, positive margin
    frac2, margin2 = redundancy.combined_separation([2.0, 2.6], [0.02, 0.30],
                                                    noise_se, noise_mo)
    assert frac2 == 1.0 and margin2 > 0
