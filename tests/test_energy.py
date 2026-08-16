import math
import numpy as np
import pytest
from centres.centers import Center
from centres.energy import (
    hierarchy_energy,
    reinforcement_energy,
    coverage_energy,
    alignment_energy,
    field_energy,
    locality_energy,
    total_energy,
)
from centres.graph import build_graph, propagate_strength
from centres.field import reconstruct_field
from centres.pipeline import assign_hierarchy


def c(id, x, y, scale, strength=1.0, parent=None):
    return Center(
        id=id,
        x=float(x),
        y=float(y),
        scale=float(scale),
        strength=strength,
        parent=parent,
    )


# --- hierarchy_energy ---


def test_hierarchy_zero_at_ratio_three():
    parent = c(0, 0, 0, 30.0)
    child = c(1, 5, 5, 10.0, parent=0)
    assert hierarchy_energy([parent, child]) == pytest.approx(0.0, abs=1e-10)


def test_hierarchy_nonzero_wrong_ratio():
    parent = c(0, 0, 0, 10.0)
    child = c(1, 0, 0, 10.0, parent=0)  # ratio = 1, not 3
    assert hierarchy_energy([parent, child]) > 0


def test_hierarchy_zero_no_parents():
    centers = [c(i, 0, 0, float(i + 1)) for i in range(4)]
    assert hierarchy_energy(centers) == 0.0


# --- reinforcement_energy ---


def test_reinforcement_more_negative_with_higher_strength():
    G_weak = build_graph([c(0, 0, 0, 10, strength=0.1), c(1, 5, 0, 10, strength=0.1)])
    G_strong = build_graph([c(0, 0, 0, 10, strength=2.0), c(1, 5, 0, 10, strength=2.0)])
    assert reinforcement_energy(G_strong) < reinforcement_energy(G_weak)


def test_reinforcement_zero_no_edges():
    G = build_graph([c(0, 0, 0, 3), c(1, 500, 0, 3)])
    assert not G.has_edge(0, 1)
    assert reinforcement_energy(G) == 0.0


def test_reinforcement_negative_when_edges_exist():
    G = build_graph([c(0, 0, 0, 10, strength=1.0), c(1, 5, 0, 10, strength=1.0)])
    assert G.has_edge(0, 1)
    assert reinforcement_energy(G) < 0.0


# --- coverage_energy ---


def test_coverage_zero_at_ideal():
    # child scale = sqrt(0.65) * parent_scale → coverage exactly 0.65
    child_scale = math.sqrt(0.65) * 10.0
    parent = c(0, 0, 0, 10.0)
    child = c(1, 3, 0, child_scale, parent=0)
    assert coverage_energy([parent, child]) == pytest.approx(0.0, abs=1e-6)


def test_coverage_nonzero_overcrowded():
    # Many large children → coverage >> 0.65
    parent = c(0, 0, 0, 10.0)
    children = [c(i + 1, float(i), 0, 8.0, parent=0) for i in range(5)]
    assert coverage_energy([parent] + children) > 0


def test_coverage_zero_no_children():
    centers = [c(i, 0, 0, 10.0) for i in range(3)]
    assert coverage_energy(centers) == 0.0


# --- alignment_energy ---


def test_alignment_zero_at_half_radius():
    parent = c(0, 0, 0, 20.0)
    child = c(1, 10, 0, 5.0, parent=0)  # dist=10 = 0.5 * 20
    assert alignment_energy([parent, child]) == pytest.approx(0.0, abs=1e-10)


def test_alignment_nonzero_at_center():
    parent = c(0, 0, 0, 20.0)
    child = c(1, 0, 0, 5.0, parent=0)  # dist=0, d=0, far from 0.5
    assert alignment_energy([parent, child]) > 0


def test_alignment_zero_no_parents():
    centers = [c(i, float(i * 10), 0, 10.0) for i in range(3)]
    assert alignment_energy(centers) == 0.0


# --- field_energy ---


def test_field_energy_zero_for_uniform():
    assert field_energy(np.ones((50, 50))) == pytest.approx(0.0, abs=1e-10)


def test_field_energy_positive_for_gradient():
    field = np.zeros((50, 50))
    field[:, 25:] = 1.0  # step discontinuity
    assert field_energy(field) > 0


def test_field_energy_higher_for_sharper_gradient():
    gentle = np.linspace(0, 1, 50)[np.newaxis, :] * np.ones((50, 1))
    sharp = np.zeros((50, 50))
    sharp[:, 25:] = 1.0
    assert field_energy(sharp) > field_energy(gentle)


# --- total_energy ---


def test_total_energy_finite():
    centers = [c(0, 30, 30, 20.0), c(1, 35, 30, 6.0, parent=0)]
    G = build_graph(centers)
    G = propagate_strength(G)
    field = reconstruct_field((60, 60), centers)
    assert np.isfinite(total_energy(field, centers, G))


def _random_centers(n, side, seed, scale_range=(2.8, 16.8)):
    """n centres drawn from a fixed distribution over a side x side canvas."""
    rng = np.random.default_rng(seed)
    lo, hi = np.log(scale_range[0]), np.log(scale_range[1])
    centers = [
        c(i, rng.uniform(0, side), rng.uniform(0, side),
          float(np.exp(rng.uniform(lo, hi))),
          strength=float(rng.uniform(0.3, 1.0)))
        for i in range(n)
    ]
    return assign_hierarchy(centers)


def _energy_of(centers, shape):
    G = propagate_strength(build_graph(centers))
    return total_energy(reconstruct_field(shape, centers), centers, G)


def test_total_energy_does_not_track_centre_count():
    """Regression for the defect that made the total the centre count times a
    constant (AUDIT.md section 2, issue #14).

    Centre sets are drawn at *constant spatial density* — the canvas area grows
    with n — so every per-pair, per-parent and per-edge statistic is the same in
    distribution at every n, and an intensive energy must be flat in n. Before
    the fix, hierarchy, coverage and alignment entered as sums over centres and
    the total grew almost exactly linearly with n (r = 0.99 over the audit
    corpus). The threshold matches the acceptance criterion for that issue.

    The measured value here is about -0.40 rather than 0. Roughly -0.33 of that
    is locality_energy, which is a mean over all N(N-1)/2 centre pairs while the
    pairs that actually overlap number O(N), so it decays as 1/N when the whole
    scene is scaled up. That does not affect comparability between images — on
    the corpus locality contributes 0.004 of a total of about 1.3 — but it is
    the one term that is not intensive, and it is why the threshold is not
    tighter. Dropping the locality term leaves r = -0.07.
    """
    ns, energies = [], []
    for n in (32, 64, 128, 256):
        side = 24.5 * math.sqrt(n)  # constant density: area proportional to n
        shape = (int(side), int(side))
        for seed in (0, 1, 2):
            centers = _random_centers(n, side, seed)
            ns.append(n)
            energies.append(_energy_of(centers, shape))
    r = float(np.corrcoef(ns, energies)[0, 1])
    assert abs(r) < 0.5, f"structural energy still tracks centre count: r = {r:+.3f}"


def test_total_energy_invariant_under_uniform_rescaling():
    """Doubling every length must not change the energy.

    Every term is dimensionless except the field term, whose mean squared
    gradient carries units of 1/pixel^2; total_energy scales it by the squared
    rms centre radius for exactly this reason.
    """
    base = _random_centers(24, 160.0, seed=7)
    scaled = [c(x.id, x.x * 2, x.y * 2, x.scale * 2, x.strength, x.parent) for x in base]
    e1 = _energy_of(base, (160, 160))
    e2 = _energy_of(scaled, (320, 320))
    assert e2 == pytest.approx(e1, rel=0.05)


def test_collapse_costs_more_than_a_spread_configuration():
    """The locality barrier must keep the degenerate cluster uphill.

    All centres at one point and one scale is the configuration that maximises
    reinforcement, and it also exempts itself from the hierarchy, coverage and
    alignment terms because assign_hierarchy needs a strictly larger centre to
    make a parent. WEIGHT_LOCALITY is derived to cover both.
    """
    spread = _random_centers(30, 300.0, seed=3)
    collapsed = assign_hierarchy(
        [c(i, 150.0, 150.0, 20.0, strength=x.strength) for i, x in enumerate(spread)]
    )
    assert locality_energy(collapsed) == pytest.approx(1.0)
    assert _energy_of(collapsed, (300, 300)) > _energy_of(spread, (300, 300))
