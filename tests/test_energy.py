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
    total_energy,
)
from centres.graph import build_graph, propagate_strength
from centres.field import reconstruct_field


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
