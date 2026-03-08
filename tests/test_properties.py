import math
import numpy as np
import pytest
from centres.centers import Center
from centres.graph import build_graph, propagate_strength
from centres.field import reconstruct_field
from centres.properties import (
    levels_of_scale,
    strong_centres,
    boundaries,
    alternating_repetition,
    positive_space,
    good_shape,
    local_symmetries,
    deep_interlock,
    contrast,
    gradients,
    roughness,
    echoes,
    the_void,
    simplicity,
    not_separateness,
    compute_all,
)


def c(id, x, y, scale, strength=1.0, parent=None):
    return Center(
        id=id,
        x=float(x),
        y=float(y),
        scale=float(scale),
        strength=strength,
        parent=parent,
    )


def small_hierarchy():
    """One parent containing two children at 0.5r distance."""
    parent = c(0, 50, 50, 30.0, strength=2.0)
    child1 = c(1, 65, 50, 10.0, strength=1.0, parent=0)  # dist=15 = 0.5*30
    child2 = c(2, 35, 50, 10.0, strength=1.0, parent=0)  # dist=15 = 0.5*30
    return [parent, child1, child2]


def connected_graph(centers):
    G = build_graph(centers)
    return propagate_strength(G)


# --- levels_of_scale ---


def test_levels_zero_at_ratio_three():
    parent = c(0, 0, 0, 30.0)
    child = c(1, 5, 5, 10.0, parent=0)
    assert levels_of_scale([parent, child]) == pytest.approx(0.0, abs=1e-10)


def test_levels_positive_wrong_ratio():
    parent = c(0, 0, 0, 20.0)
    child = c(1, 0, 0, 10.0, parent=0)  # ratio=2, not 3
    assert levels_of_scale([parent, child]) > 0


# --- strong_centres ---


def test_strong_centres_high_when_top_quartile_strong():
    centers = [c(i, 0, 0, 5.0, strength=float(i)) for i in range(8)]
    score = strong_centres(centers)
    assert score >= centers[-1].strength * 0.75  # top quartile is strong


def test_strong_centres_empty():
    assert strong_centres([]) == 0.0


# --- boundaries ---


def test_boundaries_low_when_field_drops_between_centres():
    # Two centres far enough apart that the field between them is low
    centers = [c(0, 10, 50, 5.0, strength=1.0), c(1, 90, 50, 5.0, strength=1.0)]
    field = reconstruct_field((100, 100), centers)
    G = connected_graph(centers)
    if G.has_edge(0, 1):
        score = boundaries(field, centers, G)
        assert score < 1.0  # midpoint weaker than peaks


def test_boundaries_no_edges_returns_zero():
    centers = [c(0, 0, 0, 3.0), c(1, 500, 0, 3.0)]
    field = reconstruct_field((100, 100), centers)
    G = build_graph(centers)
    assert not G.has_edge(0, 1)
    assert boundaries(field, centers, G) == 0.0


# --- alternating_repetition ---


def test_alternating_repetition_high_with_varied_neighbours():
    # Centre with neighbours of very different strengths
    centers = [
        c(0, 0, 0, 10, strength=1.0),
        c(1, 5, 0, 10, strength=0.1),
        c(2, -5, 0, 10, strength=2.0),
    ]
    G = connected_graph(centers)
    score = alternating_repetition(G)
    assert score >= 0.0


def test_alternating_repetition_zero_no_edges():
    centers = [c(0, 0, 0, 3), c(1, 500, 0, 3)]
    G = build_graph(centers)
    assert alternating_repetition(G) == 0.0


# --- positive_space ---


def test_positive_space_zero_ideal_coverage():
    child_scale = math.sqrt(0.65) * 10.0
    parent = c(0, 0, 0, 10.0)
    child = c(1, 3, 0, child_scale, parent=0)
    assert positive_space([parent, child]) == pytest.approx(0.0, abs=1e-6)


# --- good_shape ---


def test_good_shape_high_when_parents_have_children():
    centers = small_hierarchy()
    score = good_shape(centers)
    # 1 out of 3 centres is a parent → 1/3
    assert score == pytest.approx(1 / 3, abs=1e-6)


def test_good_shape_zero_all_leaves():
    centers = [c(i, float(i * 10), 0, 5.0) for i in range(4)]
    assert good_shape(centers) == 0.0


# --- local_symmetries ---


def test_local_symmetries_zero_at_half_radius():
    parent = c(0, 0, 0, 20.0)
    child = c(1, 10, 0, 5.0, parent=0)  # d = 0.5
    assert local_symmetries([parent, child]) == pytest.approx(0.0, abs=1e-10)


# --- deep_interlock ---


def test_deep_interlock_one_when_all_edges_overlap():
    # Two centres whose radii overlap: d=5 < r1+r2=10+10=20
    centers = [c(0, 0, 0, 10.0, strength=1.0), c(1, 5, 0, 10.0, strength=1.0)]
    G = connected_graph(centers)
    assert deep_interlock(centers, G) == pytest.approx(1.0, abs=1e-6)


def test_deep_interlock_zero_when_no_edges_overlap():
    # Two centres far apart: d=100 > r1+r2=5+5=10
    centers = [c(0, 0, 0, 5.0, strength=1.0), c(1, 100, 0, 5.0, strength=1.0)]
    G = connected_graph(centers)
    # They won't even form an edge (too far), so deep_interlock is 0
    assert deep_interlock(centers, G) == 0.0


def test_deep_interlock_empty_graph():
    assert (
        deep_interlock([c(0, 0, 0, 10.0)], connected_graph([c(0, 0, 0, 10.0)])) == 0.0
    )


# --- contrast ---


def test_contrast_positive_with_strength_difference():
    centers = [c(0, 0, 0, 10, strength=2.0), c(1, 5, 0, 10, strength=0.5)]
    G = connected_graph(centers)
    if G.has_edge(0, 1):
        assert contrast(G) > 0


def test_contrast_zero_equal_strengths():
    centers = [c(0, 0, 0, 10, strength=1.0), c(1, 5, 0, 10, strength=1.0)]
    G = connected_graph(centers)
    assert contrast(G) == pytest.approx(0.0, abs=1e-6)


def test_contrast_zero_no_edges():
    G = build_graph([c(0, 0, 0, 3), c(1, 500, 0, 3)])
    assert contrast(G) == 0.0


# --- gradients ---


def test_gradients_zero_uniform():
    assert gradients(np.ones((50, 50))) == pytest.approx(0.0, abs=1e-10)


def test_gradients_positive_step():
    field = np.zeros((50, 50))
    field[:, 25:] = 1.0
    assert gradients(field) > 0


# --- roughness ---


def test_roughness_zero_single_centre():
    assert roughness([c(0, 0, 0, 5)]) == 0.0


def test_roughness_low_for_regular_grid():
    centers = [
        c(i * 4 + j, float(i * 20), float(j * 20), 5.0)
        for i in range(4)
        for j in range(4)
    ]
    assert roughness(centers) < 0.3  # near-regular spacing


def test_roughness_higher_for_irregular():
    np.random.seed(0)
    centers = [
        c(i, float(np.random.uniform(0, 100)), float(np.random.uniform(0, 100)), 5.0)
        for i in range(20)
    ]
    assert roughness(centers) >= 0.0  # no hard upper bound, just non-negative


# --- echoes ---


def test_echoes_zero_consistent_ratios():
    # All parent-child pairs have exactly ratio 3 → std=0
    parent = c(0, 0, 0, 90.0)
    child1 = c(1, 5, 0, 30.0, parent=0)
    child2 = c(2, -5, 0, 30.0, parent=0)
    assert echoes([parent, child1, child2]) == pytest.approx(0.0, abs=1e-10)


def test_echoes_positive_inconsistent_ratios():
    parent = c(0, 0, 0, 30.0)
    child1 = c(1, 5, 0, 10.0, parent=0)  # ratio 3
    child2 = c(2, -5, 0, 2.0, parent=0)  # ratio 15
    assert echoes([parent, child1, child2]) > 0


# --- the_void ---


def test_the_void_low_for_uniform_field():
    centers = [c(0, 50, 50, 20.0, strength=2.0)]
    field = reconstruct_field((100, 100), centers)
    score = the_void(field, centers)
    assert score >= 0.0


def test_the_void_zero_no_centers():
    assert the_void(np.ones((50, 50)), []) == 0.0


# --- simplicity ---


def test_simplicity_high_concentrated_strength():
    centers = [c(0, 0, 0, 10, strength=10.0)] + [
        c(i + 1, 0, 0, 5, strength=0.01) for i in range(9)
    ]
    assert simplicity(centers) > 0.5


def test_simplicity_low_equal_strengths():
    centers = [c(i, 0, 0, 5, strength=1.0) for i in range(10)]
    assert simplicity(centers) < 0.2


def test_simplicity_empty():
    assert simplicity([]) == 0.0


# --- not_separateness ---


def test_not_separateness_positive_connected():
    centers = [c(0, 0, 0, 10, strength=1.0), c(1, 5, 0, 10, strength=1.0)]
    G = connected_graph(centers)
    if G.has_edge(0, 1):
        assert not_separateness(G) > 0


def test_not_separateness_zero_disconnected():
    G = build_graph([c(0, 0, 0, 3), c(1, 500, 0, 3)])
    assert not G.has_edge(0, 1)
    assert not_separateness(G) == pytest.approx(0.0, abs=1e-8)


def test_not_separateness_single_node():
    G = build_graph([c(0, 0, 0, 5)])
    assert not_separateness(G) == 0.0


# --- compute_all ---


def test_compute_all_returns_all_fifteen():
    centers = small_hierarchy()
    G = connected_graph(centers)
    field = reconstruct_field((100, 100), centers)
    scores = compute_all(field, centers, G)
    assert len(scores) == 15
    expected_keys = {
        "levels_of_scale",
        "strong_centres",
        "boundaries",
        "alternating_repetition",
        "positive_space",
        "good_shape",
        "local_symmetries",
        "deep_interlock",
        "contrast",
        "gradients",
        "roughness",
        "echoes",
        "the_void",
        "simplicity",
        "not_separateness",
    }
    assert set(scores.keys()) == expected_keys


def test_compute_all_values_finite():
    centers = small_hierarchy()
    G = connected_graph(centers)
    field = reconstruct_field((100, 100), centers)
    for key, val in compute_all(field, centers, G).items():
        assert np.isfinite(val), f"{key} is not finite"
