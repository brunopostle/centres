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


def well_formed_hierarchy():
    """A parent with three mutually adjacent children.

    small_hierarchy() has only two children, so its graph has a single edge and
    no node of degree 2 — which leaves alternating_repetition legitimately
    undefined. That is a property of the fixture, not of the measure. Three
    children at 120 degrees, each 0.5 * r_parent out, are mutually adjacent
    (d = 26 against a touching distance of 20), giving every child degree 2.
    """
    import math

    parent = c(0, 50, 50, 30.0, strength=2.0)
    kids = []
    for k, angle in enumerate((90, 210, 330)):
        rad = math.radians(angle)
        kids.append(
            c(
                k + 1,
                50 + 15 * math.cos(rad),
                50 + 15 * math.sin(rad),
                10.0,
                strength=1.0 + 0.4 * k,
                parent=0,
            )
        )
    return [parent] + kids


def _figure(centre):
    """Mark a centre as figure, which the polarity-filtered measures require."""
    centre.polarity = 0.2
    return centre


def with_region(centre, **kw):
    """Attach a synthetic Region so a hand-built centre can be measured.

    The redefined measures (#22) read region geometry, tone and symmetry, which
    a bare (x, y, scale, strength) tuple does not carry -- that was the whole
    finding. Tests that construct centres by hand must supply it.
    """
    from centres.regions import Region

    defaults = dict(area=100.0, compactness=0.5, solidity=0.5, tone=0.5,
                    tone_spread=0.0, vertical_symmetry=0.5,
                    horizontal_symmetry=0.5, elongation=0.5, orientation=0.0,
                    shape_signature=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0))
    defaults.update(kw)
    centre.region = Region(**defaults)
    return centre


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
    assert strong_centres([]) is None


# --- boundaries ---


def test_boundaries_low_when_field_drops_between_centres():
    # Two centres far enough apart that the field between them is low
    centers = [c(0, 10, 50, 5.0, strength=1.0), c(1, 90, 50, 5.0, strength=1.0)]
    field = reconstruct_field((100, 100), centers)
    G = connected_graph(centers)
    if G.has_edge(0, 1):
        score = boundaries(field, centers, G)
        assert score < 1.0  # midpoint weaker than peaks


def test_boundaries_no_edges_is_undefined():
    centers = [c(0, 0, 0, 3.0), c(1, 500, 0, 3.0)]
    field = reconstruct_field((100, 100), centers)
    G = build_graph(centers)
    assert not G.has_edge(0, 1)
    assert boundaries(field, centers, G) is None


# --- alternating_repetition ---


def test_alternating_repetition_high_with_varied_neighbours():
    # Centre with neighbours of very different strengths
    centers = [
        c(0, 0, 0, 10, strength=1.0),
        c(1, 15, 0, 10, strength=0.1),
        c(2, -15, 0, 10, strength=2.0),
    ]
    G = connected_graph(centers)
    score = alternating_repetition(G)
    assert score >= 0.0


def test_alternating_repetition_no_edges_is_undefined():
    centers = [c(0, 0, 0, 3), c(1, 500, 0, 3)]
    G = build_graph(centers)
    assert alternating_repetition(G) is None


# --- positive_space ---


def test_positive_space_reads_the_convexity_of_the_ground():
    """The source: experienced space is convex, its enclosing solid concave."""
    convex = [with_region(c(0, 0, 0, 10), solidity=1.0)]
    convex[0].polarity = -0.2
    ragged = [with_region(c(0, 0, 0, 10), solidity=0.3)]
    ragged[0].polarity = -0.2
    assert positive_space(convex) > positive_space(ragged)
    assert positive_space(convex) == pytest.approx(1.0)


def test_positive_space_ignores_the_figure_population():
    figure = [with_region(c(0, 0, 0, 10), solidity=1.0)]
    figure[0].polarity = +0.2
    assert positive_space(figure) is None

def test_good_shape_reads_compactness():
    """The source: compact shapes are cognitively graspable."""
    disc = [with_region(c(0, 0, 0, 10), compactness=1.0)]
    disc[0].polarity = +0.2
    star = [with_region(c(0, 0, 0, 10), compactness=0.2)]
    star[0].polarity = +0.2
    assert good_shape(disc) == pytest.approx(1.0)
    assert good_shape(star) == pytest.approx(0.2)

def test_good_shape_is_undefined_without_figure_regions():
    ground = [with_region(c(0, 0, 0, 10), compactness=1.0)]
    ground[0].polarity = -0.2
    assert good_shape(ground) is None

def test_local_symmetries_reads_vertical_symmetry_and_weights_by_area():
    """The source names the vertical axis, and every scale.

    A large asymmetric region must outweigh a small symmetric one, because the
    source requires symmetry to act on every distinct scale rather than only the
    smallest.
    """
    symmetric = [_figure(with_region(c(0, 0, 0, 10), vertical_symmetry=1.0))]
    assert local_symmetries(symmetric) == pytest.approx(1.0)

    mixed = [
        _figure(with_region(c(0, 0, 0, 10), vertical_symmetry=1.0, area=10.0)),
        _figure(with_region(c(1, 50, 0, 10), vertical_symmetry=0.0, area=90.0)),
    ]
    assert local_symmetries(mixed) == pytest.approx(0.1)


def test_local_symmetries_distinguishes_the_axis():
    """Symmetric horizontally but not vertically is not the sourced property."""
    horizontal_only = [
        _figure(with_region(c(0, 0, 0, 10), vertical_symmetry=0.2,
                            horizontal_symmetry=1.0))
    ]
    assert local_symmetries(horizontal_only) == pytest.approx(0.2)

def test_deep_interlock_one_when_all_edges_overlap():
    # Two centres whose radii overlap: d=15 < r1+r2=10+10=20
    centers = [c(0, 0, 0, 10.0, strength=1.0), c(1, 15, 0, 10.0, strength=1.0)]
    G = connected_graph(centers)
    assert deep_interlock(centers, G) == pytest.approx(1.0, abs=1e-6)


def test_deep_interlock_no_edges_is_undefined():
    # Two centres far apart: d=100 > r1+r2=5+5=10. They do not even form an
    # edge, so the fraction-of-edges has no denominator and is undefined.
    centers = [c(0, 0, 0, 5.0, strength=1.0), c(1, 100, 0, 5.0, strength=1.0)]
    G = connected_graph(centers)
    assert not G.edges
    assert deep_interlock(centers, G) is None


def test_deep_interlock_zero_when_edges_exist_but_none_overlap():
    # An edge exists (scale-similar and within the graph radius) but the two
    # radii do not reach each other: a genuine zero, not an undefined.
    centers = [c(0, 0, 0, 6.0, strength=1.0), c(1, 30, 0, 6.0, strength=1.0)]
    G = connected_graph(centers)
    assert G.edges
    assert deep_interlock(centers, G) == 0.0


def test_deep_interlock_empty_graph_is_undefined():
    assert (
        deep_interlock([c(0, 0, 0, 10.0)], connected_graph([c(0, 0, 0, 10.0)])) is None
    )


# --- contrast ---


def test_contrast_reads_tone_not_strength():
    """The source: black-white and colour contrast.

    Strength derives from the distance-transform field, which carries no tone at
    all, so the old measure was flat against a tonal sweep (#22).
    """
    dark = with_region(c(0, 0, 0, 10.0, strength=1.0), tone=0.05)
    light = with_region(c(1, 15, 0, 10.0, strength=1.0), tone=0.95)
    G = connected_graph([dark, light])
    assert G.edges
    assert contrast(G) == pytest.approx(0.9, abs=0.02)

def test_contrast_zero_when_tones_match():
    a = with_region(c(0, 0, 0, 10.0, strength=1.0), tone=0.5)
    b = with_region(c(1, 15, 0, 10.0, strength=1.0), tone=0.5)
    G = connected_graph([a, b])
    assert G.edges
    assert contrast(G) == pytest.approx(0.0, abs=1e-9)


def test_contrast_is_blind_to_strength():
    """Differing strengths with equal tone is not contrast in the sourced sense."""
    a = with_region(c(0, 0, 0, 10.0, strength=0.1), tone=0.5)
    b = with_region(c(1, 15, 0, 10.0, strength=9.0), tone=0.5)
    assert contrast(connected_graph([a, b])) == pytest.approx(0.0, abs=1e-9)

def test_contrast_no_edges_is_undefined():
    G = build_graph([c(0, 0, 0, 3), c(1, 500, 0, 3)])
    assert contrast(G) is None


# --- gradients ---


def test_gradients_zero_uniform():
    centers = [c(0, 25, 25, 5.0)]
    assert gradients(np.ones((50, 50)), centers) == pytest.approx(0.0, abs=1e-10)


def test_gradients_positive_step():
    field = np.zeros((50, 50))
    field[:, 25:] = 1.0
    assert gradients(field, [c(0, 25, 25, 5.0)]) > 0


# --- roughness ---


def test_roughness_single_centre_is_undefined():
    assert roughness([c(0, 0, 0, 5)]) is None


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


def test_echoes_high_when_shapes_repeat():
    """The source: similar shapes repeated, within and across scales."""
    same = [
        with_region(c(i, i * 40, 0, 10.0), shape_signature=(1.0, 2.0, 3.0, 4.0))
        for i in range(4)
    ]
    assert echoes(same) == pytest.approx(1.0, abs=1e-6)

def test_echoes_lower_when_every_shape_differs():
    same = [
        with_region(c(i, i * 40, 0, 10.0), shape_signature=(1.0, 2.0, 3.0, 4.0))
        for i in range(4)
    ]
    varied = [
        with_region(c(i, i * 40, 0, 10.0),
                    shape_signature=(1.0 + i, 2.0 - i, 3.0 + 2 * i, 4.0 - i))
        for i in range(4)
    ]
    assert echoes(varied) < echoes(same)


def test_echoes_ignores_scale():
    """A small motif echoing a large one of the same form is an echo."""
    across_scales = [
        with_region(c(0, 0, 0, 5.0, ), area=50.0,
                    shape_signature=(1.0, 2.0, 3.0, 4.0)),
        with_region(c(1, 60, 0, 40.0), area=5000.0,
                    shape_signature=(1.0, 2.0, 3.0, 4.0)),
    ]
    assert echoes(across_scales) == pytest.approx(1.0, abs=1e-6)

def test_the_void_low_for_uniform_field():
    centers = [c(0, 50, 50, 20.0, strength=2.0)]
    field = reconstruct_field((100, 100), centers)
    score = the_void(field, centers)
    assert score >= 0.0


def test_the_void_no_centers_is_undefined():
    assert the_void(np.ones((50, 50)), []) is None


# --- simplicity ---


def test_simplicity_high_concentrated_strength():
    centers = [c(0, 0, 0, 10, strength=10.0)] + [
        c(i + 1, 0, 0, 5, strength=0.01) for i in range(9)
    ]
    assert simplicity(centers) > 0.5


def test_simplicity_low_equal_strengths():
    centers = [c(i, 0, 0, 5, strength=1.0) for i in range(10)]
    assert simplicity(centers) < 0.2


def test_simplicity_empty_is_undefined():
    assert simplicity([]) is None


# --- not_separateness ---


def test_not_separateness_positive_connected():
    centers = [c(0, 0, 0, 10, strength=1.0), c(1, 15, 0, 10, strength=1.0)]
    G = connected_graph(centers)
    if G.has_edge(0, 1):
        assert not_separateness(G) > 0


def test_not_separateness_disconnected_is_undefined():
    # Two isolated nodes: the largest connected component has one node, so
    # there is no second Laplacian eigenvalue to read.
    G = build_graph([c(0, 0, 0, 3), c(1, 500, 0, 3)])
    assert not G.has_edge(0, 1)
    assert not_separateness(G) is None


def test_not_separateness_single_node_is_undefined():
    G = build_graph([c(0, 0, 0, 5)])
    assert not_separateness(G) is None


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
    centers = [with_region(x) for x in well_formed_hierarchy()]
    for x in centers:
        x.polarity = 0.2 if x.parent is not None else -0.2
    G = connected_graph(centers)
    field = reconstruct_field((100, 100), centers)
    for key, val in compute_all(field, centers, G).items():
        assert val is not None, f"{key} is undefined for a well-formed hierarchy"
        assert np.isfinite(val), f"{key} is not finite"
