import os

import cv2
import numpy as np
import pytest

from centres.centers import Center
from centres.field import build_structural_field
from centres.graph import build_graph, propagate_strength
from centres.pipeline import assign_hierarchy, detect_centers
from centres.properties import contrast, normalize_all, strong_centres

IMAGES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")

RAW_KEYS = [
    "levels_of_scale", "strong_centres", "boundaries", "alternating_repetition",
    "positive_space", "good_shape", "local_symmetries", "deep_interlock",
    "contrast", "gradients", "roughness", "echoes", "the_void",
    "simplicity", "not_separateness",
]


def c(id, x, y, scale, strength=0.5):
    return Center(id=id, x=float(x), y=float(y), scale=float(scale), strength=strength)


def grid_centers(n=3, spacing=20, scale=10, strength=0.5, start_id=0):
    """An n x n block of mutually connected, scale-similar centres.

    Spacing equals the touching distance (r_i + r_j = 20), where the adjacency
    kernel peaks. 8px would put them at 0.4x touching -- deeply overlapping, and
    barely connected under a kernel that treats superposition as identity.
    """
    return [
        c(start_id + i * n + j, i * spacing, j * spacing, scale, strength)
        for i in range(n)
        for j in range(n)
    ]


# --- build_graph ---


def test_nearby_same_scale_connected():
    G = build_graph([c(0, 0, 0, 10), c(1, 15, 0, 10)])
    assert G.has_edge(0, 1)


def test_distant_centres_not_connected():
    # dist=500 >> 3 * mean_scale=9 — weight will be negligible
    G = build_graph([c(0, 0, 0, 3), c(1, 500, 0, 3)])
    assert not G.has_edge(0, 1)


def test_very_different_scales_reduce_weight():
    # scale ratio of 10x — scale_term = exp(-(log10)^2) ≈ 0.005
    G = build_graph([c(0, 0, 0, 5), c(1, 3, 0, 50)])
    if G.has_edge(0, 1):
        assert G[0][1]["weight"] < 0.1


def test_edge_weight_between_zero_and_one():
    G = build_graph([c(0, 0, 0, 10), c(1, 15, 0, 10)])
    for _, _, data in G.edges(data=True):
        assert 0.0 < data["weight"] <= 1.0


def test_empty_input():
    G = build_graph([])
    assert len(G.nodes) == 0
    assert len(G.edges) == 0


def test_single_centre_no_edges():
    G = build_graph([c(0, 0, 0, 10)])
    assert len(G.nodes) == 1
    assert len(G.edges) == 0


def test_node_stores_center():
    centre = c(0, 10.0, 20.0, 5.0)
    G = build_graph([centre])
    assert G.nodes[0]["center"] is centre


# --- propagate_strength ---


def test_connected_centres_stay_strong():
    # Two adjacent high-strength centres — reinforcement should sustain them
    G = build_graph([c(0, 0, 0, 10, strength=0.9), c(1, 15, 0, 10, strength=0.9)])
    assert G.has_edge(0, 1)
    G = propagate_strength(G)
    assert G.nodes[0]["center"].strength > 0.5
    assert G.nodes[1]["center"].strength > 0.5


def test_isolated_centre_keeps_intrinsic_strength():
    # No edges — no neighbours to reinforce with — the centre keeps exactly the
    # intrinsic strength it carried in from the field. It is neither reinforced
    # nor penalised. (Previously it decayed as (1 - beta)^steps while connected
    # centres grew as 1.15^steps, so the gap between isolated and connected
    # centres was a function of the step count rather than of the image.)
    G = build_graph([c(0, 0, 0, 3, strength=0.8), c(1, 500, 0, 3, strength=0.8)])
    assert not G.has_edge(0, 1)
    G = propagate_strength(G, steps=10)
    assert G.nodes[0]["center"].strength == pytest.approx(0.8)
    assert G.nodes[1]["center"].strength == pytest.approx(0.8)


def test_strengths_clipped_positive():
    G = build_graph([c(0, 0, 0, 10, strength=0.5), c(1, 15, 0, 10, strength=0.5)])
    G = propagate_strength(G)
    for n in G.nodes:
        assert G.nodes[n]["center"].strength >= 0.0


def test_empty_graph_no_error():
    G = build_graph([])
    G = propagate_strength(G)  # should not raise
    assert len(G.nodes) == 0


# --- propagate_strength: stationarity (issue #12) ---

STEP_COUNTS = [5, 10, 20, 40, 100]


def _load(name, max_size=512):
    img = cv2.imread(os.path.join(IMAGES, name))
    assert img is not None, f"missing test image {name}"
    h, w = img.shape[:2]
    s = min(max_size / max(h, w), 1.0)
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def test_fixed_point_reached_regardless_of_step_count():
    """The update is a contraction, so the strengths themselves are stationary."""
    centers = grid_centers() + [c(9, 5000, 5000, 10, 0.5)]
    results = []
    for steps in STEP_COUNTS:
        cs = [c(x.id, x.x, x.y, x.scale, 0.5) for x in centers]
        G = propagate_strength(build_graph(cs), steps=steps)
        results.append(np.array([G.nodes[n]["center"].strength for n in G.nodes]))
    # At steps=5 the iteration cap binds before tol, leaving a residual of
    # order alpha^5 ~ 3e-4; every larger step count is converged to tol.
    for r in results[1:]:
        assert np.allclose(r, results[0], atol=1e-3)
    for r in results[2:]:
        assert np.allclose(r, results[1], atol=1e-9)


@pytest.mark.parametrize("name", ["ardabil.jpg", "bidjar.jpg"])
def test_strong_centres_and_contrast_stable_across_step_counts(name):
    """Issue #12 criterion 1.

    With the image, the field and the centre set held completely fixed, the
    normalised strong_centres and contrast scores must not depend on the
    propagation step count. Before the fix these ran 1.0 -> 1.9 -> 7.4 -> 10.0
    and 4.8 -> 5.4 -> 10.0 -> 0.0 across these step counts, because the update
    had gain 1.15 per step and no fixed point.
    """
    field = build_structural_field(_load(name))
    detected = assign_hierarchy(detect_centers(field))
    assert len(detected) > 20, "expected a non-trivial centre set"

    scores = {"strong_centres": [], "contrast": []}
    for steps in STEP_COUNTS:
        centers = [
            Center(id=x.id, x=x.x, y=x.y, scale=x.scale, strength=x.strength, parent=x.parent)
            for x in detected
        ]
        G = propagate_strength(build_graph(centers), steps=steps)
        raw = {k: 0.0 for k in RAW_KEYS}
        raw["strong_centres"] = strong_centres(centers)
        raw["contrast"] = contrast(G)
        norm = normalize_all(raw)
        for k in scores:
            scores[k].append(norm[k])

    for k, vals in scores.items():
        spread = max(vals) - min(vals)
        assert spread < 0.5, f"{name}: {k} varies by {spread:.3f} across steps {STEP_COUNTS}: {vals}"


def test_clustered_centres_stronger_than_isolated():
    """Issue #12 criterion 2.

    A fixed point is not enough: it must still rank centres meaningfully. A
    scheme that flattened every strength to a constant would satisfy criterion 1
    while destroying the measure. With identical intrinsic strengths, centres
    embedded in a dense cluster of scale-similar neighbours must end up stronger
    than an isolated centre, and the most embedded centre strongest of all.
    """
    cluster = grid_centers(n=3, strength=0.5)
    isolated = c(9, 5000, 5000, 10, 0.5)
    G = build_graph(cluster + [isolated])
    assert G.degree(9) == 0
    propagate_strength(G)

    clustered = [x.strength for x in cluster]
    assert min(clustered) > isolated.strength
    # the centre of the 3x3 block is the most embedded, so the strongest
    assert cluster[4].strength == pytest.approx(max(clustered))
    # and the strengths must not have collapsed to a constant
    assert np.std(clustered + [isolated.strength]) > 1e-3


def test_strengths_bounded_by_contraction():
    """s* is bounded by max(s0) / (1 - alpha) — nothing can run away."""
    centers = grid_centers(n=4, strength=1.0)
    G = propagate_strength(build_graph(centers), steps=100, alpha=0.2)
    s = np.array([G.nodes[n]["center"].strength for n in G.nodes])
    assert s.max() <= 1.0 / (1 - 0.2) + 1e-9
    assert s.min() >= 1.0 - 1e-9


def test_nodes_keyed_by_position_not_id():
    """Regression for #25: node keys must not depend on Center.id.

    build_graph previously added nodes by c.id but edges by list index, so a
    list whose ids were not 0..n-1 produced phantom nodes carrying the edges
    while every real centre was left isolated.
    """
    centers = [
        Center(id=5, x=0.0, y=0.0, scale=10.0, strength=1.0),
        Center(id=7, x=15.0, y=0.0, scale=10.0, strength=1.0),
    ]
    G = build_graph(centers)
    assert len(G.nodes) == len(centers)
    assert all("center" in G.nodes[n] for n in G.nodes)
    assert list(G.edges) == [(0, 1)]
    assert {G.nodes[n]["center"].id for n in G.nodes} == {5, 7}


def test_nodes_keyed_by_position_survives_filtering():
    """A filtered centre list keeps working — the case #8 and #26 will hit."""
    centers = [
        Center(id=i, x=float(i * 12), y=0.0, scale=10.0, strength=1.0) for i in range(6)
    ]
    kept = [c for i, c in enumerate(centers) if i % 2 == 0]  # ids 0, 2, 4
    G = build_graph(kept)
    assert len(G.nodes) == 3
    assert all("center" in G.nodes[n] for n in G.nodes)
    G = propagate_strength(G)
    assert all(G.nodes[n]["center"].strength > 0 for n in G.nodes)
