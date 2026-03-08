from centres.centers import Center
from centres.graph import build_graph, propagate_strength


def c(id, x, y, scale, strength=0.5):
    return Center(id=id, x=float(x), y=float(y), scale=float(scale), strength=strength)


# --- build_graph ---


def test_nearby_same_scale_connected():
    G = build_graph([c(0, 0, 0, 10), c(1, 5, 0, 10)])
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
    G = build_graph([c(0, 0, 0, 10), c(1, 5, 0, 10)])
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
    # Two high-strength centres close together — reinforcement should sustain them
    G = build_graph([c(0, 0, 0, 10, strength=0.9), c(1, 5, 0, 10, strength=0.9)])
    assert G.has_edge(0, 1)
    G = propagate_strength(G)
    assert G.nodes[0]["center"].strength > 0.5
    assert G.nodes[1]["center"].strength > 0.5


def test_isolated_centre_decays():
    # No edges — no reinforcement — strength decays by factor (1 - beta) per step
    G = build_graph([c(0, 0, 0, 3, strength=0.8), c(1, 500, 0, 3, strength=0.8)])
    assert not G.has_edge(0, 1)
    G = propagate_strength(G, steps=10, beta=0.05)
    assert G.nodes[0]["center"].strength < 0.8


def test_strengths_clipped_positive():
    G = build_graph([c(0, 0, 0, 10, strength=0.5), c(1, 5, 0, 10, strength=0.5)])
    G = propagate_strength(G)
    for n in G.nodes:
        assert G.nodes[n]["center"].strength >= 0.0


def test_empty_graph_no_error():
    G = build_graph([])
    G = propagate_strength(G)  # should not raise
    assert len(G.nodes) == 0
