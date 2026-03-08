import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist


def build_graph(centers):
    """Build reinforcement graph with scale-relative spatial decay.

    Edge weight combines:
      - spatial proximity: Gaussian with sigma = 3 * mean_scale of the pair,
        so the interaction radius scales with centre size rather than being
        fixed at 50px regardless of image size.
      - log-scale similarity: centres of similar scale reinforce each other more.

    Edges with weight < 0.1 are dropped.
    """
    G = nx.Graph()
    for c in centers:
        G.add_node(c.id, center=c)
    if len(centers) < 2:
        return G
    positions = np.array([[c.x, c.y] for c in centers])
    scales = np.array([c.scale for c in centers])
    dist = cdist(positions, positions)
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            mean_scale = (scales[i] + scales[j]) / 2
            spatial = np.exp(-(dist[i, j] ** 2) / (2 * (3 * mean_scale) ** 2))
            scale_term = np.exp(-((np.log(scales[i]) - np.log(scales[j])) ** 2))
            w = spatial * scale_term
            if w > 0.1:
                G.add_edge(i, j, weight=w)
    return G


def propagate_strength(G, steps=10, alpha=0.2, beta=0.05):
    """Diffuse strength through the reinforcement graph.

    Update rule: s ← (1 - β)s + α(Ws)
    where W is the row-normalised adjacency matrix.

    Previously the pre-computed reinforcement vector was discarded and W@s
    recomputed on the same line; now the cached value is used.
    """
    if len(G.nodes) == 0:
        return G
    strengths = np.array([G.nodes[n]["center"].strength for n in G.nodes])
    W = nx.to_numpy_array(G)
    row_sum = W.sum(axis=1, keepdims=True) + 1e-8
    W = W / row_sum
    for _ in range(steps):
        reinforcement = W @ strengths
        strengths = (1 - beta) * strengths + alpha * reinforcement
        strengths = np.clip(strengths, 0, 10)
    for i, n in enumerate(G.nodes):
        G.nodes[n]["center"].strength = strengths[i]
    return G
