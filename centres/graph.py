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


def propagate_strength(G, steps=50, alpha=0.2, tol=1e-9):
    """Reinforce centre strengths through the graph, to a stationary point.

    Update rule: s ← s₀ + α(Ŵs),  Ŵ = W / max_i Σ_j W_ij

    where s₀ is the intrinsic strength each centre carries in from the
    structural field, W is the raw edge-weight matrix, and Ŵ is W scaled by
    its largest row sum. Iterated to convergence; `steps` is a maximum
    iteration cap rather than a tuning constant.

    This is Katz/Bonacich reinforcement seeded by the field: a centre's
    strength is its own evidence plus a geometrically discounted sum of the
    strength reaching it along every walk in the graph,

        s* = (I - αŴ)⁻¹ s₀ = s₀ + αŴs₀ + α²Ŵ²s₀ + …

    Because ‖αŴ‖_∞ = α < 1 the map is a contraction, so the fixed point
    exists, is unique, and is reached geometrically at rate α — roughly 13
    iterations for α = 0.2 to reach tol = 1e-9. Strengths are bounded by
    max(s₀)/(1 - α), so nothing can run away and no clip is needed.

    The previous rule, s ← (1 - β)s + α(Ws) against a row-stochastic W, had
    gain (1 - β) + α = 1.15 per step on a uniform vector. It had no fixed
    point: strengths grew as 1.15^t until they saturated the clip at 10, which
    made strong_centres, contrast, alternating_repetition, simplicity and the
    E_R energy term readouts of the iteration counter rather than of the image.

    Note that a plain row-stochastic diffusion cannot be repaired by setting
    α + (1 - β) = 1, nor by renormalising after each step. Both do give a fixed
    point, but it is the *consensus* vector — the leading right eigenvector of
    a row-stochastic matrix is uniform, so every centre converges to the same
    strength and contrast collapses to zero. Retaining the s₀ source term is
    what keeps the stationary distribution informative, and leaving W
    un-normalised (scaled globally rather than per row) is what lets density
    matter: row normalisation would erase degree, so a centre with ten strong
    neighbours would score the same as one with a single strong neighbour.

    Isolated centres keep their intrinsic strength exactly: with no edges their
    row of Ŵ is zero, so s* = s₀. They are neither reinforced nor penalised,
    which is the honest reading of "no neighbours to reinforce with". Under the
    old rule they decayed as 0.95^t while connected centres grew as 1.15^t, so
    the gap between isolated and connected centres was itself a function of the
    step count.
    """
    if len(G.nodes) == 0:
        return G
    s0 = np.array([G.nodes[n]["center"].strength for n in G.nodes], dtype=float)
    W = nx.to_numpy_array(G)
    scale = W.sum(axis=1).max()
    if scale > 0:
        W = W / scale
    strengths = s0.copy()
    for _ in range(steps):
        updated = s0 + alpha * (W @ strengths)
        delta = np.abs(updated - strengths).max()
        strengths = updated
        if delta < tol:
            break
    for i, n in enumerate(G.nodes):
        G.nodes[n]["center"].strength = float(strengths[i])
    return G
