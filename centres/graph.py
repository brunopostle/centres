import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist

#: Width, in log-distance units, of the reinforcement kernel about adjacency.
#: 0.5 admits partners separated by roughly 0.34x to 2.9x their combined radius
#: at the EDGE_THRESHOLD below.
ADJACENCY_WIDTH = 0.5

#: Edges weaker than this are dropped.
EDGE_THRESHOLD = 0.1


def build_graph(centers):
    """Build the reinforcement graph. Weight peaks where centres are adjacent.

    Edge weight combines:
      - adjacency: a Gaussian in log(d / (r_i + r_j)), peaking where the two
        centres touch, and falling to zero both as they coincide and as they
        separate.
      - log-scale similarity: centres of similar scale reinforce each other more.

    Edges with weight < EDGE_THRESHOLD are dropped.

    **Why the kernel peaks at adjacency rather than at coincidence.** The
    previous weight was a Gaussian in raw distance, ``exp(-d^2 / (2 (3 r_bar)^2))``,
    which is *maximal at d = 0*. It therefore rewarded two centres for being the
    same centre, and made the collapsed configuration — every centre at one point,
    at one scale — the global optimum of the energy. Measured on 40 centres in
    300x300, collapse had the most negative reinforcement of any configuration
    tested (-0.391, against -0.154 for a lattice), and only a large locality
    penalty held it up.

    That penalty was a crutch for this kernel. Alexander's centres reinforce one
    another by *adjacency, nesting and interlock*; two superimposed centres are
    one centre, and there is nothing there to reinforce. With the peak moved to
    contact, a lattice earns more reinforcement than a collapse does (-0.216
    against -0.161), which is what the term was always meant to express.

    Working in ``log(d / (r_i + r_j))`` rather than in pixels does two things: it
    sends the weight to zero as d -> 0, and it makes the kernel depend only on
    separation *relative to* the pair's own size, per the scale invariant
    documented in ``centres/field.py``.

    Nodes are keyed by **list position**, not by ``Center.id``. Everything else
    in the pipeline already addresses centres positionally — ``Center.parent`` is
    an index, and ``energy.py`` and ``properties.py`` both resolve it with
    ``centers[c.parent]`` — so position is the codebase's actual convention and
    ``Center.id`` is informational only.

    Keying nodes by ``c.id`` while keying edges by position was a latent bug: any
    centre list whose ids were not exactly ``0..n-1`` in order produced phantom
    nodes carrying the edges, with every real centre left isolated. It never fired
    in production because ``detect_centers`` and ``random_centers`` both assign
    ``id=i``, but it would have fired the moment a caller filtered the list.
    """
    G = nx.Graph()
    for i, c in enumerate(centers):
        G.add_node(i, center=c)
    if len(centers) < 2:
        return G
    positions = np.array([[c.x, c.y] for c in centers])
    scales = np.array([c.scale for c in centers])
    dist = cdist(positions, positions)
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            d = dist[i, j]
            if d <= 0:
                continue  # coincident centres are one centre, not two
            touching = scales[i] + scales[j]
            adjacency = np.exp(
                -(np.log(d / touching) ** 2) / (2 * ADJACENCY_WIDTH**2)
            )
            scale_term = np.exp(-((np.log(scales[i]) - np.log(scales[j])) ** 2))
            w = adjacency * scale_term
            if w > EDGE_THRESHOLD:
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
