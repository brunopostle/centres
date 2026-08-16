"""Measures for Alexander's 15 structural properties.

Each function returns a scalar derived from the already-computed field,
centre set, and reinforcement graph. Where a property maps directly onto
one of the energy terms it is used directly. Where an additional metric
is needed it is noted as a proxy.

Direction is indicated in each docstring:
  ↓  lower value = property more present
  ↑  higher value = property more present
  ~  no single direction; interpretation is noted

Some properties (alternating repetition, roughness) are genuinely hard to
measure cleanly from a centre field; those are marked as approximations.

**Undefined results.** A measure returns ``None`` when the inputs it needs do
not exist — no centres, no graph edges, no parent-child pairs, and so on. It
does *not* substitute 0, because for the seven deviation-based measures a raw 0
is the ideal value and ``normalize_all`` maps it to a perfect 10. That made a
featureless grey canvas, which detects no centres at all, score 10/10 on levels
of scale, boundaries, positive space, local symmetries, gradients, echoes and
the void — the tool could not tell *no structure* from *ideal structure*.

``None`` propagates through ``normalize_all`` unchanged, and every output path
renders it as "undefined" rather than as a number. Each function's docstring
states where its own boundary lies; the boundary differs per measure and is not
a single blanket rule.
"""

import numpy as np
import networkx as nx

from .energy import hierarchy_energy, coverage_energy, alignment_energy, field_energy


def levels_of_scale(centers):
    """↓  Mean per-pair E_H: deviation from scale ratio ~3 between parent and child.

    Mean of (log(r_parent/r_child) - log 3)² across all parent-child pairs.
    Zero when every pair has exactly a 3× scale ratio. Normalised per pair
    so the score is comparable across centre sets of different sizes.

    Undefined (``None``) with no parent-child pairs: there is no hierarchy, so
    there is no scale ratio to deviate from. One pair is enough — the mean of a
    single deviation is a perfectly good deviation.
    """
    pairs = [c for c in centers if c.parent is not None]
    if not pairs:
        return None
    return hierarchy_energy(centers) / len(pairs)


def strong_centres(centers):
    """↑  Mean strength of the top-quartile centres after reinforcement propagation.

    High value means dominant focal centres have emerged from the
    reinforcement dynamics.

    Undefined (``None``) with no centres: there is no strength distribution to
    take a top quartile of. A single centre is its own top quartile, which is a
    defined if uninformative answer.
    """
    if not centers:
        return None
    s = np.array([c.strength for c in centers])
    return float(s[s >= np.percentile(s, 75)].mean())


def boundaries(field, centers, G):
    """↓  Mean ratio of field value at midpoints between connected centres to
    the average of their peak values.

    Low ratio means a clear drop in the wholeness field between adjacent
    centres — the signature of a defined boundary zone separating them.

    Undefined (``None``) with no graph edges: a boundary is a thing between two
    adjacent centres, so with nothing adjacent there is no boundary to measure.
    Also undefined when edges exist but every pair's peak field value is ~0, so
    no ratio can be formed — the reconstruction has nothing there to drop from.
    """
    if not G.edges:
        return None
    h, w = field.shape
    ratios = []
    for i, j, _ in G.edges(data=True):
        c1 = G.nodes[i]["center"]
        c2 = G.nodes[j]["center"]
        mx = int(np.clip((c1.x + c2.x) / 2, 0, w - 1))
        my = int(np.clip((c1.y + c2.y) / 2, 0, h - 1))
        p1 = field[int(np.clip(c1.y, 0, h - 1)), int(np.clip(c1.x, 0, w - 1))]
        p2 = field[int(np.clip(c2.y, 0, h - 1)), int(np.clip(c2.x, 0, w - 1))]
        peak = (p1 + p2) / 2
        if peak > 1e-8:
            ratios.append(field[my, mx] / peak)
    return float(np.mean(ratios)) if ratios else None


def alternating_repetition(G):
    """↑  Mean standard deviation of strengths across each centre's neighbours.

    Approximation. Alternating repetition produces systematic strength
    alternation between adjacent centres. High neighbour-strength variance
    indicates this pattern.

    Undefined (``None``) when no centre has at least two neighbours: the
    quantity averaged is a standard deviation *across a centre's neighbours*,
    and one neighbour cannot alternate with anything. This is a stricter
    condition than "the graph has edges" — a graph of isolated pairs has edges
    but every node has degree 1, and there is genuinely no alternation there.
    """
    stds = []
    for n in G.nodes:
        nbrs = list(G.neighbors(n))
        if len(nbrs) < 2:
            continue
        stds.append(np.std([G.nodes[m]["center"].strength for m in nbrs]))
    return float(np.mean(stds)) if stds else None


def positive_space(centers):
    """↓  Mean per-parent E_C: deviation from ideal child coverage (~0.65).

    Mean of (C_i - 0.65)² across all centres that have children. Zero when
    every parent's children cover exactly 65% of the parent area. Normalised
    per parent so the score is comparable across different centre counts.

    Undefined (``None``) with no parents: coverage is the area of a parent
    filled by its children, so with no parent-child relation there is no
    coverage. One parent is enough.
    """
    parents = {c.parent for c in centers if c.parent is not None}
    if not parents:
        return None
    return coverage_energy(centers) / len(parents)


def good_shape(centers):
    """↑  Fraction of centres that contain at least one child in the hierarchy.

    Centres with sub-structure are interpreted as having sufficient internal
    coherence to constitute a 'good shape'. Pure leaf nodes — centres with
    no children — are not forming enclosing regions at their scale.

    Undefined (``None``) with no centres — the denominator is the centre count.
    Note the boundary is *not* "no parent-child pairs": with centres present but
    no hierarchy the fraction is a genuine 0, meaning no centre has
    sub-structure, which is a real and low result rather than a missing one.
    """
    if not centers:
        return None
    parents_with_children = {c.parent for c in centers if c.parent is not None}
    return len(parents_with_children) / len(centers)


def local_symmetries(centers):
    """↓  Mean per-pair E_A: deviation of child distance from 0.5 × r_parent.

    Mean of (d/r_parent - 0.5)² across all parent-child pairs. Measures
    alignment of children within their parent. Normalised per pair so the
    score is comparable across centre sets of different sizes.

    Undefined (``None``) with no parent-child pairs: the measured quantity is a
    child's radial position within its parent, which needs a pair to exist.
    """
    pairs = [c for c in centers if c.parent is not None]
    if not pairs:
        return None
    return alignment_energy(centers) / len(pairs)


def deep_interlock(centers, G):
    """↑  Fraction of connected centre pairs whose spatial extents overlap.

    Two centres interlock when d(i, j) < r_i + r_j — each extends into the
    territory of the other, creating the interwoven boundary structure
    Alexander called deep interlock. Measured over reinforcement graph edges
    rather than parent-child pairs, since LoG hierarchy places children
    outside parent radii (the hierarchy describes scale nesting, not spatial
    containment at the individual-blob level).

    Undefined (``None``) with no graph edges: the value is a fraction *of the
    edges*, so with no edges the denominator is zero. One edge is enough.
    """
    if not G.edges:
        return None
    overlap = sum(
        1
        for i, j in G.edges()
        if np.sqrt(
            (G.nodes[i]["center"].x - G.nodes[j]["center"].x) ** 2
            + (G.nodes[i]["center"].y - G.nodes[j]["center"].y) ** 2
        )
        < G.nodes[i]["center"].scale + G.nodes[j]["center"].scale
    )
    return overlap / G.number_of_edges()


def contrast(G):
    """↑  Mean absolute strength difference between connected centres,
    weighted by edge weight.

    Measures whether adjacent centres differ in strength. Zero when all
    connected centres have equal strength (flat, undifferentiated field).

    Undefined (``None``) with no graph edges: contrast is between adjacent
    centres, so with nothing adjacent there is no difference to take. Note the
    distinction from the zero case above, which the old return of 0.0 conflated:
    equal strengths across real edges is genuinely minimal contrast.
    """
    if not G.edges:
        return None
    total, w_sum = 0.0, 0.0
    for i, j, data in G.edges(data=True):
        w = data["weight"]
        total += w * abs(G.nodes[i]["center"].strength - G.nodes[j]["center"].strength)
        w_sum += w
    return total / (w_sum + 1e-8)


def gradients(field, centers):
    """↓  E_φ: mean squared gradient of the wholeness field.

    Directly measures smooth directional transitions across the field.
    Low value = gentle gradients between regions rather than abrupt jumps.

    Undefined (``None``) with no centres. The field this reads is the
    *reconstructed* field — a sum of Gaussians placed at the detected centres,
    not the image (see AUDIT.md §10). With no centres it is identically zero
    everywhere, so its mean squared gradient is 0 by construction, which
    ``normalize_all`` would report as a perfect 10. There are no regions, so
    there are no transitions between regions to be gentle or abrupt. ``centers``
    is required rather than optional so a caller cannot silently get the old
    behaviour back by omitting it.
    """
    if not centers:
        return None
    return field_energy(field)


def roughness(centers):
    """~  Coefficient of variation of nearest-neighbour distances between centres.

    Approximation. Pure regularity (grid) gives values near zero; natural
    irregularity within order gives moderate values; disorder gives high values.
    Alexander's roughness is the moderate, not-perfectly-regular case.

    Undefined (``None``) with fewer than two centres: a lone centre has no
    nearest neighbour, so there is no spacing distribution. Two centres do give
    a distribution (a degenerate one, CV = 0), which is defined.
    """
    if len(centers) < 2:
        return None
    from scipy.spatial.distance import cdist

    pos = np.array([[c.x, c.y] for c in centers])
    dist = cdist(pos, pos)
    np.fill_diagonal(dist, np.inf)
    nn = dist.min(axis=1)
    return float(nn.std() / (nn.mean() + 1e-8))


def echoes(centers):
    """↓  Standard deviation of log-scale ratios between parent-child pairs.

    Low value means the same scale ratio recurs consistently across all
    levels of the hierarchy — the mathematical signature of self-similar
    echoes (patterns that repeat at different scales).

    Undefined (``None``) with fewer than two parent-child pairs. This is the one
    measure whose boundary is at two rather than one: a single ratio has a
    population standard deviation of exactly 0, and 0 is this measure's ideal
    value, so one pair would be reported as perfect self-similarity. An echo is
    a relation between at least two occurrences.
    """
    ratios = [
        np.log(centers[c.parent].scale / (c.scale + 1e-8))
        for c in centers
        if c.parent is not None
    ]
    return float(np.std(ratios)) if len(ratios) > 1 else None


def the_void(field, centers):
    """↓  Mean gradient magnitude within the radius of the strongest centre.

    Low value means there is a calm, undifferentiated region at the heart
    of the dominant centre — Alexander's void, the still point that the
    rest of the composition organises around.

    Undefined (``None``) with no centres: there is no dominant centre whose
    interior could be calm. Also undefined when the strongest centre's disc
    covers no pixel of the field — it lies off-frame, or its scale is
    sub-pixel — so there is nothing to average a gradient over.
    """
    if not centers:
        return None
    strongest = max(centers, key=lambda c: c.strength)
    h, w = field.shape
    Y, X = np.mgrid[0:h, 0:w]
    mask = (X - strongest.x) ** 2 + (Y - strongest.y) ** 2 < strongest.scale**2
    if not mask.any():
        return None
    gy, gx = np.gradient(field)
    return float(np.sqrt(gx**2 + gy**2)[mask].mean())


def simplicity(centers):
    """↑  Gini coefficient of centre strengths.

    High value means a few dominant centres among many weaker ones —
    the structural clarity Alexander called simplicity and inner calm.
    Low value means all centres are equally strong, which indicates
    complexity without hierarchy rather than coherence.

    Undefined (``None``) with no centres: a Gini coefficient needs a
    population. One centre gives 0, no inequality, which is defined.
    """
    if not centers:
        return None
    s = np.sort([c.strength for c in centers])
    n = len(s)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * s).sum() / (n * s.sum() + 1e-8)) - (n + 1) / n)


def not_separateness(G):
    """↑  Fiedler value: second-smallest eigenvalue of the graph Laplacian,
    computed on the largest connected component.

    Measures algebraic connectivity of the reinforcement graph. Higher value
    means the centre network is tightly integrated as a single coherent system
    rather than a collection of isolated parts.

    The full graph is typically near-disconnected (several isolated nodes),
    which forces the global Fiedler value to zero and makes it uninformative.
    Restricting to the largest connected component gives the connectivity of
    the main structural network.

    Undefined (``None``) unless the largest connected component has at least two
    nodes. The Fiedler value is the *second*-smallest Laplacian eigenvalue, so a
    one-node component has no second eigenvalue to read. Because the measure is
    defined on the LCC rather than the whole graph, "fewer than two nodes
    overall" is not the right test: a graph of isolated nodes has many nodes and
    an LCC of one, and its algebraic connectivity is not zero-but-measured, it
    is unmeasurable.
    """
    if len(G.nodes) < 2:
        return None
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    LCC = G.subgraph(components[0])
    if len(LCC) < 2:
        return None
    L = nx.laplacian_matrix(LCC, weight="weight").toarray()
    return float(np.linalg.eigvalsh(L)[1])


def compute_all(field, centers, G):
    """Return raw scores for all 15 of Alexander's structural properties.

    A value is ``None`` where that property is undefined for these inputs —
    see the module docstring and each measure's own docstring for where its
    boundary lies. Callers must handle ``None``; it is never a number.
    """
    return {
        "levels_of_scale": levels_of_scale(centers),
        "strong_centres": strong_centres(centers),
        "boundaries": boundaries(field, centers, G),
        "alternating_repetition": alternating_repetition(G),
        "positive_space": positive_space(centers),
        "good_shape": good_shape(centers),
        "local_symmetries": local_symmetries(centers),
        "deep_interlock": deep_interlock(centers, G),
        "contrast": contrast(G),
        "gradients": gradients(field, centers),
        "roughness": roughness(centers),
        "echoes": echoes(centers),
        "the_void": the_void(field, centers),
        "simplicity": simplicity(centers),
        "not_separateness": not_separateness(G),
    }


def normalize_all(raw):
    """Map raw property scores to 0–10 wholeness scores (10 = most present).

    All 15 scores use the same direction: higher = more of Alexander's property.
    The transformation for each property is either:
      - exp(-k·x) for ↓ properties (penalise positive raw values, ideal = 0)
      - min(x / ref, 1) · 10 for ↑ properties with a reference saturation level
      - exp(-(x-opt)² / σ²) · 10 for the ~ roughness property (ideal at 0.5)

    Scale parameters are set so that typical real-image values span roughly 3–8,
    leaving room at both ends for ideally wholesome or very poor configurations.

    A raw value of ``None`` — the property is undefined for these inputs — maps
    to ``None``, not to a number. This is the whole point of the ``None``
    contract: for the ↓ properties ``decay(0) = 10`` and for boundaries
    ``10 * (1 - 0) = 10``, so any substituted default would be read as a perfect
    score.
    """

    def decay(x, k):
        return 10.0 * float(np.exp(-k * max(x, 0)))

    def rise(x, ref):
        return 10.0 * min(float(x) / ref, 1.0)

    def roughness_peak(x):
        return 10.0 * float(np.exp(-((x - 0.5) ** 2) / 0.04))

    transforms = {
        # ↓ properties — lower raw = better
        "levels_of_scale": lambda x: decay(x, 2.0),
        "boundaries": lambda x: 10.0 * (1.0 - float(x)),
        "positive_space": lambda x: decay(x, 7.0),
        "local_symmetries": lambda x: decay(x, 0.25),
        "gradients": lambda x: decay(x, 4000.0),
        "echoes": lambda x: decay(x, 2.0),
        "the_void": lambda x: decay(x, 40.0),
        # ↑ properties — higher raw = better
        "strong_centres": lambda x: rise(x, 10.0),
        "alternating_repetition": lambda x: rise(x, 0.2),
        "good_shape": lambda x: rise(x, 1.0),
        "deep_interlock": lambda x: rise(x, 1.0),
        "contrast": lambda x: rise(x, 0.2),
        "simplicity": lambda x: rise(x, 1.0),
        "not_separateness": lambda x: rise(x, 0.02),
        # ~ roughness — ideal at moderate irregularity, peak at 0.5
        "roughness": roughness_peak,
    }

    return {
        key: (None if raw[key] is None else fn(raw[key]))
        for key, fn in transforms.items()
    }
