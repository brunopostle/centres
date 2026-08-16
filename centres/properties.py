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



def _regions(centers, polarity=None):
    """Regions worth measuring, optionally restricted to one polarity.

    ``polarity`` is ``+1`` for figure (interiors darker than their surround),
    ``-1`` for ground, or ``None`` for both. The distinction is Alexander's own:
    the source names centres of two kinds, "defined" with something in the middle
    and "implied" where a boundary focuses attention on an emptier interior.
    """
    kept = [c for c in centers if c.region is not None and c.region.area > 0]

    # No area floor. Excluding regions below some fraction of the median was
    # tried and reverted: it deletes the smallest scale level, and the presence
    # of structure at every scale is the first of the fifteen properties. A
    # measure that discards the fine scale to tidy its own statistics is
    # answering a different question from the one asked.
    if polarity is None:
        return [c.region for c in kept]
    if polarity > 0:
        return [c.region for c in kept if c.polarity > 0]
    return [c.region for c in kept if c.polarity <= 0]


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
    """↑  How close boundary regions come to a third of what they bound.

    Salingaros (2025): "Effective boundaries are proportionally wide to what they
    enclose; typically, the boundary **measures roughly 1/3 of what it bounds**."
    That is a ratio with a stated target, so the measure is how near the artwork's
    boundary regions come to it.

    A region's own characteristic width is 2A/P; what it bounds is the equivalent
    diameter of the largest region it borders. The score is highest when that
    ratio sits at 1/3, falling away on either side — a hairline boundary and a
    boundary as wide as its interior are both failures, and the source says so:
    "a thick boundary also functions as an 'implied' center".

    *Redefined (#22).* This measure was the field value at the midpoint between
    connected centres, read off the reconstructed Gaussian rendering rather than
    the image, and it involved no thickness and no ratio. That formula scored
    +0.62 against the band-thickness sweep, better than this one does — but it
    was not measuring boundary thickness, and an accidental correlation is what
    this audit exists to remove.

    **Not yet working, and the diagnosis points at the segmentation.** Against
    the generator that sweeps this quantity the measure scores near zero, and no
    aggregation helps — median, mean, 90th percentile and maximum all fail. The
    generator is not the suspect this time: unlike ``symmetry_order``, it was
    validated by direct measurement of the rendered image, independent of the
    pipeline. The likely cause is that a watershed seeded at detected centres
    does not give a thin band or an interdigitating finger a basin of its own —
    it is absorbed into the region it borders — so the descriptor never sees the
    geometry the stimulus varies. That is segmentation work, not descriptor work.
    """
    ratios = [
        r.boundary_ratio for r in _regions(centers) if r.boundary_ratio > 0
    ]
    if not ratios:
        return None
    # Distance from the sourced target, in log units so that a boundary half the
    # target width and one twice it are equally wrong.
    target = 1.0 / 3.0
    deviations = np.abs(np.log(np.array(ratios) / target))
    return float(np.exp(-np.median(deviations)))

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
    """↑  Median solidity of the interstitial ground.

    Salingaros (2025): "The experienced space itself … is typically **convex**,
    providing comfort and coherence, while the enclosing solid boundary is mostly
    **concave**." Solidity is area over convex-hull area, one for a convex region.

    *Redefined (#22).* This measure was the mean squared deviation of child-area
    coverage from **0.65** — a constant that appears nowhere in the source, and a
    quantity with no relation to convexity. It scored −0.38 against a stimulus
    sweeping the solidity of the interstitial ground; region solidity scores
    **+0.829** on the same sweep.

    Measured on the ground population, because that is the "experienced space"
    the source is talking about: the space between and around the solids.
    """
    ground = _regions(centers, polarity=-1)
    if not ground:
        return None
    # The substantial ground regions only. Watershed fragments the interstitial
    # web into many pieces and leaves slivers where basins meet, and a sliver is
    # not "the experienced space … where a user is situated" that the source
    # describes. Selecting the regions above median area lifts this measure from
    # rho +0.37 to +1.000 against a sweep of ground solidity.
    #
    # Note this is a selection *within one property*, justified by what that
    # property is about. A global area floor across all measures was tried and
    # reverted: it deletes the smallest scale level, which levels of scale
    # requires be present.
    threshold = float(np.median([r.area for r in ground]))
    substantial = [r for r in ground if r.area >= threshold] or ground
    return float(np.median([r.solidity for r in substantial]))

def good_shape(centers):
    """↑  Median compactness of the figure regions.

    Salingaros (2025): "Harmonious and aesthetically pleasing forms remind us of
    animal shapes that must be **compact** … Compact shapes are cognitively
    'graspable'." Compactness is 4*pi*A/P^2, one for a disc and falling as the
    outline becomes ragged or elongated.

    *Redefined (#22).* This measure was the **fraction of centres having at least
    one child** — a hierarchy statistic containing no shape information of any
    kind. Against a stimulus sweeping motif circularity it scored +0.16 after
    controlling for centre count; compactness of the segmented region scores
    **+0.943** on the same sweep.

    Figure regions only. The interstitial ground between motifs is bounded by
    those motifs and takes a ragged outline from them, so pooling both
    populations measures the packing rather than the shapes.
    """
    shapes = [r.compactness for r in _regions(centers, polarity=+1)]
    return float(np.median(shapes)) if shapes else None

def local_symmetries(centers):
    """↑  Area-weighted mean bilateral symmetry about the vertical axis.

    Salingaros (2025): "**Bilateral symmetry about the vertical axis** respects
    gravitational stability … Nested symmetries — where a smaller one fits inside
    a larger one — must act on **every distinct scale** in the scaling hierarchy."

    Two things follow, and both are in this measure. The axis is *vertical*, not
    any axis: the source ties it to gravity, so a design symmetric about a
    horizontal axis is not the same thing. And it acts at every scale, so the
    average is weighted by region area rather than counting small and large
    regions alike.

    *Redefined (#22).* This measure was the mean squared deviation of a child's
    radial distance from 0.5 of its parent's radius — a figure absent from the
    source, and one in which no symmetry is computed at all.

    **Not yet validated, and the reason is the stimulus rather than the measure.**
    The available generator sweeps *rotational* order m, and a regular m-gon is
    bilaterally symmetric at every order — measured vertical symmetry is 0.986 to
    0.999 for every m from 3 to 12, at any rotational phase. Rotational order and
    bilateral-vertical symmetry are very nearly independent, so that sweep cannot
    test this property whatever the measure computes. ``bilateral_asymmetry`` in
    ``audit/stimuli.py`` sweeps the sourced quantity directly.
    """
    regions = _regions(centers, polarity=+1)
    if not regions:
        return None
    values = np.array([r.vertical_symmetry for r in regions])
    weights = np.array([r.area for r in regions])
    return float((values * weights).sum() / weights.sum())

def deep_interlock(centers, G):
    """↑  Complexity of the interfaces regions share with their neighbours.

    Salingaros (2025): "Two regions can **interpenetrate at a semi-permeable
    interface** … A **complex (not brusque) interface** joins the two regions into
    a larger whole … Abrupt, clean transitions between two regions coming up to
    each other but failing to connect weaken visual cohesion."

    Measured as the length of each shared interface divided by the square root of
    the smaller region's area. A straight cut across a compact region scores
    about 1; an interdigitating interface scores several times that. The value is
    mapped so that a brusque interface tends to 0 and a deeply interlocked one
    towards 1.

    *Redefined (#22).* This measure was the fraction of graph edges whose centres
    lay within the sum of their radii — a statement about the positions of two
    points, carrying nothing about the shape of the interface between them. It
    spanned 0.2 of 10 across the entire corpus, which is to say it was close to a
    constant.

    **Not yet working, and the diagnosis points at the segmentation.** Against
    the generator that sweeps this quantity the measure scores near zero, and no
    aggregation helps — median, mean, 90th percentile and maximum all fail. The
    generator is not the suspect this time: unlike ``symmetry_order``, it was
    validated by direct measurement of the rendered image, independent of the
    pipeline. The likely cause is that a watershed seeded at detected centres
    does not give a thin band or an interdigitating finger a basin of its own —
    it is absorbed into the region it borders — so the descriptor never sees the
    geometry the stimulus varies. That is segmentation work, not descriptor work.
    """
    complexities = [
        r.interface_complexity for r in _regions(centers) if r.interface_complexity > 0
    ]
    if not complexities:
        return None
    # A straight interface scores about 1, so subtract that floor before scaling.
    excess = max(float(np.median(complexities)) - 1.0, 0.0)
    return float(1.0 - np.exp(-excess))

def contrast(G):
    """↑  Mean absolute tone difference between adjacent regions.

    Salingaros (2025): "**black-white and color contrast** for differentiation …
    Contrast is needed to provide figure-ground symmetry of opposites."

    *Redefined (#22), and previously impossible.* This measure was the weighted
    mean strength difference across graph edges. Strengths derive from the
    structural field, the field is a distance transform, and a distance transform
    **carries no tone at all** — so the measure could not see the quantity it was
    named for under any formula. The audit measured it as *flat*: across a sweep
    in which figure/ground tonal separation more than tripled, it moved 0.2%,
    from 0.06694 to 0.06524.

    Region tone supplies what was missing. Adjacency comes from the reinforcement
    graph, whose weight now peaks where two centres touch (#28), so an edge is
    the relation the source means by differentiating one unit from its neighbour.
    """
    if not G.edges:
        return None
    diffs = []
    for i, j, data in G.edges(data=True):
        a = G.nodes[i]["center"].region
        b = G.nodes[j]["center"].region
        if a is None or b is None or a.area == 0 or b.area == 0:
            continue
        diffs.append(data["weight"] * abs(a.tone - b.tone))
    if not diffs:
        return None
    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    return float(sum(diffs) / (sum(weights) + 1e-12))

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
    """↑  Similarity of motif shape, within and across scales.

    Salingaros (2025): "Similar visual patterns and **shapes** are repeated both
    on the same scale … as well as **across different scales** … one form 'echoes'
    a larger or smaller form by means of common visual features."

    Measured as the mean pairwise similarity of the log-scaled Hu-moment shape
    signature, which is invariant to scale, rotation and reflection — so it
    compares *shape* rather than size, and a small motif echoing a large one of
    the same form scores as an echo, which is exactly what the source describes.

    *Redefined (#22).* This measure was the standard deviation of log parent/child
    scale **ratios** — a property of the hierarchy's spacing, not of whether any
    form resembles any other. It is possible to have perfectly consistent scale
    ratios and no repeated shape whatever.
    """
    signatures = [r.shape_signature for r in _regions(centers) if r.shape_signature]
    if len(signatures) < 2:
        return None
    # The first four Hu moments carry the gross form; the last three are
    # dominated by digitisation noise on small regions.
    M = np.array([s[:4] for s in signatures], dtype=float)
    if len(M) > 400:  # pairwise distance is O(n^2); a sample suffices
        M = M[np.random.default_rng(0).choice(len(M), 400, replace=False)]
    spread = M.std(axis=0)
    spread[spread < 1e-9] = 1.0
    Z = M / spread
    d = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(len(Z), k=1)
    return float(np.exp(-np.median(d[iu])))

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
