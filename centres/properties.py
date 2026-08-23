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

import cv2
import numpy as np
import networkx as nx

from .energy import hierarchy_energy, coverage_energy, alignment_energy, field_energy



#: The sourced band of optimal magnification factors between successive scales.
#: Salingaros (2025): "optimal magnification factors range between approximately
#: 2 to 5", with 1.5 too close to distinguish and 10 disengaging.
BAND_LOW = 2.0
BAND_HIGH = 5.0


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


def _tone_threshold(gray):
    """Symmetrised Otsu threshold for splitting figure from ground.

    Otsu adapts the split to the image's own tonal histogram, so a gamma or JPEG
    change — which remaps tone — moves the split *with* it, instead of leaving a
    fixed grey-128 cut at a now-different relative level. Measured on the corpus,
    replacing the fixed 128 with this cut the gamma sensitivity of ``boundaries``
    from 2.9 to 0.25 and of ``deep_interlock`` from 1.7 to 0.46 on the 0–10 scale,
    which is the noise floor #31 set out to lower.

    Symmetrised — ``t = (otsu(g) + 255 - otsu(255 - g)) / 2`` — so that
    ``t(255 - g) = 255 - t(g)`` exactly. That keeps the figure/ground split, and
    the two measures built on it, invariant to tone inversion (#30): plain Otsu is
    only *approximately* inversion-symmetric because it thresholds at integers, and
    measured at a 0.34 inversion error on ``boundaries`` that the symmetrisation
    removes (back to 0.00, as the fixed threshold had).
    """
    t_fwd, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_inv, _ = cv2.threshold(255 - gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return 0.5 * (float(t_fwd) + 255.0 - float(t_inv))


def levels_of_scale(centers):
    """↑  How far parent/child scale ratios fall inside the sourced band.

    Salingaros (2025): "Each scale must be distinct, with scales in a hierarchy
    spaced closely enough in size for scaling coherence, but not too close to
    blur the distinction … **Optimal magnification factors range between
    approximately 2 to 5.** A subtle magnification factor of 1.5 is too close to
    distinguish one scale from another, whereas an abrupt jump in adjacent scales
    by a factor of 10 is disengaging."

    That is a **band**, not a point. A ratio anywhere in [2, 5] is optimal, so the
    measure is zero-penalty inside it and grows in log units outside — reaching a
    substantial penalty at the 1.5 and 10 the source names as failures.

    *Repaired (#22).* The previous formula was a quadratic penalty about a single
    ratio of 3, described as the midpoint of an unsourced range of "2 to 4". Both
    the range and the point target were wrong: the source gives 2 to 5, and gives
    it as a band, so a quadratic about any single value is the wrong shape however
    the constant is chosen. The audit measured the old measure's minimum at a
    ratio of 2.381 — inside the sourced band — so it was being scored against a
    target that was itself incorrect.
    """
    from .pipeline import _extent

    # The ratio is between *extents*, the same quantity the hierarchy is built
    # on. Blob scale understates a centre's true size by about 28%, and using two
    # different notions of size for containment and for the ratio would make the
    # measure disagree with the structure it reads.
    ratios = [
        _extent(centers[c.parent]) / (_extent(c) + 1e-8)
        for c in centers
        if c.parent is not None
    ]
    if not ratios:
        return None
    ratios = np.array(ratios, dtype=float)
    below = np.log(BAND_LOW / np.clip(ratios, 1e-8, None))
    above = np.log(np.clip(ratios, 1e-8, None) / BAND_HIGH)
    outside = np.maximum(np.maximum(below, above), 0.0)
    return float(np.exp(-float(np.mean(outside))))

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


def boundaries(field, centers, G, gray=None):
    """↑  How close the artwork's boundaries come to a third of what they bound.

    Salingaros (2025): "Effective boundaries are proportionally wide to what they
    enclose; typically, the boundary **measures roughly 1/3 of what it bounds**."
    A ratio with a stated target, so the score is how near the boundaries come to
    it, highest at 1/3 and falling away on either side — a hairline boundary and
    one as wide as its interior are both failures, and the source says so: "a
    thick boundary also functions as an 'implied' center".

    Read from the image, not from the centre set, and this is essential rather
    than incidental. The structural field is a distance transform, so a region's
    salience is its distance to the nearest edge, and a boundary is thin *by
    definition* — measured, not one detected centre lands on a band at any
    thickness, because a band of half-width w peaks at w while the areas it
    separates peak far higher. A representation whose salience is
    distance-to-nearest-edge is structurally incapable of making a thin thing
    salient, so no centre-based measure can see a boundary. The boundaries are
    instead the dark connected components of the image; their width is twice the
    component area over its perimeter, and what they bound is the light regions
    they enclose.

    *Redefined (#22).* This measure was the field value at the midpoint between
    connected centres, read off the reconstructed Gaussian rendering. That scored
    +0.62 against the band-thickness sweep, better than the centre-region version
    that replaced it first — but it was an accidental correlation, measuring no
    thickness and no ratio. Read from the image this scores +1.000 on the same
    sweep, and its score peaks where the band is a third of what it bounds.

    Returns ``None`` when the image is not available — the generative ``evolve``
    path has a centre set but no image, and boundaries cannot be read without one.
    """
    if gray is None:
        return None

    def widths_and_diameters(mask):
        widths, diameters = [], []
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area < 10:
                continue
            contours, _ = cv2.findContours(
                (labels == i).astype(np.uint8), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
            if contours:
                p = float(cv2.arcLength(max(contours, key=cv2.contourArea), True))
                if p > 0:
                    widths.append(2.0 * area / p)
            diameters.append(2.0 * np.sqrt(area / np.pi))
        return widths, diameters

    t = _tone_threshold(gray)
    dark = widths_and_diameters((gray < t).astype(np.uint8))
    light = widths_and_diameters((gray >= t).astype(np.uint8))
    if not dark[0] or not light[0]:
        return None

    # Which tone is "the boundary" is a convention that inverting the image would
    # flip, so it must not change the answer: the boundary is the thinner
    # population and what it bounds is the thicker, whichever is dark (#30).
    if float(np.median(dark[0])) <= float(np.median(light[0])):
        boundary, bounded = dark, light
    else:
        boundary, bounded = light, dark

    ratio = float(np.median(boundary[0])) / float(np.median(bounded[1]))
    return float(np.exp(-abs(np.log((ratio + 1e-9) / (1.0 / 3.0)))))

def alternating_repetition(G):
    """↑  Mean standard deviation of strengths across each centre's neighbours.

    Approximation, and the one measure of the fifteen that still does **not** track
    its ground truth (rho +0.049 against a checkerboard-of-two-sizes sweep). Strength
    is not what alternates, and — measured (#22, PLAN D2 note) — no reachable measure
    fixes it: the detection pipeline does not encode motif-size alternation in centre
    attributes, the reinforcement graph's scale term actively drops the alternating
    (dissimilar-scale) adjacencies, and image-domain periodicity is either swamped
    (autocorrelation decays) or dominated by the base lattice (FFT). Seven candidates
    were tried; none clears the tracking bar. A real fix needs a purpose-built
    periodicity detector robust on aperiodic real art, so this is left honest-and-
    failing rather than swapped for another measure that also does not track.

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

    Measured over every substantial region of both tonal populations: the source
    applies this to "All spaces, both built and open", and using both makes it
    invariant to tone inversion (#30).
    """
    # Every region's own convexity, over both tonal populations. The source
    # applies positive space to "All spaces, both built and open … the duality
    # between figure and background", so it is not a ground-only property; and
    # reading both populations makes it invariant to which tone is called figure,
    # i.e. to inverting the image (#30). Restricting to one polarity scored the
    # same +1.000 on the ground_solidity sweep but moved 0.07 under inversion.
    ground = _regions(centers)
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

def deep_interlock(centers, G, gray=None):
    """↑  How far region boundaries depart from their own smooth envelope.

    Salingaros (2025): "Two regions **interpenetrate at a semi-permeable
    interface** … A **complex (not brusque) interface** joins the two regions into
    a larger whole … Abrupt, clean transitions … weaken visual cohesion."

    For each figure component, the ratio of its contour perimeter to the
    perimeter of its convex hull. A brusque, convex outline sits at 1; an
    interdigitating one, whose boundary weaves in and out, is longer than its
    hull and scores above 1. Area-weighted across components, then mapped to
    ``1 - 1/ratio`` so a straight interface is 0 and a deeply woven one approaches
    1. The convex hull is the boundary's own smooth envelope, so no smoothing
    kernel and no scale constant is needed — the measure is dimensionless by
    construction.

    Read from the image, for the same structural reason as *thick boundaries*: an
    interdigitating finger is thin, a distance-transform field gives no maximum to
    a thin thing, so the finger never becomes a centre and a watershed seeded at
    centres never cuts along it. Measured, the centre-region version scored +0.04
    against the interdigitation sweep; this scores +1.000, monotone, and is
    independent of good_shape across the corpus (Spearman −0.09), so it is not
    that measure renamed.

    *Redefined (#22).* The original measure was the fraction of graph edges whose
    centres lay within the sum of their radii — a statement about two points'
    positions, carrying nothing about the shape of the interface. It spanned 0.2
    of 10 across the whole corpus.

    Returns ``None`` without an image (the generative ``evolve`` path).

    Confound, stated: like every boundary measure here it cannot distinguish a
    genuinely interpenetrating pair of regions from a single region with a merely
    ragged outline. On the corpus that shows as a high value for the Ardabil
    (perimeter ratio 6.4), whose dense arabesque has very convoluted component
    boundaries whether or not one would call them "interlocked".
    """
    if gray is None:
        return None
    ratios, weights = [], []
    # Both tonal populations, not just the dark one: two interpenetrating regions
    # share the same interface, so the convolution is the same measured from
    # either side, and reading both makes the result invariant to which tone is
    # called figure -- i.e. to inverting the image (#30).
    populated = 0
    t = _tone_threshold(gray)
    for mask in ((gray < t).astype(np.uint8), (gray >= t).astype(np.uint8)):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        contributed = False
        for i in range(1, n):
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area < 20:
                continue
            contours, _ = cv2.findContours(
                (labels == i).astype(np.uint8), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
            if not contours:
                continue
            c = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(c, True)
            hull_perimeter = cv2.arcLength(cv2.convexHull(c), True)
            if hull_perimeter > 0:
                ratios.append(perimeter / hull_perimeter)
                weights.append(area)
                contributed = True
        populated += contributed
    # Interlock is a relation between two regions, so both tones must be present;
    # a uniform field has nothing to interpenetrate and is undefined, not zero.
    if not ratios or populated < 2:
        return None
    ratio = float(np.average(ratios, weights=weights))
    return float(1.0 - 1.0 / max(ratio, 1.0))

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

def gradients(field, centers, G=None):
    """↑  How gradually tone changes across space, rather than in steps.

    Salingaros (2025): "**Gradual changes and transitions in colour, size, or
    texture** … Gradients represent controlled transitions and avoid abrupt
    interruptions. Certain regions need continuous variation instead of contrast."

    Measured as the rate of tonal change between adjacent regions: the difference
    in tone divided by the distance between them, that distance expressed in
    units of their own combined size. A wide transition spreads a given tonal
    change over many regions, so the rate per step is low; an abrupt one
    concentrates it into a single step.

    That the source sets this *against* contrast is deliberate on its part, and
    they are not simply inverses here: contrast is the size of the step between
    neighbours, this is the step divided by the distance it is taken over. A
    design can have strong contrast and gradual transitions at once, which is
    what the source describes when it says both are needed in different regions.

    *Redefined (#22).* This measure was the mean squared gradient of the
    *reconstructed* Gaussian field — a property of the blob rendering, not of the
    image, and one carrying no tone at all.
    """
    if G is None or not G.edges:
        return None
    rates = []
    for i, j, _ in G.edges(data=True):
        a, b = G.nodes[i]["center"], G.nodes[j]["center"]
        if a.region is None or b.region is None:
            continue
        if a.region.area <= 0 or b.region.area <= 0:
            continue
        span = a.scale + b.scale
        if span <= 0:
            continue
        separation = np.hypot(a.x - b.x, a.y - b.y) / span
        if separation <= 0:
            continue
        rates.append(abs(a.region.tone - b.region.tone) / separation)
    if not rates:
        return None
    # The 90th percentile, not the median. Most adjacent pairs sit inside a
    # uniform area and differ in tone by nothing at all, so the median is 0 and
    # the measure saturates at 1 for every image -- measured, before this. The
    # property is about how gradually the *transitions* are taken, so the
    # statistic has to be drawn from the steepest of them.
    return float(np.exp(-float(np.percentile(rates, 90)) * 4.0))

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


def compute_all(field, centers, G, gray=None):
    """Return raw scores for all 15 of Alexander's structural properties.

    A value is ``None`` where that property is undefined for these inputs —
    see the module docstring and each measure's own docstring for where its
    boundary lies. Callers must handle ``None``; it is never a number.
    """
    return {
        "levels_of_scale": levels_of_scale(centers),
        "strong_centres": strong_centres(centers),
        "boundaries": boundaries(field, centers, G, gray),
        "alternating_repetition": alternating_repetition(G),
        "positive_space": positive_space(centers),
        "good_shape": good_shape(centers),
        "local_symmetries": local_symmetries(centers),
        "deep_interlock": deep_interlock(centers, G, gray),
        "contrast": contrast(G),
        "gradients": gradients(field, centers, G),
        "roughness": roughness(centers),
        "echoes": echoes(centers),
        "the_void": the_void(field, centers),
        "simplicity": simplicity(centers),
        "not_separateness": not_separateness(G),
    }


#: Soft-saturation scale S for each ↑ property that has a reference level:
#: ``normalize_all`` maps its raw value through ``10·(1 − e^(−x/S))``. Each S is the
#: corpus **median / ln 2**, so the median artwork maps to exactly 5.0 and the range
#: stays open at both ends (nothing pins at 10). Measured over the 44-image corpus at
#: --max-size 1024 (2026-08-22); re-measure with the same corpus if the front end
#: changes. These are display-normalisation constants only — the reported degree of
#: life is computed from the energy terms, not from these 0–10 property scores — so
#: anchoring them to the corpus is the documented intent ("typical values span 3–8"),
#: not the frame-versus-artwork error the score invariant forbids. Raw corpus medians:
#: strong_centres 0.685, alternating_repetition 0.118, simplicity 0.228,
#: not_separateness 0.0039. See #16 and AUDIT.md §8.
SATURATION = {
    "strong_centres": 0.989,          # 0.6853 / ln 2
    "alternating_repetition": 0.170,  # 0.1181 / ln 2
    "simplicity": 0.328,              # 0.2275 / ln 2
    "not_separateness": 0.00563,      # 0.0039 / ln 2
}


def normalize_all(raw):
    """Map raw property scores to 0–10 wholeness scores (10 = most present).

    All 15 scores use the same direction: higher = more of Alexander's property.
    The transformation for each property is either:
      - exp(-k·x) for ↓ properties (penalise positive raw values, ideal = 0)
      - 10·(1 − e^(−x/S)) for ↑ properties with a soft-saturation scale (see
        ``SATURATION`` and ``rise``); this replaced the old linear clip, which
        pinned real artworks at 10.0 (#16)
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

    def rise(x, s):
        """Soft-saturating rise, 10·(1 − e^(−x/s)): 0 at x=0, → 10 as x grows.

        Replaces the old linear clip ``10·min(x/ref, 1)``. The clip pinned real
        artworks at exactly 10.0 — ``not_separateness`` did so for 4 of the 44 corpus
        images (and for half of the old six-carpet corpus), and the ceiling ties
        were being read as agreement between artworks when they were only evidence
        that the constant was too low (#16, AUDIT.md §8). It also *floored* the
        opposite cases: ``strong_centres`` and ``simplicity`` spanned only 0.5–1.1
        and 1.6–3.1 of the 0–10 range. A soft exponential — the same shape
        ``decay`` and the energy reward already use — never pins, and anchoring its
        scale at the corpus median (below) puts the typical artwork at 5 with the
        range open at both ends, whatever the shape of the raw distribution (which
        for ``not_separateness`` is strongly right-skewed, so no linear constant
        could span 3–8 without pinning the tail).
        """
        return 10.0 * float(-np.expm1(-max(float(x), 0.0) / s))

    def roughness_peak(x):
        return 10.0 * float(np.exp(-((x - 0.5) ** 2) / 0.04))

    #: The nine measures redefined or repaired against the source (#22) all
    #: return a value already in [0, 1] with 1 as the ideal -- a compactness, a
    #: solidity, a symmetry fraction, a tone difference, or an exp(-deviation).
    #: They need scaling to the 0-10 display range and nothing else, and a
    #: reference constant would only reintroduce the saturation that #16 removed.
    direct = {
        "levels_of_scale", "boundaries", "positive_space", "good_shape",
        "local_symmetries", "deep_interlock", "contrast", "gradients", "echoes",
    }

    transforms = {
        # ↑ properties already on a natural 0-1 scale
        **{k: (lambda x: 10.0 * min(max(float(x), 0.0), 1.0)) for k in direct},
        # ↑ properties with a soft-saturation scale (see SATURATION and rise)
        **{k: (lambda x, s=s: rise(x, s)) for k, s in SATURATION.items()},
        # ↓ property — lower raw = better
        "the_void": lambda x: decay(x, 40.0),
        # ~ roughness — ideal at moderate irregularity, peak at 0.5
        "roughness": roughness_peak,
    }

    return {
        key: (None if raw[key] is None else min(max(fn(raw[key]), 0.0), 10.0))
        for key, fn in transforms.items()
    }
