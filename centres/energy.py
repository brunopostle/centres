"""Degree of life, and the energy the generative mode minimises.

The reported quantity is the **degree of life** L — Alexander's own term — with
the semantics the theory needs: **zero for nothing, higher for more**. `E = -L`
is retained as the quantity `evolve()` minimises, because simulated annealing is
written to descend.

Why the sign was wrong before. Every term here began as a *penalty for
deviation* evaluated only over the objects it applies to, and every one of them
is 0 when its set is empty: no parent-child pairs, no parents, no graph edges,
no overlapping pairs. So the global minimum of the old functional was the
*absence of structure* — 36 centres at one scale, spaced far apart, scored
E = +0.081 against +1.40 for random and +0.95..+1.74 for the six carpets. The
energy analogy is only sound with a constraint: a catenary minimises energy
*subject to fixed endpoints*, and without the constraint every such minimum is
trivial. Nothing here constrained how much structure existed. Issue #28.

The repair, per the repository owner on #28:

1. the empty case must score exactly **zero**, so the optimiser has a neutral
   baseline rather than an attractor;
2. structure must be **rewarded**;
3. the reward must **not** be a sum over centres — that would simply cram in as
   many centres as the optimiser can fit, contradicting *the void*.

Every descriptive term is therefore factored into **participation x quality**:

    L_term = participation * quality

*Participation* is the fraction of the centres that take part in that kind of
structure — 0 when none do, 1 when all do. *Quality* maps the term's existing
per-object mean deviation onto [0, 1], ideal = 1 and poor = 0. The three
constraints then hold by construction:

- **Empty is exactly zero**, because participation is zero — not because a sum
  has no terms, but because nothing participates.
- **Structure is rewarded**, because participation and quality are both
  non-negative and only structure makes them positive.
- **Cramming does not pay**, because participation is a fraction and quality is
  a mean. Adding weak centres raises the denominator of both. Twenty
  well-related centres beat a hundred badly-related ones, and a deliberately
  empty region costs nothing at all — participation is measured over the centres
  that exist, so *the void* survives.
"""

import numpy as np
from collections import defaultdict


def hierarchy_energy(centers):
    """Total deviation from a scale ratio of ~3 between parent and child.

    Natural hierarchies (Alexander's 'levels of scale') follow approximately
    constant ratios between successive scales. The target ratio of 3 is the
    midpoint of the empirically observed range of 2-4.

    Summed over parent-child pairs. `total_energy` divides by the pair count and
    turns the result into a quality; on its own this is a raw deviation, and
    `properties.py` reads it as one.
    """
    E = 0
    for c in centers:
        if c.parent is None:
            continue
        parent = centers[c.parent]
        ratio = parent.scale / (c.scale + 1e-8)
        E += (np.log(ratio) - np.log(3)) ** 2
    return E


def reinforcement_energy(G):
    """Reward mutual strength between connected centres, per connection.

    E_R = -mean over graph edges of W_ij * s_i * s_j

    The mean is taken over the edges that exist, not over the N(N-1)/2 pairs
    that could exist. Both denominators remove the O(N²) growth of the raw sum,
    but only the edge count gives a value that is independent of N: the
    reinforcement graph is spatially local, so its edge count grows roughly
    linearly with N and dividing by N(N-1)/2 leaves E_R decaying as O(1/N).
    Dividing by the edge count makes this "the typical reinforcement across a
    connection", which is the quantity that is comparable between images.

    This is the one term that was already a reward rather than a deviation, and
    it is the one that is turned into a quality by ``_quality_of_reward`` rather
    than by ``_quality_of_deviation``. It is returned negative for backwards
    compatibility with ``properties.py``, which reads it directly.

    Note that it can no longer be made arbitrarily negative by collapsing every
    centre onto one point: since the kernel in ``graph.build_graph`` peaks at
    *adjacency* rather than at coincidence, coincident centres share no edge at
    all. See that function for the measurement.
    """
    m = G.number_of_edges()
    if m == 0:
        return 0.0
    E = 0.0
    for i, j, data in G.edges(data=True):
        s1 = G.nodes[i]["center"].strength
        s2 = G.nodes[j]["center"].strength
        E -= data["weight"] * s1 * s2
    return E / m


def locality_energy(centers):
    """Mean pairwise Gaussian overlap exp(-d²/(r_i+r_j)²).

    1.0 when all centres are coincident, ~0 when well-separated, ~1e-3 on real
    images.

    This is the one term that is subtracted from the degree of life rather than
    added to it, and it remains a barrier rather than a descriptor. With the
    reinforcement kernel now peaking at adjacency, coincident centres share no
    edges and score 0 on every participation-weighted term — so a *plain*
    collapse is no longer an attractor and needs no barrier. What still needs
    one is the **concentric** collapse: centres stacked at a single point but at
    scales in a 3:1 ladder do have parent-child pairs, and would otherwise
    collect the full hierarchy and coverage reward for a configuration with no
    spatial extent at all.

    It enters the degree of life **squared**, with weight
    ``sum(PRIORITY) = 1``. Both follow from what a barrier is for.

    The weight is what the new bound makes of it: the descriptive terms are each
    a product of two numbers in [0, 1] weighted by priorities that sum to 1, so
    they can contribute at most 1, and a unit weight at full coincidence is
    exactly enough to cancel the best score any configuration could earn. The
    old value of 5.54 — the reinforcement lower bound plus the worst ensemble
    energy — belonged to the additive framing and no longer applies.

    The square is what keeps the barrier *flat where the constraint is not
    active*. This term is the one quantity here that is not intensive: it
    divides by all N(N-1)/2 pairs while only O(N) of them overlap at all, so at
    fixed spatial density it decays as 1/N. Entering linearly, it therefore
    smuggles the centre-count dependence that #14 removed back into the total —
    measured over random configurations at fixed density with N = 32..256, it
    took r(L, N) from +0.11 to +0.55, and that rise was the barrier alone, not
    the descriptive terms. Squaring concentrates the barrier where it belongs:
    it still costs the full 1.0 at coincidence and 0.51 for a tight cluster
    (E_L = 0.71), while costing 1e-6 on a real image where E_L is about 1e-3.
    r(L, N) returns to +0.14.

    Two genuinely intensive alternatives were measured and rejected. Mean
    largest-overlap-per-centre is N-independent, but it reads 0.20-0.26 on the
    six carpets and 0.56-0.62 on random configurations at fixed density, so at
    barrier weight it would push every random configuration below the empty one
    — it is a descriptor, not a barrier, and it belongs to step 3 of #28 as one.
    Its square has the same character, more weakly.

    Re-deriving E_L as a *descriptor* — Alexander's *deep interlock and
    ambiguity* says centres should interpenetrate, so a term whose minimum is
    maximal separation arguably has the theory backwards — is left to the third
    step of #28, after this one.
    """
    n = len(centers)
    if n < 2:
        return 0.0
    pos = np.array([[c.x, c.y] for c in centers])
    scales = np.array([c.scale for c in centers])
    dx = pos[:, 0:1] - pos[:, 0]
    dy = pos[:, 1:2] - pos[:, 1]
    d2 = dx**2 + dy**2
    r = scales[:, None] + scales[None, :]
    overlap = np.exp(-d2 / (r**2 + 1e-8))
    # upper triangle only (exclude self-pairs on diagonal)
    i_upper, j_upper = np.triu_indices(n, k=1)
    return float(overlap[i_upper, j_upper].mean())


def coverage_energy(centers):
    """Total deviation from ideal child-area coverage of a parent (~0.65).

    Coverage C_i = sum(r_child^2) / r_parent^2. Values near 0.65 indicate
    children fill their parent region without overcrowding it. Summed over
    parents.
    """
    children = defaultdict(list)
    for c in centers:
        if c.parent is not None:
            children[c.parent].append(c)
    E = 0
    C_target = 0.65
    for parent_id, childs in children.items():
        parent = centers[parent_id]
        coverage = sum(ch.scale**2 for ch in childs) / (parent.scale**2 + 1e-8)
        E += (coverage - C_target) ** 2
    return E


def alignment_energy(centers):
    """Total deviation of child radial distance from 0.5 * r_parent.

    Alexander observed that child centres tend to lie at roughly 0.3-0.7 of the
    parent radius from the parent centre. Target is the midpoint, 0.5. Summed
    over parent-child pairs.
    """
    E = 0
    d_target = 0.5
    for c in centers:
        if c.parent is None:
            continue
        parent = centers[c.parent]
        d = np.sqrt((c.x - parent.x) ** 2 + (c.y - parent.y) ** 2) / (
            parent.scale + 1e-8
        )
        E += (d - d_target) ** 2
    return E


def field_energy(field):
    """Mean squared gradient: penalises abrupt transitions in the wholeness field."""
    gy, gx = np.gradient(field)
    return np.mean(gx**2 + gy**2)


#: Relative priority of each descriptive term, summing to 1. These are the only
#: free numbers in the degree of life. They are the emphasis the original
#: weights were trying to express — hierarchy and reinforcement first, coverage
#: second, alignment and field smoothness third — carried over unchanged.
#:
#: Because every term is now a product of a participation in [0, 1] and a
#: quality in [0, 1], and these sum to 1, the descriptive part of the degree of
#: life is bounded in [0, 1]: 0 is "no structure of any kind", 1 is "every
#: centre participates in every kind of structure, perfectly". Neither end is
#: reachable by a real image, and the six carpets sit at 0.32 to 0.41.
PRIORITY = {"H": 0.3, "R": 0.3, "C": 0.2, "A": 0.1, "F": 0.1}

#: Deviation scale of each term: the value at which its quality has fallen to
#: 1/e (or, for the reinforcement reward, risen to 1 - 1/e).
#:
#: Purpose. Under the old additive total these constants were unit conversions —
#: the five terms are measured in unrelated units (squared log ratios,
#: weight-times-strength products, squared area fractions, squared radius
#: fractions, squared field gradients) whose spreads span two orders of
#: magnitude, so each was divided by its own standard deviation before being
#: added. Nothing is added across units any more: each term is mapped through
#: exp(-D/S) into a dimensionless quality first. The constant's job is therefore
#: no longer "one unit of this term" but "the deviation at which this term stops
#: counting as good", and the statistic that fixes it changes accordingly, from
#: the ensemble standard deviation to the **ensemble median**. Anchoring on the
#: median puts the typical measured artwork at a quality of 1/e = 0.37, which is
#: where the map has its steepest response and so discriminates best between
#: real images. Anchoring on the standard deviation instead would have put the
#: corpus at a quality of 0.003 to 0.05 for three of the five terms — every real
#: image indistinguishably bad.
#:
#: The reference ensemble is unchanged: the 33 non-degenerate cases of the audit
#: corpus — the six carpets in images/, the null controls in audit/stimuli.py
#: that yield centres, and every frame of the five parametric sweeps, each at
#: --max-size 1024. "Non-degenerate" excludes the cases that produce no
#: parent-child pair at all, for which H, C and A are not defined. Re-measured
#: 2026-08-16 against the adjacency-peaked reinforcement kernel, which
#: invalidated the previous values; re-measure with the same ensemble if the
#: front end changes.
SCALE = {
    "H": 0.316774,  # (log(r_parent/r_child) - log 3)^2, per parent-child pair
    "R": 0.086566,  # W_ij s_i s_j, per graph edge
    "C": 0.140928,  # (coverage - 0.65)^2, per parent
    "A": 3.447728,  # (d/r_parent - 0.5)^2, per parent-child pair
    "F": 0.025169,  # r_rms^2 * mean|grad phi|^2, dimensionless
}

#: Weight on the locality barrier. The descriptive terms sum to at most
#: sum(PRIORITY) = 1, so a unit weight is exactly enough for full coincidence to
#: cancel the best score any configuration could otherwise earn. See
#: locality_energy for why a barrier is still wanted at all.
WEIGHT_LOCALITY = float(sum(PRIORITY.values()))

#: Exponent on the locality barrier. 2 rather than 1, so the barrier is flat
#: where the constraint is not active; see locality_energy for the measurement.
LOCALITY_EXPONENT = 2.0


def _rms_scale(centers):
    """Quadratic mean of centre radii — the characteristic length of a centre set."""
    if not centers:
        return 0.0
    return float(np.sqrt(np.mean([c.scale**2 for c in centers])))


def _quality_of_deviation(deviation, scale):
    """Map a mean deviation onto a quality in (0, 1]: ideal = 1, poor -> 0.

    exp(-D/S) rather than max(0, 1 - D/S): a clamped map is flat above S, and a
    flat region of the objective is a region the annealer cannot descend. The
    exponential is strictly monotone everywhere, so every configuration has a
    direction of improvement.
    """
    return float(np.exp(-deviation / scale))


def _quality_of_reward(reward, scale):
    """Map a non-negative reward onto a quality in [0, 1): none = 0, large -> 1.

    The mirror image of _quality_of_deviation, for reinforcement, which is the
    one term that was already a reward rather than a deviation.
    """
    return float(-np.expm1(-reward / scale)) + 0.0  # + 0.0 normalises -0.0


def life_terms(field, centers, G):
    """Per-term (participation, quality) of the degree of life.

    Returned as a dict of term key -> (participation, quality, contribution),
    where contribution is PRIORITY * participation * quality. Exposed so the
    audit and the tests can see which half of a term moved.

    Participation is, for every term, "the fraction of the centres that this
    term's per-object mean was actually taken over":

    ============ ================================================ ============
    term         mean is over                                     participants
    ============ ================================================ ============
    H hierarchy  parent-child pairs, one per child                children
    A alignment  parent-child pairs, one per child                children
    C coverage   parents                                          parents
    R reinforce. graph edges                                      non-isolated
    F field      pixels of the reconstructed field                non-isolated
    ============ ================================================ ============

    The field term is the awkward one: it is a property of the reconstructed
    field rather than of any set of centres, so there is no set to take a
    fraction of. It is given the reinforcement participation because that is the
    honest reading of what it measures — the smoothness of the field *between*
    centres. A lone centre's Gaussian bump has a gradient, but that gradient is
    an artefact of the reconstruction and not evidence of structure, and an
    isolated centre is exactly a centre with no neighbour whose field meets its
    own. Without this the field term alone would keep a spread-out,
    unconnected, equal-scale configuration off zero, which is the defect #28
    reports.
    """
    n = len(centers)
    if n == 0:
        return {k: (0.0, 0.0, 0.0) for k in PRIORITY}

    nchildren = sum(1 for c in centers if c.parent is not None)
    nparents = len({c.parent for c in centers if c.parent is not None})
    nconnected = sum(1 for i in G.nodes if G.degree(i) > 0)

    p = {
        "H": nchildren / n,
        "A": nchildren / n,
        "C": nparents / n,
        "R": nconnected / n,
        "F": nconnected / n,
    }
    q = {
        "H": _quality_of_deviation(hierarchy_energy(centers) / nchildren, SCALE["H"])
        if nchildren else 0.0,
        "A": _quality_of_deviation(alignment_energy(centers) / nchildren, SCALE["A"])
        if nchildren else 0.0,
        "C": _quality_of_deviation(coverage_energy(centers) / nparents, SCALE["C"])
        if nparents else 0.0,
        "R": _quality_of_reward(-reinforcement_energy(G), SCALE["R"]),
        "F": _quality_of_deviation(
            field_energy(field) * _rms_scale(centers) ** 2, SCALE["F"]
        ),
    }
    return {k: (p[k], q[k], PRIORITY[k] * p[k] * q[k]) for k in PRIORITY}


def degree_of_life(field, centers, G):
    """Degree of life L: zero for nothing, higher for more.

    L = sum_k PRIORITY_k * participation_k * quality_k  -  WEIGHT_LOCALITY * E_L^2

    The five descriptive terms are each a fraction times a mean, so the sum lies
    in [0, 1] and is intensive: it does not grow with the number of detected
    centres, and neither the empty configuration nor a crammed one can win.
    Locality is subtracted as a barrier against the concentric collapse; on real
    images it costs about 1e-6 of a score around 0.35.

    Measured (2026-08-16, --max-size 1024):

    ==========================  ========
    empty canvas, 0 centres      +0.0000
    36 equal scales, far apart   -0.0000
    40 centres collapsed         -1.0000
    concentric 3:1 ladder        -0.8041
    lattice of 36                +0.2994
    six carpets                  +0.3237 .. +0.4047
    white noise / random blobs   +0.4957 / +0.4833
    ==========================  ========

    The first four lines are the point of the change: emptiness and collapse are
    no longer the optimum, they are the floor. **The last line is a failure, and
    it is not one this change caused** — it is the underlying measures, and it
    is equally present in the functional this replaces. See THEORY.md section 8
    and issue #28.

    The field term is multiplied by the squared rms centre radius before its
    quality is taken. Every other deviation is dimensionless, but
    mean|grad phi|^2 has units of 1/pixel^2, so without this the result would
    change under a pure resize of the image. Scaling by the square of a length
    taken from the centre set itself removes the dependence on pixel size,
    leaving the shape of the reconstructed field. field_energy itself is left in
    raw units because properties.gradients reports it directly and has its own
    normaliser.
    """
    terms = life_terms(field, centers, G)
    barrier = WEIGHT_LOCALITY * locality_energy(centers) ** LOCALITY_EXPONENT
    return sum(t[2] for t in terms.values()) - barrier


def total_energy(field, centers, G):
    """Structural energy, E = -L: what ``evolve()`` minimises.

    Kept as an energy only because simulated annealing descends. Everything the
    tool *reports* is the degree of life; see ``degree_of_life``.
    """
    return -degree_of_life(field, centers, G)
