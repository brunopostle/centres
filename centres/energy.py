import numpy as np
from collections import defaultdict


def hierarchy_energy(centers):
    """Penalise deviation from a scale ratio of ~3 between parent and child.

    Natural hierarchies (Alexander's 'levels of scale') follow approximately
    constant ratios between successive scales. The target ratio of 3 is the
    midpoint of the empirically observed range of 2-4.
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

    Bounds. After propagate_strength every strength satisfies
    s <= max(s0)/(1 - alpha), and edge weights lie in (0.1, 1], so
    E_R in [-(max(s0)/(1-alpha))^2, 0). The lower bound is attained only by the
    degenerate cluster in which all centres coincide at one scale (every pair an
    edge, every weight 1, every strength maximally reinforced). That bound is
    what fixes the weight on locality_energy in total_energy.
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
    """Penalise spatial overlap between centres.

    Computes mean pairwise Gaussian overlap exp(-d²/(r_i+r_j)²).
    Value is 1.0 when all centres are coincident, ~0 when well-separated.
    This prevents the degenerate minimum where all centres cluster at one point.

    This is the one term of the six that is not intensive, and the mean is a
    mean in name only: it divides by all N(N-1)/2 pairs while the pairs that
    contribute anything number O(N), so for a fixed centre density the value
    decays as 1/N. Its magnitude is roughly 7 * r_rms^2 / area, which is
    N-independent when the frame is fixed and more centres are found in it —
    the regime the corpus varies over — but not when the whole scene is scaled
    up. On real images it sits near 1e-3 and contributes ~0.3% of the total, so
    the residual is small in practice.

    It is left as it is because the alternatives trade one defect for another.
    Any normalisation that is genuinely intensive — total overlap per centre, or
    each centre's largest overlap — is O(1) on real centre sets, whose measured
    overlap load is 0.35 to 1.8 neighbours per centre, so it could not carry a
    barrier-sized weight without dominating the descriptive terms. Making this
    both a barrier and an intensive descriptor needs the two roles separated
    into two terms, which is a larger change than issue #14.
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
    """Penalise deviation from ideal child-area coverage of a parent (~0.65).

    Coverage C_i = sum(r_child^2) / r_parent^2. Values near 0.65 indicate
    children fill their parent region without overcrowding it.
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
    """Penalise children whose radial distance from parent deviates from 0.5 * r_parent.

    Alexander observed that child centres tend to lie at roughly 0.3-0.7 of the
    parent radius from the parent centre. Target is the midpoint, 0.5.
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


#: Relative priority of each descriptive term in the total, summing to 1. These
#: are the only free numbers in total_energy. They are the emphasis the original
#: weights were trying to express — hierarchy and reinforcement first, coverage
#: second, alignment and field smoothness third — carried over unchanged. What
#: has changed is that they are now *only* priorities: the conversion into
#: commensurable units is done separately, by SCALE below.
PRIORITY = {"H": 0.3, "R": 0.3, "C": 0.2, "A": 0.1, "F": 0.1}

#: Unit of each term: its standard deviation over a fixed reference ensemble.
#:
#: The five descriptive terms are measured in unrelated units — squared log
#: ratios, weight-times-strength products, squared area fractions, squared
#: radius fractions, squared field gradients — and their empirical spreads span
#: two orders of magnitude (0.0097 to 1.18). Adding them directly means the
#: weights are doing two incompatible jobs at once: unit conversion and
#: prioritisation. Dividing each term by its own spread does the unit conversion
#: on its own, after which PRIORITY does nothing but prioritise, and each term
#: contributes exactly its priority share of the variation of the total.
#:
#: The reference ensemble is the 33 non-degenerate cases of the audit corpus:
#: the six carpets in images/, the four null controls in audit/stimuli.py that
#: yield centres, and every frame of the five parametric sweeps, each at
#: --max-size 1024. "Non-degenerate" excludes the four cases that produce no
#: parent-child pair at all, for which H, C and A are not defined. Measured
#: 2026-08-16; re-measure with the same ensemble if the front end changes.
SCALE = {
    "H": 0.139057,  # (log(r_parent/r_child) - log 3)^2, per parent-child pair
    "R": 0.125571,  # -W_ij s_i s_j, per graph edge
    "C": 0.133911,  # (coverage - 0.65)^2, per parent
    "A": 1.178729,  # (d/r_parent - 0.5)^2, per parent-child pair
    "F": 0.009658,  # r_rms^2 * mean|grad phi|^2, dimensionless
}

#: Contraction factor of propagate_strength: strengths are bounded above by
#: max(s0)/(1 - PROPAGATION_ALPHA). Mirrors `alpha` in graph.propagate_strength.
PROPAGATION_ALPHA = 0.2

#: Upper bound on an intrinsic centre strength. build_structural_field
#: normalises the field to a maximum of 1 and detect_centers samples strengths
#: from it, so no centre enters the pipeline with s0 > 1.
MAX_INTRINSIC_STRENGTH = 1.0

#: Largest value the five descriptive terms sum to over the reference ensemble
#: (attained by contrast_field(1.0)). Their minimum there is -0.21 and their
#: mean 1.19.
REFERENCE_ENERGY_MAX = 1.8089

#: Weight on the locality barrier.
#:
#: locality_energy is not a descriptor of an image — on real inputs it is three
#: orders of magnitude below the descriptive terms and carries no information
#: about them. It is a barrier, and its weight follows from the one thing it has
#: to do: make the collapsed configuration, in which every centre sits at the
#: same point and the same scale, cost more than any real configuration.
#:
#: At full collapse locality_energy is exactly 1, and two things happen that
#: lower the rest of the total:
#:
#: 1. Reinforcement reaches its minimum — every pair is an edge, every weight is
#:    1, every strength is maximally reinforced. It is bounded below by
#:    -(max(s0)/(1 - alpha))^2 (see reinforcement_energy), so this is worth at
#:    most w_R * (max(s0)/(1 - alpha))^2 = 3.73.
#: 2. Hierarchy, coverage and alignment all go to *zero*, because
#:    assign_hierarchy needs a strictly larger centre to make a parent and there
#:    is none. Collapse does not merely score well on those three terms, it
#:    exempts itself from them.
#:
#: Point 2 is why a barrier sized only against reinforcement is not enough: at
#: w_L = 3.73 the collapsed configuration comes out 0.03 *below* a random one.
#: Requiring instead that collapse cost more than the worst configuration in the
#: reference ensemble gives
#:
#:     w_L = w_R * (max(s0)/(1 - alpha))^2 + max_ensemble(E_descriptive)
#:
#: which is the smallest weight for which collapse is uphill of every image
#: measured. The old value of 50.0 was nine times this.
WEIGHT_LOCALITY = (
    PRIORITY["R"] / SCALE["R"]
    * (MAX_INTRINSIC_STRENGTH / (1.0 - PROPAGATION_ALPHA)) ** 2
    + REFERENCE_ENERGY_MAX
)


def _rms_scale(centers):
    """Quadratic mean of centre radii — the characteristic length of a centre set."""
    if not centers:
        return 0.0
    return float(np.sqrt(np.mean([c.scale**2 for c in centers])))


def total_energy(field, centers, G):
    """Structural energy: a weighted sum of six dimensionless terms.

    Every term is a mean over the objects it is defined on — parent-child pairs
    for hierarchy and alignment, parents for coverage, graph edges for
    reinforcement, centre pairs for locality, pixels for the field term — so the
    total does not grow with the number of detected centres and energies are
    comparable between images. The five descriptive terms are intensive;
    locality is a barrier whose contribution on real inputs is about 0.3% of the
    total, and the one respect in which it is not intensive is set out in its
    own docstring.

    Previously hierarchy, coverage and alignment entered as *sums* over centres
    while reinforcement and locality entered as means. The total therefore
    tracked the centre count almost exactly (r = 0.99 across the audit corpus,
    E/n ~ 0.27), which made it a readout of the detector's sensitivity rather
    than a measure of the image. properties.py had already divided three of
    these by their pair counts; this is the same fix applied to the energy.

    The field term is additionally multiplied by the squared rms centre radius.
    Every other term is dimensionless, but mean|grad phi|^2 has units of
    1/pixel^2, so without this the total changes under a pure resize of the
    image — halving the resolution quadruples it. Scaling by the square of a
    length taken from the centre set itself removes the dependence on pixel
    size, leaving the shape of the reconstructed field. field_energy itself is
    left in raw units because properties.gradients reports it directly and has
    its own normaliser.

    The result is measured in units of the reference-ensemble standard
    deviation of a single term (see SCALE), so a change of 1 in the total is a
    change of about one ensemble standard deviation of the whole index. Over the
    six-carpet corpus it spans roughly 0.95 to 1.75.
    """
    npairs = sum(1 for c in centers if c.parent is not None)
    nparents = len({c.parent for c in centers if c.parent is not None})
    w = {k: PRIORITY[k] / SCALE[k] for k in PRIORITY}
    return (
        w["H"] * (hierarchy_energy(centers) / npairs if npairs else 0.0)
        + w["R"] * reinforcement_energy(G)
        + w["C"] * (coverage_energy(centers) / nparents if nparents else 0.0)
        + w["A"] * (alignment_energy(centers) / npairs if npairs else 0.0)
        + w["F"] * field_energy(field) * _rms_scale(centers) ** 2
        + WEIGHT_LOCALITY * locality_energy(centers)
    )
