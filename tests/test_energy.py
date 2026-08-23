import math
import numpy as np
import pytest
from centres.centers import Center
from centres.energy import (
    WEIGHT_LOCALITY,
    alignment_energy,
    coverage_energy,
    degree_of_life,
    field_energy,
    hierarchy_energy,
    largest_containment_fraction,
    life_terms,
    locality_energy,
    reinforcement_energy,
    total_energy,
    wholeness,
)
from centres.graph import build_graph, propagate_strength
from centres.field import reconstruct_field
from centres.pipeline import assign_hierarchy, evolve, random_centers


def c(id, x, y, scale, strength=1.0, parent=None):
    return Center(
        id=id,
        x=float(x),
        y=float(y),
        scale=float(scale),
        strength=strength,
        parent=parent,
    )


# --- hierarchy_energy ---


def test_hierarchy_zero_at_ratio_three():
    parent = c(0, 0, 0, 30.0)
    child = c(1, 5, 5, 10.0, parent=0)
    assert hierarchy_energy([parent, child]) == pytest.approx(0.0, abs=1e-10)


def test_hierarchy_nonzero_wrong_ratio():
    parent = c(0, 0, 0, 10.0)
    child = c(1, 0, 0, 10.0, parent=0)  # ratio = 1, not 3
    assert hierarchy_energy([parent, child]) > 0


def test_hierarchy_zero_no_parents():
    centers = [c(i, 0, 0, float(i + 1)) for i in range(4)]
    assert hierarchy_energy(centers) == 0.0


# --- reinforcement_energy ---


def test_reinforcement_more_negative_with_higher_strength():
    G_weak = build_graph([c(0, 0, 0, 10, strength=0.1), c(1, 15, 0, 10, strength=0.1)])
    G_strong = build_graph([c(0, 0, 0, 10, strength=2.0), c(1, 15, 0, 10, strength=2.0)])
    assert reinforcement_energy(G_strong) < reinforcement_energy(G_weak)


def test_reinforcement_zero_no_edges():
    G = build_graph([c(0, 0, 0, 3), c(1, 500, 0, 3)])
    assert not G.has_edge(0, 1)
    assert reinforcement_energy(G) == 0.0


def test_reinforcement_negative_when_edges_exist():
    G = build_graph([c(0, 0, 0, 10, strength=1.0), c(1, 15, 0, 10, strength=1.0)])
    assert G.has_edge(0, 1)
    assert reinforcement_energy(G) < 0.0


# --- coverage_energy ---


def test_coverage_zero_at_ideal():
    # child scale = sqrt(0.65) * parent_scale → coverage exactly 0.65
    child_scale = math.sqrt(0.65) * 10.0
    parent = c(0, 0, 0, 10.0)
    child = c(1, 3, 0, child_scale, parent=0)
    assert coverage_energy([parent, child]) == pytest.approx(0.0, abs=1e-6)


def test_coverage_nonzero_overcrowded():
    # Many large children → coverage >> 0.65
    parent = c(0, 0, 0, 10.0)
    children = [c(i + 1, float(i), 0, 8.0, parent=0) for i in range(5)]
    assert coverage_energy([parent] + children) > 0


def test_coverage_zero_no_children():
    centers = [c(i, 0, 0, 10.0) for i in range(3)]
    assert coverage_energy(centers) == 0.0


# --- alignment_energy ---


def test_alignment_zero_at_half_radius():
    parent = c(0, 0, 0, 20.0)
    child = c(1, 10, 0, 5.0, parent=0)  # dist=10 = 0.5 * 20
    assert alignment_energy([parent, child]) == pytest.approx(0.0, abs=1e-10)


def test_alignment_nonzero_at_center():
    parent = c(0, 0, 0, 20.0)
    child = c(1, 0, 0, 5.0, parent=0)  # dist=0, d=0, far from 0.5
    assert alignment_energy([parent, child]) > 0


def test_alignment_zero_no_parents():
    centers = [c(i, float(i * 10), 0, 10.0) for i in range(3)]
    assert alignment_energy(centers) == 0.0


# --- field_energy ---


def test_field_energy_zero_for_uniform():
    assert field_energy(np.ones((50, 50))) == pytest.approx(0.0, abs=1e-10)


def test_field_energy_positive_for_gradient():
    field = np.zeros((50, 50))
    field[:, 25:] = 1.0  # step discontinuity
    assert field_energy(field) > 0


def test_field_energy_higher_for_sharper_gradient():
    gentle = np.linspace(0, 1, 50)[np.newaxis, :] * np.ones((50, 1))
    sharp = np.zeros((50, 50))
    sharp[:, 25:] = 1.0
    assert field_energy(sharp) > field_energy(gentle)


# --- total_energy ---


def test_total_energy_finite():
    centers = [c(0, 30, 30, 20.0), c(1, 35, 30, 6.0, parent=0)]
    G = build_graph(centers)
    G = propagate_strength(G)
    field = reconstruct_field((60, 60), centers)
    assert np.isfinite(total_energy(field, centers, G))


def _random_centers(n, side, seed, scale_range=(2.8, 16.8)):
    """n centres drawn from a fixed distribution over a side x side canvas."""
    rng = np.random.default_rng(seed)
    lo, hi = np.log(scale_range[0]), np.log(scale_range[1])
    centers = [
        c(i, rng.uniform(0, side), rng.uniform(0, side),
          float(np.exp(rng.uniform(lo, hi))),
          strength=float(rng.uniform(0.3, 1.0)))
        for i in range(n)
    ]
    return assign_hierarchy(centers)


def _energy_of(centers, shape):
    G = propagate_strength(build_graph(centers))
    return total_energy(reconstruct_field(shape, centers), centers, G)


def _life_of(centers, shape):
    centers = assign_hierarchy(centers)
    G = propagate_strength(build_graph(centers))
    return degree_of_life(reconstruct_field(shape, centers), centers, G)


def test_energy_is_minus_the_degree_of_life():
    """The two are one quantity with two signs, and nothing sits between them.

    `evolve()` minimises E because simulated annealing descends; everything
    reported — CLI, --json, GUI, the audit harness — is L = -E, which is zero
    for a configuration with no structure and higher for more (#28).
    """
    centers = _random_centers(20, 200.0, seed=11)
    G = propagate_strength(build_graph(centers))
    field = reconstruct_field((200, 200), centers)
    assert total_energy(field, centers, G) == -degree_of_life(field, centers, G)


# --- issue #28: the functional's minimum must not be the absence of structure ---


def test_empty_configuration_scores_exactly_zero():
    """No centres at all is the neutral baseline, not the optimum.

    Exactly zero rather than merely small: the optimiser needs a fixed point of
    reference, and "a sum with no terms is 0" is what made emptiness the global
    minimum in the first place. Here it is zero because *participation* is zero.
    """
    G = build_graph([])
    assert degree_of_life(np.zeros((50, 50)), [], G) == 0.0
    assert total_energy(np.zeros((50, 50)), [], G) == 0.0


def test_structureless_configuration_scores_zero():
    """The exact configuration #28 was opened about.

    36 centres at one identical scale, spaced far apart: identical scales admit
    no parent, so there are no parent-child pairs; the spacing is far beyond the
    reinforcement kernel's reach, so there are no graph edges; and there is no
    overlap. Under the old functional this scored E = +0.081, *better* than every
    real artwork. It must now score 0 — no structure, no life.
    """
    spacing, scale, k = 220.0, 10.0, 6
    centers = assign_hierarchy(
        [c(i, spacing * (1 + i % k), spacing * (1 + i // k), scale) for i in range(k * k)]
    )
    shape = (int(spacing * (k + 1)),) * 2
    G = propagate_strength(build_graph(centers))
    assert sum(1 for x in centers if x.parent is not None) == 0
    assert G.number_of_edges() == 0
    assert locality_energy(centers) < 1e-12
    assert degree_of_life(reconstruct_field(shape, centers), centers, G) == pytest.approx(
        0.0, abs=1e-9
    )


def _nested_hierarchy():
    """A genuine whole: one root, four children at a 3:1 ratio and half-radius, each
    with two grandchildren — all nesting into a single containment tree."""
    centers = [c(0, 200, 200, 60.0)]
    idx = 1
    for dx, dy in ((30, 0), (-30, 0), (0, 30), (0, -30)):
        cx, cy = 200 + dx, 200 + dy
        centers.append(c(idx, cx, cy, 20.0))
        idx += 1
        for ddx in (10, -10):
            centers.append(c(idx, cx + ddx, cy, 7.0))
            idx += 1
    return centers


def test_composed_whole_scores_above_structurelessness():
    """A configuration that forms one whole beats one with none — #28's third
    constraint, under the #29 wholeness gate.

    The gate multiplies the descriptive sum by wholeness, so what beats
    structurelessness is not *any* relationships but relationships that nest into a
    single thing. A three-level containment hierarchy (``_nested_hierarchy``) does,
    and scores well above zero; a structureless scatter of equal, far-apart centres
    has no relationships at all and scores zero.
    """
    spacing, k = 220.0, 6
    structureless = [
        c(i, spacing * (1 + i % k), spacing * (1 + i // k), 10.0) for i in range(k * k)
    ]
    assert _life_of(_nested_hierarchy(), (400, 400)) > _life_of(
        structureless, (int(spacing * (k + 1)),) * 2
    ) + 0.3


def test_flat_lattice_is_not_alive_under_the_wholeness_gate():
    """The theory change #29 brought: a flat lattice is reinforced but not a whole.

    A grid of identical, equal-scale centres has adjacency and reinforcement but no
    containment hierarchy — nothing nests under anything — so its wholeness is zero
    and the gate takes its degree of life to ~0. Before the gate this lattice scored
    +0.30, *above* several carpets: it was a face of the interior-optimum failure
    (#29, §3 of AUDIT.md). Living order requires one whole, not merely local
    relationships.
    """
    lattice = [c(i, 50.0 * (1 + i % 6), 50.0 * (1 + i // 6), 12.0) for i in range(36)]
    assert _life_of(lattice, (350, 350)) == pytest.approx(0.0, abs=1e-3)


def test_largest_containment_fraction_one_for_a_nested_whole():
    """All centres in one containment tree -> fraction 1.0; a flat grid -> low."""
    assert largest_containment_fraction(assign_hierarchy(_nested_hierarchy())) == pytest.approx(1.0)
    lattice = [c(i, 50.0 * (1 + i % 6), 50.0 * (1 + i // 6), 12.0) for i in range(36)]
    assert largest_containment_fraction(assign_hierarchy(lattice)) < 0.1
    assert largest_containment_fraction([]) == 0.0


def test_wholeness_zero_for_flat_and_empty_positive_for_nested():
    """The gate: 0 for fewer than two centres and for anything no more nested than
    its random-field null (a flat grid), positive for a genuine hierarchy."""
    assert wholeness([]) == 0.0
    assert wholeness([c(0, 0, 0, 10.0)]) == 0.0
    lattice = [c(i, 50.0 * (1 + i % 6), 50.0 * (1 + i // 6), 12.0) for i in range(36)]
    assert wholeness(assign_hierarchy(lattice)) == 0.0
    assert wholeness(assign_hierarchy(_nested_hierarchy())) > 0.5


def test_wholeness_is_count_invariant_on_random_fields():
    """The property that lets it gate the score without reintroducing the #14 count
    confound: subtracting the random-field baseline leaves wholeness flat in n."""
    ns, ws = [], []
    for n in (32, 64, 128, 256):
        side = 24.5 * math.sqrt(n)
        for seed in range(3):
            ns.append(n)
            ws.append(wholeness(_random_centers(n, side, seed)))
    assert abs(float(np.corrcoef(ns, ws)[0, 1])) < 0.5


def test_participation_is_a_fraction_and_is_zero_when_nothing_participates():
    """The half of each term that makes the empty case zero."""
    spacing, k = 220.0, 6
    centers = assign_hierarchy(
        [c(i, spacing * (1 + i % k), spacing * (1 + i // k), 10.0) for i in range(k * k)]
    )
    shape = (int(spacing * (k + 1)),) * 2
    G = propagate_strength(build_graph(centers))
    for key, (p, _q, contribution) in life_terms(
        reconstruct_field(shape, centers), centers, G
    ).items():
        assert p == 0.0, key
        assert contribution == 0.0, key

    related = _random_centers(40, 200.0, seed=5)
    G2 = propagate_strength(build_graph(related))
    for key, (p, q, _c) in life_terms(
        reconstruct_field((200, 200), related), related, G2
    ).items():
        assert 0.0 <= p <= 1.0, key
        assert 0.0 <= q <= 1.0, key


def test_degree_of_life_does_not_track_centre_count():
    """Cramming must not pay: #28's third constraint, and #14's acceptance test.

    Centre sets are drawn at *constant spatial density* — the canvas area grows
    with n — so every per-pair, per-parent and per-edge statistic is the same in
    distribution at every n, and an intensive score must be flat in n. Before
    #14, hierarchy, coverage and alignment entered as sums over centres and the
    total grew almost exactly linearly with n (r = 0.99 over the audit corpus).

    The remaining +0.14 is the descriptive terms, which are flat to within
    0.03 in the mean. The locality barrier used to contribute most of it: it is
    a mean over all N(N-1)/2 pairs while only O(N) overlap, so it decays as 1/N
    and, entering linearly, took this correlation to +0.55 on its own. It enters
    squared for that reason — see locality_energy.
    """
    ns, lives = [], []
    for n in (32, 64, 128, 256):
        side = 24.5 * math.sqrt(n)  # constant density: area proportional to n
        shape = (int(side), int(side))
        for seed in (0, 1, 2):
            ns.append(n)
            lives.append(_life_of(_random_centers(n, side, seed), shape))
    r = float(np.corrcoef(ns, lives)[0, 1])
    assert abs(r) < 0.5, f"degree of life still tracks centre count: r = {r:+.3f}"
    by_n = {n: np.mean([v for m, v in zip(ns, lives) if m == n]) for n in (32, 64, 128, 256)}
    assert max(by_n.values()) - min(by_n.values()) < 0.1, by_n


def test_degree_of_life_invariant_under_uniform_rescaling():
    """Doubling every length must not change the score.

    Every deviation is dimensionless except the field term, whose mean squared
    gradient carries units of 1/pixel^2; it is scaled by the squared rms centre
    radius for exactly this reason.
    """
    base = _random_centers(24, 160.0, seed=7)
    scaled = [c(x.id, x.x * 2, x.y * 2, x.scale * 2, x.strength, x.parent) for x in base]
    e1 = _energy_of(base, (160, 160))
    e2 = _energy_of(scaled, (320, 320))
    assert e2 == pytest.approx(e1, rel=0.05)


def test_collapse_loses_by_a_clear_margin():
    """All centres at one point and one scale is the floor, not the optimum.

    It was the optimum until the reinforcement kernel was moved from coincidence
    to adjacency, and after that it still sat only 0.004 from a lattice, because
    hierarchy, coverage and alignment all *vanish* at collapse — identical
    scales admit no parent, so collapse exempted itself from every deviation
    penalty. Under participation x quality it earns nothing from any term and
    pays the full locality barrier.
    """
    spread = _random_centers(30, 300.0, seed=3)
    collapsed = assign_hierarchy(
        [c(i, 150.0, 150.0, 20.0, strength=x.strength) for i, x in enumerate(spread)]
    )
    assert locality_energy(collapsed) == pytest.approx(1.0)
    L_collapsed = _life_of(collapsed, (300, 300))
    assert L_collapsed == pytest.approx(-WEIGHT_LOCALITY)
    assert _life_of(spread, (300, 300)) - L_collapsed > 0.5


def test_concentric_collapse_loses():
    """The degeneracy the barrier is actually still needed for.

    Centres stacked on one point at scales in a 3:1 ladder *do* have
    parent-child pairs, so unlike a plain collapse they collect hierarchy and
    alignment reward for a configuration with no spatial extent at all. Without
    the barrier this scores +0.196, above a lattice at +0.299 only by a little
    and inside the corpus range once jittered.
    """
    ladder = [c(i, 150.0, 150.0, min(2.5 * 3.0 ** (i % 5), 80.0)) for i in range(40)]
    lattice = [c(i, 50.0 * (1 + i % 6), 50.0 * (1 + i // 6), 12.0) for i in range(36)]
    assert _life_of(ladder, (300, 300)) < _life_of(lattice, (350, 350))
    assert _life_of(ladder, (300, 300)) < 0.0


def _degenerate_proposals(centers, shape, rng):
    """Replace the configuration with an outright degenerate one, in place."""
    h, w = shape
    if rng.random() < 0.5:
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        s = float(np.exp(rng.uniform(np.log(4), np.log(40))))
        for x in centers:
            x.x, x.y, x.scale = cx, cy, s
    else:  # spread out at one identical scale — #28's structureless optimum
        s = float(np.exp(rng.uniform(np.log(4), np.log(20))))
        k = int(np.ceil(np.sqrt(len(centers))))
        for i, x in enumerate(centers):
            x.x = w * (i % k + 0.5) / k
            x.y = h * (i // k + 0.5) / k
            x.scale = s


def _anneal_aggressively(shape, n, iterations, seed, pos_sigma, log_scale_sigma):
    """Anneal the objective with moves large enough to reach the degeneracies.

    ``evolve()``'s own move set is +-2 px and x exp(N(0, 0.02)) — far too weak
    to cross the canvas or to reach a common scale within a few hundred
    iterations, which is exactly why #28's defect went unnoticed for so long. A
    test that only ran ``evolve()`` would therefore prove nothing about the
    objective. This move set jumps a seventh of the canvas at a time, rescales
    by up to a factor of e, and every tenth iteration proposes an outright
    degenerate configuration.
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    np.random.seed(seed)
    centers = assign_hierarchy(random_centers(n, shape))
    intrinsic = [x.strength for x in centers]

    def L(cs):
        for x, s in zip(cs, intrinsic):
            x.strength = s
        cs = assign_hierarchy(cs)
        return degree_of_life(
            reconstruct_field(shape, cs), cs, propagate_strength(build_graph(cs))
        )

    current = L(centers)
    for t in range(iterations):
        T = 1.0 * 0.01 ** (t / iterations)
        saved = [(x.x, x.y, x.scale, x.parent) for x in centers]
        if rng.random() < 0.1:
            _degenerate_proposals(centers, shape, rng)
        else:
            for x in centers:
                x.x = float(np.clip(x.x + rng.normal(0, pos_sigma), 0, w))
                x.y = float(np.clip(x.y + rng.normal(0, pos_sigma), 0, h))
                x.scale = float(
                    np.clip(x.scale * np.exp(rng.normal(0, log_scale_sigma)), 2, 80)
                )
        new = L(centers)
        delta = current - new  # descending E is ascending L
        if delta < 0 or rng.random() < np.exp(-delta / (T + 1e-10)):
            current = new
        else:
            for x, (px, py, ps, pp) in zip(centers, saved):
                x.x, x.y, x.scale, x.parent = px, py, ps, pp
    return current, centers


def _rms_spread(centers):
    p = np.array([[x.x, x.y] for x in centers])
    return float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean()))


def test_evolve_converges_to_a_non_degenerate_configuration():
    """The default move set, as shipped."""
    shape = (200, 200)
    np.random.seed(0)
    _field, centers = evolve(shape=shape, iterations=120, n_centers=25)
    assert _rms_spread(centers) > 0.2 * shape[0]
    assert locality_energy(centers) < 0.5
    assert _life_of(centers, shape) > 0.0


def test_aggressive_annealing_does_not_find_a_degenerate_optimum():
    """Criterion 5 of #28, and the one that says the objective is sound.

    A move set strong enough to reach both degeneracies in one step, and which
    proposes them outright every tenth iteration, still ends spread out and
    multi-scale — and ends *above* both of them. That is a property of the
    objective, not of the annealer's reach.
    """
    shape = (200, 200)
    L, centers = _anneal_aggressively(
        shape, n=25, iterations=250, seed=0, pos_sigma=30.0, log_scale_sigma=0.5
    )
    assert _rms_spread(centers) > 0.2 * shape[0]
    assert max(x.scale for x in centers) / min(x.scale for x in centers) > 3.0
    assert locality_energy(centers) < 0.5

    collapsed = [c(i, 100.0, 100.0, 15.0) for i in range(25)]
    k = 5
    equal_scale = [
        c(i, 200.0 * (i % k + 0.5) / k, 200.0 * (i // k + 0.5) / k, 6.0) for i in range(25)
    ]
    assert L > _life_of(collapsed, shape) + 0.5
    assert L > _life_of(equal_scale, shape)
