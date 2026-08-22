"""Candidate global-redundancy discriminators for the noise problem (#29).

Everything in ``centres/`` derives its measures from *local* relations between
centres — a parent, a neighbour, an adjacent region. Section 2b of AUDIT.md
shows why that cannot separate a carpet from dense noise: local structure is not
scarce in noise, it is *abundant*. Noise outscores every carpet on the reported
degree of life, and — measured directly — on eleven of the fifteen individual
properties as well. Nothing computed one edge at a time distinguishes composed
structure from a dense random field, because both have edges, parents and
neighbours everywhere.

What a carpet has that noise does not is **redundancy**: the same few kinds of
centre recur across the whole image. A carpet is built from a small vocabulary
of scales and strengths; noise spreads across every scale and strength there is.
That is a *global* property of the centre population's distribution, not of any
local relation, so no per-edge measure sees it and the aggregate is blind to it.

Redundancy is one answer, but not the deepest one. The deepest is that a
composition is **one thing made of many**: the lesser centres nest under a
dominant whole. That is a *global* structural fact too, and it turns out to be the
strongest discriminator of all — see ``nesting`` below. This module measures three
global statistics, used by the ``discrimination`` stage of the audit: ``strength_entropy``
(redundancy), ``spatial_coherence`` (are the strengths arranged coherently in
space) and ``nesting`` (do the parts form a single whole). On the 44-artwork corpus
their single-axis rank separations against noise are 0.949, 0.977 and **1.000**.

**None is wired into the reported score.** A wholeness gate built on ``nesting`` was
measured to lift the score's own #29 separation from 0.131 to ~0.99 and to fix the
mechanical-grid interior-optimum at the same time, but it does not fully close #29
(two dense, hierarchy-fragmented artworks remain) and it redefines the score's
calibrated semantics; see AUDIT.md §17. So these stay candidates, ORed together in
the ``discrimination`` stage, recorded here with the evidence for and against each.

The three measures, and why each earns its place:

``strength_entropy`` — Shannon entropy of the centre-strength distribution.
  Separates all six carpets from noise (carpet 1.99–2.38, noise 2.53–2.74) with a
  clean margin, and — the property that matters — **holds under every practical
  transform**: mirror, rot90, gamma, JPEG, tone inversion, crop, pad, vignette,
  perspective and a resize to 512 px all leave the carpets below the noise floor
  (highest transformed carpet 2.39, noise floor 2.54). The histogram is ranged to
  the data, so a monotone rescaling of the strengths does not move it.

``scale_entropy`` — Shannon entropy of the log-blob-scale distribution.
  Separates the same six carpets even more cleanly at rest (carpet 1.41–1.60,
  noise 1.74–1.91), but it rides the detector's **absolute-pixel** scale ladder
  (``min_sigma=2, max_sigma=48``, in pixels not artwork units). Under a resize the
  detections shift along the ladder and clip at its ends, and the separation
  **breaks**: the worst transformed carpet reaches 1.743 against a noise floor of
  1.741. Making the ladder ``edge_spacing``-relative (#9) was tried to fix this and
  reverted — it does not deliver resolution invariance, because the driver is the
  detector resolving fewer centres at lower resolution, not the sigma range (see
  PLAN A2). So ``scale_entropy`` is kept only as a diagnostic; the usable axis is
  ``strength_entropy``, which needs no ladder change.

**Why neither is adopted yet.** Both separate carpets from noise monotonically,
but a mechanical lattice (``regular_grid``) sits at the *low*-entropy extreme,
*below* the carpets — a perfect grid is maximally redundant. So the target is not
"least entropy" but an **interior optimum**: Alexander's organised complexity,
living order between the rigid lattice and the random field. The location of that
optimum is an empirical quantity, and with a corpus of six Persian carpets it can
only be fitted, not measured — which is issue #34. Shipping a corpus-fitted
constant into the score is exactly the frame-versus-artwork error the project
invariant forbids. So the resolution of #29 is a redundancy term of this kind, once
#34 supplies a corpus wide enough (and varied enough — not six near-identical rugs)
to locate the optimum without fitting it. It builds on ``strength_entropy``, which
is transform-stable already; the crisper ``scale_entropy`` would have needed the
#9 ladder change, but that was attempted and reverted (above), so it is not the path.

**The caveat that decides whether any of this generalises — now partly tested.**
A carpet is hyper-redundant, so on the original six-carpet corpus "few recurring
strengths" and "alive" cannot be told apart. The corpus was widened to 32 real
CC-licensed artworks (17 carpets, 15 tile panels across many cultures; #34), and
the caveat bit: strength entropy alone dropped from a clean 1.000 to 0.997 — a bold
ancient-Egyptian geometric tile crossed the noise floor.

**The resolution is a second axis: spatial coherence.** ``strength_entropy`` asks
whether a *few strength values recur* (redundancy); ``spatial_coherence`` asks
whether the strengths are *arranged coherently in space* (Moran's I over nearest
neighbours). The two are complementary — a repetitive ornament has both, a
*non-repetitive* composed image has coherence without redundancy, and dense noise
has neither — and on the 32-artwork corpus they fail on *disjoint* artworks: the
Egyptian tile that entropy misses has strong spatial coherence, and the Varamin
that has spatially-flat strengths is redundant. The rule "alive = strength entropy
below the noise floor **OR** spatial coherence above the noise ceiling" separates
**all 32 artworks from noise, at rest and under every benign and practical
transform** (mirror, rot90, gamma, JPEG, invert, crop, vignette, resize-512),
tightest margin +0.074. This is the first robust clean separator on the wider
corpus, and it is an OR precisely so that a transform must break *both* an
artwork's redundancy and its coherence at once to misclassify it.

**The painting test is now run, and it locates the remaining limit.** The corpus was
widened again to 44 works with twelve non-repetitive Beardsley illustrations — the
"painting against noise" case. At native resolution the direction holds decisively:
all twelve separate from noise, six of them on spatial coherence *alone* (their
strength entropy sits above the noise floor), and with the figurative works added
``spatial_coherence`` becomes the single strongest axis (rank separation 0.977 vs
``strength_entropy``'s 0.949). But the coherence-only works have no redundancy
fallback, so they are **resolution-gated**: downscaled to 512 their coherence
collapses below the noise ceiling (``the_herald`` and ``et_in_arcadia_ego`` fail even
at identity@512), which is the #9 detection-count-versus-resolution dependence — not
a flaw in the rule — biting the figurative works harder than ornament.

So two things are still open, and neither is wired into the reported score. The OR
rule is robust on ornament at every resolution but on non-repetitive art only near
native resolution, so adopting it needs the detector to resolve a resolution-stable
set of centres first (#9, re-scoped to detection-count stability). And the
floor/ceiling are still constants — derived from noise rather than fitted to art,
which is more defensible, but constants. See AUDIT.md §17.
"""

from collections import Counter

import numpy as np
from scipy.spatial import cKDTree


def _entropy(values, bins=16):
    """Shannon entropy (nats) of a 1-D sample, histogram ranged to the data.

    Ranging the histogram to the observed min/max rather than to a fixed absolute
    range is deliberate: it makes the statistic invariant to a monotone affine
    rescaling of the inputs, so a field whose strengths are all multiplied by a
    constant — which a change of normalisation or exposure can do — reports the
    same entropy. What it measures is the *shape* of the distribution: how spread
    the population is across whatever range it occupies, not where that range sits.
    """
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return None
    hist, _ = np.histogram(v, bins=bins)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log(p)).sum())


def strength_entropy(centers, bins=16):
    """Entropy of the centre-strength distribution — the transform-stable discriminator.

    Low for a redundant population (a few recurring strengths), high for a diffuse
    one. Carpets sit below noise and stay there under every practical transform;
    see the module docstring for the measured ranges. ``None`` for fewer than two
    centres, where there is no distribution to speak of.
    """
    if len(centers) < 2:
        return None
    return _entropy([c.strength for c in centers], bins)


def scale_entropy(centers, bins=16):
    """Entropy of the log-blob-scale distribution — separates, but is transform-fragile.

    Cleaner than ``strength_entropy`` at rest, but built on the absolute-pixel
    scale ladder, so a resize moves it enough to erase the carpet/noise gap. Making
    the ladder relative (#9) was tried and reverted (PLAN A2 — the ladder is not the
    cause), so this stays a diagnostic only; ``strength_entropy`` is the usable axis.
    """
    if len(centers) < 2:
        return None
    return _entropy(np.log(np.array([c.scale for c in centers]) + 1e-6), bins)


def spatial_coherence(centers, k=6):
    """Moran's I of centre strength over k-nearest-neighbour adjacency — the
    complement of ``strength_entropy``, and the axis that should survive
    non-repetitive art.

    ``strength_entropy`` asks whether a *few strength values recur* (redundancy).
    This asks whether the strengths are *arranged coherently in space*: do
    neighbouring centres resemble each other? It is Moran's I, the standard index
    of spatial autocorrelation — near +1 when nearby centres have similar
    strengths, near 0 when they are placed at random, which is what dense noise is
    (measured: noise 0.02–0.07, artworks 0.10–0.45, with one spatially-flat carpet
    at ≈0 that redundancy catches instead).

    ``k`` nearest neighbours *by position*, not the reinforcement graph: position
    adjacency is dimensionless and scale-free, so the statistic does not inherit the
    graph's edge-admission threshold and holds under a resize. ``None`` for fewer
    than ``k + 2`` centres, where the neighbourhood is not defined.
    """
    if len(centers) < k + 2:
        return None
    pos = np.array([[c.x, c.y] for c in centers], dtype=float)
    x = np.array([c.strength for c in centers], dtype=float)
    if float(np.ptp(x)) <= 1e-12:   # every strength identical: nothing to correlate
        return 0.0
    z = x - x.mean()
    zz = float(z @ z)
    _, idx = cKDTree(pos).query(pos, k=k + 1)   # column 0 is the point itself
    neighbours = z[idx[:, 1:]]                  # (n, k)
    return float((z[:, None] * neighbours).sum() / (k * zz))


def nesting(centers):
    """Fraction of centres in the single largest containment tree — 'multiple
    things making one thing', and the strongest discriminator of the three.

    A composition is *one* thing built from many: the lesser centres nest under a
    dominant containing centre, which nests under a larger one, up to the whole.
    This walks ``Center.parent`` (assigned by ``assign_hierarchy`` — the smallest
    centre whose extent contains each) as an undirected forest and returns the
    share of centres in its biggest tree. It is Alexander's actual claim, that a
    whole is a single centre all lesser centres support, made countable.

    Why it beats both entropy and coherence. Noise has almost no nesting — its
    centres are all one size, so none contains a meaningful share of the rest
    (measured 0.01–0.06); a *mechanical grid* is flat for the same reason (≈0.10),
    which is why nesting does **not** fall into the interior-optimum trap that
    ``strength_entropy`` does — a grid is middling here, not maximal. Every one of
    the 44 artworks nests more than every noise field (single-axis rank separation
    1.000 at native resolution, against 0.977 for coherence and 0.949 for entropy),
    and — because containment is measured in relative *extent*, not pixel
    neighbourhoods — it survives downscaling where local coherence collapses: the
    two paintings whose coherence the local axis loses at 512 (``the_herald``,
    ``et_in_arcadia_ego``) keep a wide nesting margin under every practical
    transform, and ``varamin``, which *both* coherence and entropy miss, is caught
    by nesting because it is still one deeply-nested carpet.

    Its one weakness is *dense* images: when the detector resolves many hundreds of
    centres the hierarchy fragments into many small trees, so nesting reads low even
    for real art (``ghashghai`` 0.06 at n=678, ``tile_panel_delft`` 0.07 at n=463).
    That is the detection-count/hierarchy-fragmentation limit (#9), not a flaw in
    the measure, and it is what keeps nesting — and the score-integration built on
    it — from fully closing #29. See AUDIT.md §17. ``None`` for fewer than two
    centres, where there is nothing to nest.
    """
    n = len(centers)
    if n < 2:
        return None
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, c in enumerate(centers):
        p = getattr(c, "parent", None)
        if p is not None:
            ra, rb = find(i), find(int(p))
            if ra != rb:
                parent[ra] = rb
    return max(Counter(find(i) for i in range(n)).values()) / n


def or_rule(axes):
    """Generalised 'A OR B OR …' separation of art from noise over several axes.

    ``axes`` is a list of ``(art_values, noise_values, higher_is_life)``. Each axis
    contributes a bound taken from the noise cloud — its ``max`` when higher is life
    (a ceiling to clear), its ``min`` when lower is life (a floor to stay under) —
    so no bound is fitted to the art it judges. An artwork is separated if it clears
    *any* axis; its margin is the best (``max``) of the per-axis margins.

    Returns ``(fraction_separated, tightest_margin)``: the fraction of artworks that
    clear at least one axis, and the smallest margin over the artworks — how close
    the whole rule comes to failing. An OR because a transform must break *every*
    axis at once to misclassify an artwork, and the axes fail on disjoint artworks.
    """
    bounds = []
    for art_v, noise_v, higher in axes:
        nz = [v for v in noise_v if v is not None]
        bounds.append(max(nz) if higher else min(nz))
    n_art = len(axes[0][0])
    margins = []
    for i in range(n_art):
        per_axis = []
        for (art_v, _, higher), bound in zip(axes, bounds):
            v = art_v[i]
            if v is None:
                continue
            per_axis.append((v - bound) if higher else (bound - v))
        if per_axis:
            margins.append(max(per_axis))
    if not margins:
        return None, None
    return sum(m > 0 for m in margins) / len(margins), min(margins)


def combined_separation(art_se, art_mo, noise_se, noise_mo):
    """The 'redundancy OR spatial coherence' rule — the two-axis case of ``or_rule``.

    An artwork counts as separated from noise if its strength entropy is below the
    noise floor (``min`` over the noise samples) *or* its spatial coherence is above
    the noise ceiling (``max`` over them). Both bounds are taken from noise, not from
    art, so the rule is not fitted to the corpus it judges. Kept as a named
    two-axis entry point; the ``discrimination`` stage now ORs in ``nesting`` as a
    third axis via ``or_rule`` directly.

    Returns ``(fraction_separated, tightest_margin)``. On the 44-artwork corpus this
    is 1.0 and +0.154 at rest.
    """
    return or_rule([(art_se, noise_se, False), (art_mo, noise_mo, True)])


def pairwise_order(art, noise, life_is_higher_for_art=True):
    """Fraction of (art, noise) pairs the statistic orders the way life demands.

    The rank-separation figure the ``discrimination`` stage reports: 1.0 means
    every artwork is scored as more alive than every noise field, 0.0 the reverse,
    0.5 no separation at all. Robust to the scale of the statistic and to outliers,
    which a difference of means is not.
    """
    a = [x for x in art if x is not None]
    z = [x for x in noise if x is not None]
    if not a or not z:
        return None
    if life_is_higher_for_art:
        correct = sum(1 for x in a for y in z if x > y)
    else:
        correct = sum(1 for x in a for y in z if x < y)
    return correct / (len(a) * len(z))
