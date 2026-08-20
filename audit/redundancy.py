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

This module measures that redundancy two ways and is used by the ``discrimination``
stage of the audit. **Neither is wired into the reported score.** They are
candidates, recorded here with the evidence for and against each, because
adopting one into the degree of life is blocked on two open issues (see below).

The two measures, and why one is usable and one is not *yet*:

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
  (``min_sigma=2, max_sigma=48``; issue #9 is open precisely because that ladder
  is in pixels, not in units of the artwork). Under a resize the detections shift
  along the ladder and clip at its ends, and the separation **breaks**: the worst
  transformed carpet reaches 1.743 against a noise floor of 1.741. It is kept here
  as a diagnostic and as the reason #9 blocks the clean version of the fix.

**Why neither is adopted yet.** Both separate carpets from noise monotonically,
but a mechanical lattice (``regular_grid``) sits at the *low*-entropy extreme,
*below* the carpets — a perfect grid is maximally redundant. So the target is not
"least entropy" but an **interior optimum**: Alexander's organised complexity,
living order between the rigid lattice and the random field. The location of that
optimum is an empirical quantity, and with a corpus of six Persian carpets it can
only be fitted, not measured — which is issue #34. Shipping a corpus-fitted
constant into the score is exactly the frame-versus-artwork error the project
invariant forbids. So the resolution of #29 is: a redundancy term of this kind,
once #9 makes the scale version stable and #34 supplies a corpus wide enough to
locate the optimum without fitting it to six near-identical rugs.

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

Two things are still open, and neither is wired into the reported score. The corpus
is **ornament only**, so the decisive test — a *painting* against noise — is still
not run; but the spatial-coherence axis is exactly what a painting (coherent, not
repetitive) would be caught by, and the least-repetitive pieces here (a pictorial
Delft tile, the Egyptian geometric) are already carried by it, which is real
evidence for generalisation. And the floor/ceiling are still constants — derived
from noise rather than fitted to art, which is more defensible, but constants.
See AUDIT.md §17.
"""

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
    scale ladder (#9), so a resize moves it enough to erase the carpet/noise gap.
    Kept as a diagnostic, not a candidate for adoption until #9 lands.
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


def combined_separation(art_se, art_mo, noise_se, noise_mo):
    """The 'redundancy OR spatial coherence' rule, evaluated against the noise cloud.

    An artwork counts as separated from noise if its strength entropy is below the
    noise floor (``min`` over the noise samples) *or* its spatial coherence is above
    the noise ceiling (``max`` over them). Both bounds are taken from noise, not from
    art, so the rule is not fitted to the corpus it judges.

    Returns ``(fraction_separated, tightest_margin)``. The margin is
    ``max(floor - entropy, coherence - ceiling)`` — positive when at least one axis
    clears the noise cloud — and its minimum over the artworks is how close the rule
    comes to failing. On the 32-artwork corpus this is 1.0 and +0.15 at rest, and
    stays 1.0 with a tightest margin of +0.074 across all benign/practical transforms.
    """
    floor = min(v for v in noise_se if v is not None)
    ceiling = max(v for v in noise_mo if v is not None)
    margins = [max(floor - se, mo - ceiling)
               for se, mo in zip(art_se, art_mo) if se is not None and mo is not None]
    if not margins:
        return None, None
    return sum(m > 0 for m in margins) / len(margins), min(margins)


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
