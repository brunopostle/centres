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
"""

import numpy as np


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
