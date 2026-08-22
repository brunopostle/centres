# Audit: what the 15 measures actually measure

This document records an empirical audit of the measures in `centres/properties.py`.
It was written because the mathematics in [THEORY.md](THEORY.md) had accumulated
through several rounds of revision without any test that could tell an improvement
from a regression.

Everything below is reproducible with `python -m audit`.

> ### How to read this document
>
> It grew as an investigation, oldest first, so it is part history and part
> current state. **§0 below is the current state** — the summary that supersedes
> the older headlines. **§12 (sweeps), §13 (triage), §14-16, and §17 (the noise
> problem, #29) are current**, last refreshed against the full `python -m audit`
> after the #22 redefinitions.
> **§1-11 are the original pre-repair findings**, kept as the record of how the
> pipeline got here; each is superseded in kind by a later section. Where an old
> number and a §0/§12/§13 number disagree, the later one is right.

## 0. Current state, after the repairs and redefinitions

The audit began by showing that **one of the fifteen measures tracked the quantity
it was named for** on stimuli where the answer is known by construction; the rest
measured something else, five running backwards. The cause was rarely the formula:
it was a front end feeding the measures an unstable centre set, and the measures
computing a different quantity from the property named on them.

After the phase A/B repairs and the #22 redefinitions against the sourced
definitions (Salingaros 2025, in `docs/`):

**The instrument is precise.** Scores are stable under isometries (worst delta
0.21, was 5.9), independent of frame, resolution and iteration count, bounded, and
`|r(score, centre count)| = 0.01` over the full ensemble (was 0.99). A blank
canvas scores *undefined* on every property, not 10/10. Collapse is no longer the
energy's global minimum.

**Ten measures now track their ground truth** after controlling for centre count
(partial ρ from §16, or a clean interior optimum), where one did at the start:

| measure | count-controlled ρ | field SNR |
|---|---:|---:|
| contrast | +1.00 | 4.43 |
| deep interlock | +1.00 | 4.26 |
| gradients | +0.96 | 6.24 |
| strong centres | +0.99 | 3.22 |
| positive space | +0.90 | 1.55 |
| echoes | **−0.89** | 0.68 |
| not-separateness | **+0.83** | 1.08 |
| simplicity | +0.71 | 3.72 |
| levels of scale | interior optimum (peaks at 3) | 3.18 |
| boundaries | interior optimum (peaks at 1/3) | 1.83 |

`echoes` and `not-separateness` moved into this group as a *side effect of the #30
detector fix*: local contrast normalisation made the faint shape-vocabulary and
bleed stimuli detectable, taking echoes from |0.17| to |0.89| and not-separateness
from +0.24 to +0.83. (echoes correctly runs *negative* — more distinct shapes
means less echo — which the harness now reports as tracking-downward rather than
mislabelling as a failure; #33.)

**Three track moderately:** roughness (+0.51), local symmetries (−0.46), good
shape (+0.39). `good_shape` weakened under count control this run — its raw +0.91
is substantially count-driven — so it is no longer a clean keep. **One fails:**
alternating repetition (+0.14). **One is spurious:** the void (+0.80 raw, partial
−0.07). See §13.

**A new caveat from this run: field SNR is now the binding constraint, not
tracking.** Four measures that track their ground truth (echoes, not-separateness,
positive space, boundaries) have poor between-artwork SNR on the six carpets —
they discriminate the constructed stimuli cleanly and separate real carpets
weakly. (The image-domain `boundaries`/`deep_interlock` used to use a fixed
grey-128 threshold, so they shifted under gamma and JPEG, inflating their noise
floor; #31 replaced it with a **symmetrised Otsu** threshold that adapts to the
image's own histogram, cutting the gamma sensitivity of `boundaries` from 2.9 to
0.25 and of `deep_interlock` from 1.7 to 0.46 on the 0–10 scale while keeping the
exact tone-inversion invariance #30 established and the ground-truth sweeps
unchanged — boundaries still peaks at 0.3, deep interlock still tracks at +1.000.)

**Two findings are unchanged, and are the honest residual:**

1. **Noise still scores a higher degree of life than any of the six carpets**
   (section 2b, [#29](https://github.com/brunopostle/centres/issues/29)). The
   redefined measures fixed *what each property measures*; they did not make the
   *aggregate* rank art above noise. **Section 17 now says why, and what is
   missing:** no *local* measure separates the two — noise wins eleven of the
   fifteen individual properties, because dense noise is abundant in local
   structure, not short of it — but a *global* redundancy statistic (the entropy
   of the centre population) separates every carpet from every noise field cleanly
   and stably. The missing ingredient is an organised-complexity term, blocked on
   #9 and #34; the new `discrimination` audit stage now reports the separation on
   every run.
2. **The SNR sample is six carpets, all Persian.** A measure that separates
   carpets need not separate paintings, and several that track their ground truth
   still have poor between-artwork SNR (positive space 0.69, local symmetries
   0.98, echoes 0.67) -- they work in the laboratory and are noisy on this field
   sample.

A correction worth recording in its own right: `echoes` was reported at rho =
+1.000 when validated on a six-point subsample of its sweep, and is +0.41 (partial
+0.17) on the full twelve points. That is exactly the small-sample coarseness
section 12 warns about -- one rank swap moves rho by 0.1 at n = 6 -- caught biting
its own author's spot check. Full sweeps only.

---

## 1. The detector reports one object as many centres

The most elementary possible stimulus, a single dark circle on a blank canvas:

| stimulus | centres detected | detected scales |
|---|---:|---|
| 1 circle, r=60px | 5 | 33–68 px |
| 1 circle, r=120px | 17 | 68 px (all) |
| 4 circles, r=60px | 20 | 33–68 px |
| 9 circles, r=40px | 33 | 24–68 px |

One visually unambiguous centre is reported as between 5 and 17.

*This section originally attributed the inflation to duplicate detections of the
same feature. That was wrong, and the correction matters because it changes the
fix. Investigation on a controlled lattice (see below) shows there is no
per-feature duplication at all.*

### The count is inflated by two mechanisms, neither of them duplication

**Figure/ground conflation.** On `jittered_lattice(0.0)` — 225 identical circles
on a lattice, with no background plateau — the detector finds 481 centres. But
every circle is detected *exactly once*: the per-circle histogram is `{1: 225}`,
with no circle missed. The other 256 detections are not near any circle. They sit
in the gaps *between* motifs, at coarser scales.

They cannot be thresholded away. Scale-normalised LoG response:

| | n | response (min / median / max) | field value (median) |
|---|---:|---|---:|
| on a motif | 225 | 0.0847 / 0.0847 / 0.0847 | 0.206 |
| in a gap | 256 | 0.1343 / 0.1343 / 0.3104 | 0.392 |

**Every gap detection responds more strongly than every motif detection.** By
response strength the gaps are *more* centre-like than the motifs. Worse,
`skimage`'s `_prune_blobs` discards the smaller blob of an overlapping pair, so
aggressive overlap pruning would delete the motifs and keep the gaps.

The cause is that the structural field is a distance transform from edges and so
**has no notion of figure versus ground**. The space between motifs is by
construction as much a local maximum as the motifs, and where motifs are small
and gaps wide it is a larger one. This is arguably correct Alexander —
interstitial space genuinely is a centre, which is what *positive space* means.
The defect is that the two populations are never distinguished, so every measure
that assumes a centre is a motif silently averages over both, and their ratio
shifts with lattice geometry, crop and scale. It also explains why `regular_grid`
scores roughness 9.6 and echoes 10.0 in §12: its "centres" are two interleaved
lattices, so nearest-neighbour statistics are meaningless.

**Plateau detections.** Separately, the cap `min(h, w) / 10` saturates the
distance transform wherever nothing is nearby, and `peak_local_max` returns many
degenerate maxima across the resulting flat region:

| image | plateau % of canvas | centres | on the plateau | at ladder ceiling |
|---|---:|---:|---:|---:|
| ardabil | 0.10% | 154 | 3 | 2 |
| bidjar | 0.87% | 122 | 6 | 0 |
| ghashghai | 0.00% | 156 | 0 | 1 |
| **pazyryk** | **27.19%** | **47** | **22** | **11** |
| sanguszko | 0.20% | 147 | 4 | 0 |
| varamin | 0.00% | 205 | 1 | 1 |

Negligible on five carpets, and **nearly half of the Pazyryk's centres** — which
bears on §11, since `images/README.md` builds its headline narrative on the
Pazyryk scoring highest on nine of fifteen properties.

Unlike gap centres, plateau detections are not centres under any definition:
there is no enclosing boundary within the cap radius. They can simply be
suppressed — but only once the cap is fixed, because a circle of radius 120
exceeds the cap of 102, so its own interior saturates too.

### The scale ceiling

Independent of both. Every one of the r=120px case's 17 blobs sits at 68 px,
which is `max_sigma * sqrt(2)` — the ceiling of the scale ladder. **The detector
cannot represent a centre larger than the ladder maximum.** The Ardabil's central
medallion, the single most important centre in that image, is exactly this case.

Everything downstream inherits all three. The hierarchy, the reinforcement graph
and all fifteen properties are computed over a centre set whose cardinality is
governed by the cap, the ladder and the figure/ground ambiguity rather than by
the structure of the artwork.

## 2b. Noise scores a higher degree of life than any artwork

*The headline finding as of the phase A/B repairs. Not introduced by them — the
same ordering held under the original energy functional, hidden by a sign
convention.*

Degree of life (higher = more life, 0 = no structure), full audit:

| | n | degree of life |
|---|---:|---:|
| **smooth_noise** | 397 | **+0.5207** |
| **white_noise** | 956 | **+0.4965** |
| **random_blobs** | 380 | **+0.4852** |
| sanguszko | 550 | +0.4054 |
| ardabil | 444 | +0.3868 |
| ghashghai | 613 | +0.3791 |
| regular_grid | 481 | +0.3747 |
| bidjar | 618 | +0.3580 |
| varamin | 624 | +0.3428 |
| pazyryk | 437 | +0.3245 |
| flat_grey | 0 | +0.0000 |

Every noise control outscores every artwork. Mechanically, **structure is not
scarce in noise, it is abundant**: 91.3% of white-noise centres are assigned a
parent against the Ardabil's 82.7%, and white noise carries 4278 graph edges
against the Ardabil's 1051. The terms reward *having* relationships and do not
distinguish *which*. Term by term the carpets win on one of five.

No re-weighting fixes it — art loses on four of the five terms, so only giving
the field term essentially all the weight would flip the order, making the score
a synonym for `gradients`. Tracked as
[#29](https://github.com/brunopostle/centres/issues/29).

## 2a. The energy functional is minimised by the absence of structure

*Added after the original audit. This is the most serious finding about the
theory rather than about an implementation of it, and it is present in the
initial commit.*

`THEORY.md` §8 states: *"Lower energy = greater wholeness."*

Take 36 centres at identical scale, spaced far apart on a large canvas. Identical
scales mean `assign_hierarchy` can never assign a parent, so there are no
parent-child pairs; spacing well beyond `3 × scale` puts every edge weight below
the 0.1 admission threshold, so the graph has no edges; and nothing overlaps:

```
equal scales, spacing 220 vs scale 10:
   E = +0.0809   graph edges = 0   parent-child pairs = 0   locality = 0.0000

random, same canvas:
   E = +1.3963   graph edges = 23   parent-child pairs = 12   locality = 0.0034
```

**A configuration with no hierarchy, no reinforcement and no interaction of any
kind scores 1.32 below a random one**, and below every real artwork (corpus range
0.953–1.737). Against the original weights the same configuration gives E = 0.000,
below everything.

Every term is a penalty for deviation, evaluated only over the objects it applies
to, so every one of them is 0 when its set is empty. **Nothing in the functional
rewards structure existing.** `E_R` is the only term that can go negative and so
the only candidate for that job, but it vanishes rather than penalising when the
graph is empty.

This is AUDIT §3's defect — absence scoring as perfection — appearing in the
energy rather than in the property normalisers. #15 fixed the property side by
returning "undefined"; the energy has no equivalent, because a sum with no terms
is genuinely zero.

`evolve()` does not collapse to this configuration, but that is a property of the
annealer rather than of the objective: its move set perturbs positions by ±2 px
and scales by `×exp(N(0, 0.02))`, far too weak to reach it. **The generative mode
works by failing to find the optimum of what it is optimising.** Tracked as
[#28](https://github.com/brunopostle/centres/issues/28).

## 2. Structural energy is centre count

Across the reference corpus and the synthetic controls, `total_energy` correlates
with the number of detected centres at **r = 0.993**, with E/n ≈ 0.27 and little
scatter. The reported "structural energy" is, to three significant figures, the
centre count times a constant.

The cause is in `energy.py`: `hierarchy_energy`, `coverage_energy` and
`alignment_energy` are sums over centres, while `reinforcement_energy` and
`locality_energy` are means. Mixing O(N) and O(1) terms makes the total scale
with N and makes energies incomparable between images. `properties.py` already
divides three of these by their pair counts — the fix was applied to the
properties and not to the energy.

## 3. Seven properties report a blank canvas as perfect

| stimulus | centres | properties scoring ≥ 9.5 / 10 |
|---|---:|---|
| flat grey | 0 | levels of scale, boundaries, positive space, local symmetries, gradients, echoes, the void |
| white noise | 0 | same seven |
| smooth noise | 0 | same seven |

When no centres are detected, the deviation-based measures return 0, and
`normalize_all` maps a raw 0 to a perfect 10 via `decay(0) = 10` and
`10 * (1 - 0) = 10` for boundaries. The tool cannot distinguish *no structure*
from *ideal structure*. The empty case must return "undefined", not 10/10.

## 4. Random circles score like masterpieces

| | levels of scale | echoes | the void | gradients | not-separateness |
|---|---:|---:|---:|---:|---:|
| random blobs | 5.3 | 5.7 | 8.1 | 8.4 | 10.0 |
| Ardabil carpet | 5.8 | 3.8 | 6.9 | 6.9 | 1.8 |

Uniformly random circles of random size and colour — no hierarchy, no repetition,
no symmetry — match the Ardabil on levels of scale and **beat it** on echoes, the
void, gradients and not-separateness. Whatever these measures rank, it is not
wholeness.

## 5. Scores change under transformations that cannot change structure

A mirror flip and a 90° rotation are exact isometries. They cannot alter an
artwork's structure, so any variation they produce is pure measurement noise.

| Ardabil | identity | mirror | rot90 |
|---|---:|---:|---:|
| roughness | **7.5** | **4.1** | **4.1** |
| levels of scale | 5.8 | 6.1 | 6.1 |
| positive space | 6.0 | 5.6 | 5.6 |

Roughness moves 3.4 points out of 10 under a mirror flip — a swing **larger than
the entire spread of roughness across all six carpets** (3.1). The cause is that
`blob_log`'s overlap pruning is order-dependent, so a different subset of the
duplicate detections survives.

Comparing between-artwork signal against within-artwork noise under
structure-preserving transforms (identity, mirror, rot90, gamma, JPEG):

| property | signal | noise | SNR | verdict |
|---|---:|---:|---:|---|
| local symmetries | 2.1 | 0.2 | 9.7 | usable |
| good shape | 2.8 | 0.3 | 8.3 | usable |
| boundaries | 2.9 | 0.4 | 7.0 | usable |
| strong centres | 2.1 | 0.3 | 6.8 | usable |
| alternating repetition | 3.9 | 0.8 | 4.7 | usable |
| contrast | 4.6 | 1.3 | 3.6 | usable |
| deep interlock | 0.5 | 0.2 | 3.1 | usable |
| positive space | 4.2 | 1.5 | 2.8 | usable |
| gradients | 2.9 | 1.3 | 2.2 | usable |
| echoes | 1.5 | 0.7 | 2.1 | usable |
| not-separateness | 8.2 | 6.6 | 1.2 | marginal |
| the void | 2.2 | 1.9 | 1.2 | marginal |
| simplicity | 0.6 | 0.6 | 1.1 | marginal |
| **levels of scale** | 0.9 | 1.0 | **0.94** | **noise** |
| **roughness** | 3.1 | 3.5 | **0.90** | **noise** |

Read this table with care: a good SNR here is necessary but nowhere near
sufficient. Deep interlock passes at 3.1 only because both its signal and its
noise are tiny — it is a constant (§7). Alternating repetition passes at 4.7 and
still fails its ground-truth sweep outright (§12). The table rules measures out;
it does not rule them in.

Under the transformations that actually occur in photographs — resizing,
cropping, vignetting, oblique viewpoint — the noise for contrast (7.4),
alternating repetition (7.0) and not-separateness (8.3) grows to roughly the
full width of the scale. The most striking single case: **changing `--max-size`
from 1024 to 512 moves the Ardabil's not-separateness from 1.8, the lowest in the
corpus, to 10.0, the highest.**

## 6. Four measures report the pipeline's tuning constants

Holding the image, the field and the centre set completely fixed (n = 154) and
varying only the strength-propagation step count:

| propagation steps | strong centres | contrast |
|---:|---:|---:|
| 5 | 1.0 | 4.8 |
| 10 (current) | 1.9 | 5.4 |
| 20 | 7.4 | 10.0 |
| 40 | 10.0 | 0.0 |

`propagate_strength` has gain `(1 - β) + α = 1.15` per step against a
row-stochastic matrix, so strengths grow as 1.15^t without limit until they hit
the clip at 10. There is no fixed point. **"Strong centres" is a readout of the
iteration counter**, and the choice of 10 steps is what sets the 0–10 range.
At 40 steps every strength saturates and contrast collapses to zero.

Similarly, varying only `num_sigma` — a sampling choice, not a property of the
artwork — drifts levels of scale from 6.5 to 5.1 and echoes from 4.5 to 3.5. The
LoG ladder from σ=2 to σ=48 in 10 steps has a rung ratio of 1.42, so parent/child
ratios are quantised to powers of 1.42 while the target ratio of 3 falls between
rungs. Levels of scale and echoes largely report where `log 3` sits relative to
the sampling lattice.

## 7. Two measures are constants

Across the whole corpus, deep interlock spans 0.4–0.8 and simplicity spans
0.9–1.5, both on a 0–10 scale. They carry almost no information about the image.
Both are artefacts of construction: deep interlock asks what fraction of graph
edges fall in the inner third of the edge-admission radius, which is fixed by the
graph constants; simplicity is the Gini coefficient of strengths, which after ten
diffusion steps is largely set by graph topology.

## 8. Normalisation constants saturate

`rise(x, ref)` clips at `ref`. With `ref = 0.02` for not-separateness, three of
six carpets pin at exactly 10.0; with `ref = 0.2` for alternating repetition,
four of six pin at 10.0. `boundaries` uses `10 * (1 - raw)` with no clamp and can
go negative. Ceiling ties are being read as agreement between artworks when they
are only evidence that the reference constant is too low.

## 9. There are not fifteen independent measures

Across the corpus the correlation between boundaries and deep interlock is 1.00;
strong centres correlates with good shape at 0.97 and with boundaries at 0.96.
**Three principal components explain 90% of the variance** across the fifteen
"independent" properties. Widening the sample to include all transformed variants
still gives an effective rank of six.

## 10. The measures are not always the property they are named for

Independent of the numerical problems, several measures are not operationalisations
of the concept in their name:

- **Good shape** is `fraction of centres that have at least one child`. It contains
  no shape information of any kind — no compactness, convexity, symmetry or
  boundary geometry. It is a hierarchy statistic wearing the name of a geometric one.
- **Alternating repetition** is the standard deviation of neighbour strengths. But
  strengths come from a *diffusion*, which is a smoothing operator and destroys
  alternation. Alternation is a periodicity property and needs a spectral or
  autocorrelation measure.
- **Contrast** measures dispersion of the same diffused strengths, which is why it
  correlates with alternating repetition at 0.87 — they are two names for one number.
- **The void** and **gradients** and **boundaries** are computed on the *reconstructed*
  field, a sum of Gaussians placed at the detected centres. They cannot see the
  image at all; they measure properties of the blob rendering.
- **Local symmetries** measures radial distance of children from parents. Symmetry
  is never computed.

The common cause is architectural. Every property is derived from one
representation — a list of `(x, y, scale, strength)` tuples from LoG blobs on a
capped distance transform. That representation has already discarded colour, tone,
orientation, shape and texture, which is precisely what several of these properties
are about. No reformulation on top of the centre set can recover information the
centre set does not contain.

## 11. Downstream consequences for the repository's claims

- `images/README.md` explains these numbers as art history. The Ardabil's low
  not-separateness is attributed to "its multi-zone composition where centres are
  only weakly linked"; that number is 1.8 at `--max-size 1024` and 10.0 at 512.
  The comparative analysis section should be withdrawn until the measures are stable.
- `THEORY.md` §8.4 and §8.5 attribute specific constants to Alexander — child
  coverage of 0.65, a radial band of 0.3–0.7 — with no citation. These appear to
  be invented and then attributed.
- `THEORY.md` §11 derives "approximately 15 levels of scale" by assuming
  `r_max/r_min ≈ 3^14` and then computing `log(3^14)/log(3) ≈ 14`. That is
  circular. The claim that wholeness corresponds to "a fixed point of a
  renormalisation operator" is not supported by anything in the implementation.
- The existing tests in `tests/` assert that each function returns what its formula
  computes, on hand-constructed centre lists. They are consistency checks, not
  validity checks: every one of them passes on the current code, and would keep
  passing under all fifteen of the failures above.

---

## 12. Ground-truth sweeps — how well each measure tracks its own property

*Current, refreshed after the #22 redefinitions. This supersedes every earlier
version of this section; the pre-redefinition numbers are in the git history.*

`audit/stimuli.py` builds, for each property, an image in which exactly one
structural quantity is varied along a known scalar while the rest is held fixed.
The parameter *is* the ground truth. Twelve to thirteen points per sweep — enough
that a single rank swap moves Spearman ρ by ~0.03 rather than 0.1, which is why
the six-point spot checks made during development were unreliable (the `echoes`
correction in §0 is the case in point).

Two of the fifteen have an **interior optimum** rather than a monotone target —
their ideal is in the middle of the sweep, so the right check is where the score
*peaks*, not a rank correlation. Both peak in the right place.

| generator → measure | test | ρ (raw / count-controlled) |
|---|---|---|
| tonal_delta → contrast | monotone | **+1.00 / +1.00** |
| interlock_depth → deep interlock | monotone | **+1.00 / +1.00** |
| dominance → strong centres | monotone | **+0.99 / +0.99** |
| zone_width → gradients | monotone | **+0.96 / +0.96** |
| ground_solidity → positive space | monotone | **+0.91 / +0.90** |
| shape_vocabulary → echoes | monotone (↓) | **−0.85 / −0.89** |
| bleed → not-separateness | monotone | **+0.78 / +0.83** |
| element_kinds → simplicity | monotone | +0.72 / +0.71 |
| border_band → boundaries | optimum(0.3) | score peaks at 0.3 ✓ |
| scale_ratio → levels of scale | optimum(3) | score peaks at 3 ✓ |
| jitter → roughness | monotone | +0.66 / +0.51 |
| bilateral_asymmetry → local symmetries | monotone (↓) | −0.72 / −0.46 |
| motif_circularity → good shape | monotone | +0.91 / **+0.39** |
| alternation → alternating repetition | monotone | +0.05 / +0.14 |
| void_size → the void | monotone | +0.80 / **−0.07** *(spurious, §16)* |

The two "(↓)" rows correctly run *negative*: more distinct shapes means less echo,
more shear means less symmetry. Each sweep now declares the direction it should
move, so the harness reports these as tracking-downward rather than failing (#33);
the count-controlled magnitude (0.89, 0.46) is what counts.

**Ten measures track in the right direction, eight of them at |ρ| ≥ 0.7 or as a
clean interior optimum.** At the start of the audit exactly one did. `echoes` and
`not-separateness` joined this group only after the #30 local-contrast detector
fix made their faint stimuli detectable; `good_shape` left the clean-keep group
this run, its raw +0.91 revealed as substantially count-driven (partial +0.39). The gain came
from two things: repairing the front end so the centre set is a stable estimate,
and — for the eleven measures that were computing the wrong quantity entirely —
redefining them against the sourced definitions on the region layer (§13, #22).

### The interior-optimum measures

`levels_of_scale` and `boundaries` are not supposed to rise monotonically. The
source gives levels of scale a *band* (magnification 2–5, ideal near 3) and
boundaries a *ratio* (band ≈ 1/3 of what it bounds), so each should be extremal in
the middle. `border_band → boundaries` illustrates it cleanly: the raw ratio it
computes tracks band thickness monotonically at ρ = ±1.000, and the normalised
score rises to 0.955 at param 0.3 — where the band is a third of what it bounds —
then falls away symmetrically. A monotone ρ on such a measure would be evidence
*against* it.

### The four that still fail, and why

- **contrast → the property was un-computable before, now it is exact.** This is
  the clearest vindication of the #22 approach: contrast was *flat* (moved 0.2%
  while tonal separation tripled) because the field is a distance transform and
  carries no tone. Given region tone it is +1.000.
- **deep interlock (+0.04)** and the harder half of **boundaries** are limited by
  the same structural fact: a thin thing (an interface, an interdigitating finger)
  produces no local maximum of a distance transform, so it never becomes a centre
  and the watershed never cuts along it. Boundaries was rescued by reading the
  bands from the image directly; deep interlock needs the interface traced from
  the image the same way, which is not yet done.
- **alternating repetition (+0.09)** needs periodicity — collapsibility to one
  repeating unit — which no per-region statistic captures.
- **echoes (+0.41), not-separateness (+0.24 partial), the void (spurious)** remain
  open; see §13 for the per-measure verdicts.

## 13. Triage of the fifteen measures, against the sourced definitions

*Current, refreshed after the #22 redefinitions.*

**ρ** is the count-controlled partial correlation from §16 (or the interior-optimum
result from §12). **SNR** is between-artwork signal against measurement noise, over
the six-carpet corpus — a necessary check that is *not* sufficient, and is a small,
single-genre sample.

| property | source-aligned formula? | ρ | SNR | verdict |
|---|---|---:|---:|---|
| contrast | redefined on region tone | +1.00 | 4.43 | **KEEP** |
| deep interlock | redefined on image convolution | +1.00 | 4.26 | **KEEP** |
| gradients | redefined on tonal rate | +0.96 | 6.24 | **KEEP** |
| strong centres | yes | +0.99 | 3.22 | **KEEP** |
| levels of scale | repaired to the 2–5 band | optimum ✓ | 3.18 | **KEEP** |
| simplicity | unchanged | +0.71 | 3.72 | **KEEP** |
| positive space | redefined on convexity, both populations | +0.90 | 1.55 | **KEEP (low SNR)** |
| boundaries | redefined on image bands | optimum ✓ | 1.83 | **KEEP (low SNR)** |
| echoes | redefined on shape similarity | −0.89 | 0.68 | **KEEP (low SNR)** |
| not-separateness | unchanged | +0.83 | 1.08 | **KEEP (low SNR)** |
| roughness | unchanged; source-defensible | +0.51 | 4.16 | **PROVISIONAL** |
| local symmetries | redefined on vertical symmetry | −0.46 | 0.45 | **PROVISIONAL** |
| good shape | redefined on compactness | +0.39 | 2.23 | **PROVISIONAL** |
| alternating repetition | unchanged | +0.14 | 2.28 | **FAILS** |
| the void | unchanged | −0.07 | 1.51 | **SPURIOUS** |

**10 keep · 3 provisional · 1 fail · 1 spurious.** At the first triage (before #22)
it was 1 keep. Eleven of fifteen diagonals now survive partialling out the centre
count. The four "KEEP (low SNR)" measures track their constructed ground truth by
partial correlation but separate the six real carpets weakly — validity on a
stimulus is necessary, and here it is outrunning field discrimination, which is
the next thing to characterise (and needs a corpus wider than six Persian carpets).

The #30 detector fix (local contrast normalisation) was the biggest mover this
run: it lifted `echoes` from |0.17| to |0.89| and `not-separateness` from +0.24 to
+0.83 by making their faint stimuli detectable. `good_shape` went the other way —
its raw +0.91 is now revealed as substantially count-driven (partial +0.39). Nothing is *retired*: every one is a real property in
Alexander's sense, and the failures are formulas or representations, not concepts.

### What changed each verdict

- **The #22 redefinitions moved five measures to KEEP** by making them compute the
  sourced quantity rather than a different one: contrast (region tone), gradients
  (tonal rate), positive space (convexity), boundaries and deep interlock (image
  band width and boundary convolution). levels of scale was repaired from a point
  target of 3 to the sourced band of 2–5, and its score peaks in the right place.
- **The #30 detector fix then moved two more to KEEP**, as a side effect. Local
  contrast normalisation made the faint shape-vocabulary and bleed stimuli
  detectable, taking echoes from a count-controlled |0.17| to |0.89| and
  not-separateness from +0.24 to +0.83. Both track their ground truth well;
  their limitation is now field SNR (0.68, 1.08), not tracking.
- **echoes correctly runs negative** — more distinct shapes, less echo. The
  harness now knows the sweep should fall and reports it as tracking rather than
  failing (#33); the count-controlled magnitude, 0.89, is the number that matters.
- **good shape weakened under count control this run:** raw +0.91, partial +0.39.
  Its region-compactness redefinition tracks the clean stimulus, but on the mixed
  ensemble much of that is centre-count covariation, so it drops to PROVISIONAL.
- **deep interlock is read from the image** — each component's perimeter against
  its convex hull, 1 for a brusque outline and growing as the boundary weaves.
  Partial +1.00, SNR 4.26, independent of good shape.
- **the void stays SPURIOUS:** its +0.80 sweep result is a centre-count artefact
  (§16), partial −0.07.

### The caveat that outranks the table

Seven measures now report the property named on them. The tool as a whole still
does not rank art above noise (§0, §2b, #29), and the SNR column is six Persian
carpets. A measure being valid on a constructed stimulus is necessary, not
sufficient; separating real artworks — and artworks from noise — is the open
problem the redefinitions did not close.

## 14. Sensitivity matrix: no measure is dominated by its own generator

Every measure against every generator parameter, |Spearman ρ|. **Dominance** is
`|ρ_own| / max |ρ_other|`: above 1 the measure responds most strongly to the
quantity it is named for; below 1 something else moves it more.

| measure | own ρ | strongest other | dominance |
|---|---:|---|---:|
| strong centres | 0.99 | 1.00 motif_circularity | 0.99 |
| contrast | 0.88 | 0.94 dominance | 0.93 |
| echoes | 0.79 | 0.95 jitter | 0.83 |
| the void | 0.80 | 0.97 dominance | 0.83 |
| local symmetries | 0.60 | 0.85 void_size | 0.70 |
| simplicity | 0.70 | 1.00 dominance | 0.70 |
| roughness | 0.63 | 0.91 alternation | 0.69 |
| good shape | 0.48 | 0.80 bleed | 0.60 |
| deep interlock | 0.51 | 0.92 zone_width | 0.55 |
| gradients | 0.52 | 0.99 dominance | 0.52 |
| boundaries | 0.45 | 0.89 void_size | 0.51 |
| positive space | 0.40 | 0.88 void_size | 0.45 |
| not-separateness | 0.41 | 1.00 dominance | 0.41 |
| levels of scale | 0.14 | 0.87 void_size | 0.16 |
| alternating repetition | 0.09 | 0.97 jitter | 0.09 |

**Fifteen of fifteen have dominance below 1.** Not one measure responds most
strongly to the quantity it is named for — including `strong_centres`, the only
measure that passed §12, which responds to `motif_circularity` (1.00) marginally
more than to its own `dominance` (0.99).

This is not sampling noise. On synthetic random scores the off-diagonal cells of
a 12-point Spearman land around 0.4–0.7; here they routinely reach 0.9–1.00.

### What the measures are actually responding to

Two generator columns dominate the table. `dominance` — which sweeps one motif's
radius from ×1.0 to ×5.4 — is the strongest driver for six measures, at ρ ≥ 0.94
for five of them. `void_size` — which clears a growing central region — is
strongest for four more.

Both make large changes to the *gross composition* of the image. What the fifteen
measures have in common is that they respond to that, and not much else. The
labels distinguish them; their behaviour does not.

### The distinction this draws, which §12 could not

§12 showed that fourteen of fifteen measures fail to track their own ground truth.
It could not say whether they were tracking *something else* or nothing at all.
The matrix answers it: they are tracking something else, strongly and in common.

Note the tension with the redundancy figure, which over 188 stimuli says 8
principal components explain 90% of the variance. The measures' *outputs* are
moderately independent; their *drivers* are not. A shared cause and distinct
outputs is what you would expect from fifteen different statistics computed over
one point process — which is what they are.

### Consequence for the triage in §13

The verdicts stand, but the KEEP is weaker than it looked. `strong_centres`
tracks its ground truth at ρ = +0.993 and still is not isolating: any measurement
of it on a real image is confounded by whatever else changes the gross
composition. It should be reported with that stated, not as a clean measure.

It also raises the bar for #22. Widening the representation to carry shape, tone
and symmetry is necessary but may not be sufficient: if every statistic over a
point process responds mainly to gross composition, adding attributes to the
points may not separate them. The cheapest next test is to regress each measure
against centre count across all 188 stimuli — several generators swing the count
by an order of magnitude (`symmetry_order` 113–1169, `ground_solidity` 281–975),
and if count explains the off-diagonals then the fifteen measures are one
measure, and the representation is not the problem.


## 15. What the source actually says, against what the code computes

Comparison against Salingaros (2025), the detailed expansion of Alexander's
fifteen properties written explicitly for programming software to detect them —
`docs/salingaros-2025-fifteen-properties.pdf`. This is the authority the project
previously lacked, and it changes several verdicts in §13.

### Three findings that overturn earlier conclusions

**1. Gap centres are not an artefact. They are half of Alexander's definition.**

> *Strong Centers:* "Centers may be of two types: either **'defined'**, with
> something in the middle to focus attention; or **'implied'**, where a complex
> engaging boundary focuses attention on its **emptier interior**."

§1 treated the 256 detections in the gaps of a lattice as a defect to be
suppressed, then softened to "arguably correct Alexander". It is not arguable: the
source names them. `Center.polarity` (#26) is not a filter for discarding gap
detections — it is the **defined/implied distinction itself**, and both populations
belong in the analysis. *Thick Boundaries* says the same thing again: "A thick
boundary also functions as an 'implied' center."

**2. Overlap is required, not penalised.**

> *Strong Centers:* "Many such mutually-reinforcing centers interconnect and
> **overlap**, rather than being isolated."
> *Deep Interlock:* "Two regions can **interpenetrate** at a semi-permeable
> interface."

`locality_energy` penalises overlap and carries the largest weight in the
functional. It encodes the opposite of what the source requires. This was
suspected on #28 from the theory alone; it is now sourced.

**3. Emptiness is explicitly not simplicity, and explicitly not good.**

> *Simplicity and Inner Calm:* "Simplicity in nature emerges from coherence and
> harmony, **not reductionism** … But an **empty, minimalist design has no
> informational content** and evokes a sense of disengagement and sterility."
> *The Void:* "**two empty regions will not reinforce each other**."

§2a found the energy functional minimised by the absence of structure. The source
makes that a contradiction of the theory, not merely of intuition.

### Sourced constants, replacing invented ones

| quantity | THEORY.md said | the source says |
|---|---|---|
| scale ratio between levels | "2–4", target **3** | "Optimal magnification factors range between approximately **2 to 5**"; 1.5 too close, 10 disengaging |
| boundary thickness | *(no measure)* | "the boundary measures roughly **1/3 of what it bounds**" |
| child-area coverage | **0.65** | *not stated anywhere* |
| child radial distance | **0.3–0.7** of parent radius | *not stated anywhere* |

Two consequences. **`levels_of_scale` targets the wrong thing in the wrong way**:
the source gives a *band* (2–5), not a point, so a quadratic penalty about log 3
is the wrong shape regardless of the constant. §12 found the measure's minimum at
ratio **2.381** — which is *inside* the sourced band. The measure may be less
wrong than it appeared; the target was wrong.

And **0.65 and 0.3–0.7 are not in the source.** They should be removed rather than
re-cited.

### What each property actually requires

| property | the source's operative content | what the code computes |
|---|---|---|
| levels of scale | magnification band 2–5, measured **independently in vertical and horizontal** | quadratic penalty about a single ratio of 3, direction-free |
| strong centres | defined **and** implied centres, nested, overlapping, mutually reinforcing | mean strength of the top quartile |
| thick boundaries | boundary ≈ 1/3 of what it bounds; boundary is itself an implied centre | field value at midpoints of graph edges |
| alternating repetition | information **not collapsible to one repeating unit** | dispersion of diffused strengths |
| positive space | space **convex**, enclosing boundary **concave**; figure/ground duality | deviation from 0.65 child-area coverage |
| good shape | **compact**, graspable, arising from nested symmetries | fraction of centres having a child |
| local symmetries | **bilateral about the vertical axis**, nested, one per scale | radial distance of children from parents |
| deep interlock | interpenetration at a semi-permeable interface | fraction of graph edges whose extents overlap |
| contrast | **black-white and colour** contrast; figure-ground symmetry of opposites | strength difference across graph edges |
| gradients | gradual change in **colour, size or texture** | mean squared gradient of the blob rendering |
| roughness | "**not** coarse-grained" — adaptation privileged over precision | coefficient of variation of nearest-neighbour distances |
| echoes | motif similarity within **and across** scales | standard deviation of log scale ratios |
| the void | complex structure **surrounds and defines** the void | gradient magnitude inside the strongest centre |
| simplicity | coherence, not reductionism; emptiness is sterile | Gini coefficient of strengths |
| not-separateness | connects to its **environment**, beyond internal coherence | Fiedler value of the graph |

Eleven of the fifteen compute something with no evident relation to the sourced
definition. That is a stronger statement than §12's, which only established that
they fail to track their own generators: it says several were never
operationalisations of the property in the first place.

Three properties are explicitly **directional or tonal** — levels of scale
(vertical and horizontal measured separately), contrast (colour), gradients
(colour and texture). The structural field is a direction-free, tone-free distance
transform, so these cannot be computed from it at all, which is the architectural
finding of §10 arriving independently from the source.


## 16. The centre count is not the common driver

§14 found every measure responding more strongly to some other generator than to
its own, with two generators — `dominance` and `void_size` — driving most of them.
Both make large changes to gross composition, and gross composition changes the
number of detected centres, which swings from 0 to 1169 across the stimuli. The
obvious hypothesis was that there is one underlying quantity, the count, and the
fifteen measures are fifteen views of it.

**That hypothesis is wrong**, and it is worth recording as a prediction that
failed. Pooled rank correlation of each measure against the count over all 181
generator stimuli:

| measure | ρ vs count | | measure | ρ vs count |
|---|---:|---|---|---:|
| gradients | +0.580 | | positive space | +0.262 |
| not-separateness | −0.426 | | levels of scale | −0.182 |
| alternating repetition | +0.379 | | simplicity | +0.167 |
| the void | +0.310 | | good shape | +0.163 |
| contrast | +0.302 | | boundaries | +0.155 |
| | | | echoes, strong centres, deep interlock, roughness, local symmetries | \|ρ\| ≤ 0.13 |

Only `gradients` exceeds 0.5. Eleven of fifteen are below 0.35. **The measures do
not track the centre count**, so the leakage in §14 has some other cause — most
likely that the generators which move gross composition genuinely move many
structural statistics at once, which is partly legitimate coupling rather than
purely a defect.

### Partialling the count out, one casualty and one reprieve

Each generator against its own target, before and after holding the count fixed:

| generator → measure | raw | partial | change |
|---|---:|---:|---:|
| dominance → strong centres | +0.99 | **+0.99** | −0.00 |
| shape_vocabulary → echoes | +0.79 | **+0.80** | +0.01 |
| element_kinds → simplicity | +0.70 | **+0.75** | +0.05 |
| border_band → boundaries | +0.45 | **+0.62** | +0.17 |
| symmetry_order → local symmetries | −0.60 | −0.58 | −0.02 |
| ground_solidity → positive space | −0.40 | −0.38 | −0.02 |
| scale_ratio → levels of scale | +0.14 | +0.15 | +0.00 |
| jitter → roughness | +0.63 | +0.44 | −0.19 |
| interlock_depth → deep interlock | −0.51 | −0.34 | −0.17 |
| bleed → not-separateness | +0.41 | +0.24 | −0.16 |
| motif_circularity → good shape | −0.48 | **+0.16** | **−0.32** |
| **void_size → the void** | **+0.80** | **−0.04** | **−0.77** |
| tonal_delta → contrast | −0.88 | −0.88 | *count constant* |
| zone_width → gradients | +0.52 | +0.52 | *count constant* |

**Eight of fifteen diagonals survive** (lose less than 0.1).

**`the_void` is the casualty, and it is a serious one.** Its ρ = +0.804 was among
the best in §12 and it is **entirely a centre-count effect**: partial the count out
and it is −0.04, which is nothing. Clearing a growing central region removes
centres, and `the_void` was reading that removal, not the void. Its §13 verdict
must change from STARVED to **SPURIOUS** — it was never tracking its ground truth
at all.

**`good_shape` gets a partial reprieve.** Its −0.48 was also a count artefact;
partialled it is +0.16 — still not tracking, but not running backwards either.

**`contrast` and `gradients` are strengthened.** Both generators hold the centre
count exactly constant by construction, so their figures were never confounded.
`contrast`'s −0.88 stands unqualified: it is flat, and the reason is that the
field carries no tone (§12, §15).

### Revised verdict for the void

| | §13 | after §16 |
|---|---|---|
| the void | STARVED (ρ +0.80, SNR 0.84) | **SPURIOUS** — ρ +0.80 is a count artefact, partial −0.04 |

That leaves **one measure with a defensible ground-truth result**, `strong_centres`
at +0.99 raw and +0.99 partial — though §14 still shows it is not isolating.


## 17. The noise problem is a *locality* problem: the missing ingredient is global redundancy

*The current attack on [#29](https://github.com/brunopostle/centres/issues/29),
the central open problem. Measured with the `discrimination` stage now in
`python -m audit`, and reproducible from `audit/redundancy.py`.*

Two routes on #29 had already been tried and had failed: re-weighting the six
energy terms (§2b — art loses on four of five, so nothing but making the score a
synonym for `gradients` flips the order) and figure/ground polarity (§15, #26 —
polarity magnitude and alternation both fail to separate art from noise). This
section reframes the problem in a way that says why both failed, and points at
what is left.

### The failure is not the energy functional. It is every local measure at once.

The reported degree of life is built from *local* relations — a parent, a graph
neighbour, an adjacent region. Measured directly, **no local measure separates a
carpet from dense noise, and it is not close.** Scoring the six carpets against
the three noise controls on all fifteen individual normalised properties (not the
energy terms — the separately-validated `properties.py` measures):

| | art mean | noise mean | who wins |
|---|---:|---:|---|
| not-separateness | 1.6 | 9.1 | noise, by 7.4 |
| alternating repetition | 6.1 | 10.0 | noise |
| the void | 5.9 | 7.8 | noise |
| levels of scale | 5.7 | 8.1 | noise |
| deep interlock | 3.9 | 5.7 | noise |
| roughness | 7.2 | 8.3 | noise |
| local symmetries | 4.7 | 6.1 | noise |
| positive space | 4.6 | 5.5 | noise |
| strong centres | 0.7 | 1.1 | noise |
| good shape | 1.9 | 3.2 | noise |
| contrast | 0.9 | 1.2 | noise |
| boundaries | 9.2 | 6.9 | *art, by 2.2* |
| gradients | 6.3 | 6.0 | art, by 0.3 |
| echoes | 1.4 | 1.3 | art, by 0.1 |
| simplicity | 2.2 | 2.1 | art, by 0.1 |

**Noise wins eleven of fifteen**, and the four it loses it loses by tenths (only
`boundaries` by a margin, and no measure separates the two populations cleanly).
This is the mechanism behind §2b stated in full generality: **local structure is
not scarce in noise, it is abundant.** A dense random field has more edges, more
parents, more neighbours and more of every local relation than a composed artwork,
so any statistic taken one relation at a time rewards it. Both prior routes stayed
local, and that is why they could not work. It also refutes the strongest
pessimistic reading — "a centre representation cannot separate structure from
noise" — because, as the next part shows, the information *is* in the centre set;
it is the local *aggregation* that discards it.

### What a carpet has that noise does not: redundancy

A carpet is built from a small vocabulary of things that recur — a few scales, a
few strengths, a few motifs, repeated across the whole field. Noise spreads across
every scale and strength there is. That is a **global** property of the centre
population's *distribution*, invisible to any per-relation measure. Measured as the
Shannon entropy of the centre population (`audit/redundancy.py`):

| discriminator | carpet range | noise range | mechanical `regular_grid` | rank separation |
|---|---|---|---:|---:|
| **strength entropy** | 1.99 – 2.38 | 2.53 – 2.74 | 1.09 | **1.000** |
| scale entropy | 1.41 – 1.60 | 1.74 – 1.91 | 1.01 | 1.000 |

On the original six carpets both separate **every** carpet from **every** noise
field — a clean rank separation of 1.000, where the reported degree of life scores
0.000 (fully inverted, noise above every carpet). Over a wider *synthetic* panel —
the six carpets, about twenty structured synthetic images, and fifteen noise fields
— the separation is 90/90 pairs for both. And it is **not a centre-count artefact**:
the Varamin (798 centres) and a white-noise field (780 centres) have almost
identical counts but sit 0.44 apart in scale entropy (1.43 against 1.88) and 0.4
apart in strength entropy (2.24 against 2.63).

**But 1.000 was measured on six near-identical rugs, and it does not survive the
wider real corpus** — see the next two subsections, which are the point.

### The load-bearing caveat: is it measuring wholeness, or repetitiveness?

This is the finding's central risk, and it is not yet closed. A Persian carpet is
*hyper-redundant* — a small vocabulary of motifs tiled across the whole field — so
"the strengths fall into few recurring values" and "the composition is alive" are
confounded on this corpus. If strength entropy is really reading repetition rather
than wholeness, it will separate ornament from noise and **fail on any composed
image that is not repetitive** — a painting, a portrait, a landscape — because
those have varied centres and therefore high strength entropy, like noise.

Measured, as far as the synthetic generators allow (real non-repetitive art cannot
be fetched in this environment — see #34), **the caveat bites**: as a composition's
redundancy falls, its strength entropy climbs toward the noise floor.

| composed stimulus | strength entropy | vs noise floor 2.53–2.74 |
|---|---:|---|
| `dominant_motif(5.4)` (one motif dominates) | 0.83 | far below |
| `element_vocabulary(6)` | 2.07 | below |
| `nested_squares(3.0)` | 1.89 | below |
| `element_vocabulary(12)` (12 distinct shapes) | **2.38** | margin ~0.2 |
| non-periodic medallion (hand-built) | **2.42** | margin ~0.15 |

The least-redundant compositions land within ~0.2 of the noise floor — *smaller
than the measure's own transform spread* (§ invariance below), so under a resize
they would cross it. The clean 90/90 separation held **because the corpus is six
highly-repetitive carpets.**

### The wider corpus (#34) confirms the caveat, on real art

The corpus was then widened to **32 real artworks** — 17 carpets and 15 tile panels
from Iran, Turkey, Syria, the Caucasus, Egypt (a reconstruction of decoration from
c. 1350 BCE), Mughal India, 19th-century Mexico and Delft, all CC-licensed (see
`images/README.md`). Re-run against the three noise controls:

| discriminator | rank separation, 32 artworks vs noise | at rest |
|---|---:|---|
| **degree of life** (the reported score) | **0.000** | every one of the 32 artworks scores *below* noise |
| strength entropy | **0.997** | `tile_geometric_egypt` (2.546) crosses the noise floor (2.542) |
| scale entropy | 1.000 | clean, but by a margin of only **0.047** — and it is transform-fragile (#9) |

Two things follow, and both were predicted by the caveat. **First, #29 is not a
six-carpet artefact:** on a fivefold larger, multi-cultural corpus the reported
degree of life *still* ranks every single artwork below noise (0.000). The problem
is real and robust. **Second, the strength-entropy separation degrades exactly
where predicted:** a bold, less-repetitive geometric piece (the ancient-Egyptian
reconstruction) crosses the noise floor *at rest*, before any transform. Combining
the two entropies recovers a clean 1.000 at rest, but only by margins of 0.01–0.03
— far inside the transform spread and inside measurement noise — and scale entropy
is still transform-fragile. **So on the wider corpus there is no robust, clean
single-statistic separator.**

The honest scope of the result is therefore narrower than "separates art from
noise": the redundancy statistics separate *dense redundant ornament* from noise
with a margin that erodes as compositional variety rises, and one real artwork
already crosses it. The corpus is still **ornament only** — carpets and tiles,
which are repetitive by construction — so the deepest form of the caveat (does it
separate a *painting* or a *portrait* from noise?) remains untested and at genuine
risk. What the wider corpus settled is that the redundancy direction is real
(0/32 for the score against ~31/32 for the discriminators is a large, consistent
signal) but is **a lead, not a solution**: no threshold on these entropies
survives both the variety already in the corpus and the transforms.

### A second axis closes it, on the corpus we have: spatial coherence

The redundancy measures ask one question — *do a few strength values recur?* There
is a second, independent question the centre set can answer: *are the strengths
arranged coherently in space — do neighbouring centres resemble each other?* That
is spatial autocorrelation, Moran's I of strength over each centre's nearest
neighbours (`audit.redundancy.spatial_coherence`). It is the complement of
redundancy: a repetitive ornament has both, dense noise has neither, and — the
point — **a non-repetitive but composed image has coherence without redundancy.**
It is exactly the axis on which a painting should differ from noise.

The two are complementary in the strong sense that on the 32-artwork corpus they
fail on **disjoint** artworks. Measured (noise: coherence 0.02–0.07, entropy
2.54–2.74):

| artwork | strength entropy | spatial coherence | separated by |
|---|---:|---:|---|
| `tile_geometric_egypt` (bold, low-redundancy) | 2.55 — *above* floor ✗ | **0.35** ✓ | coherence |
| `varamin` (spatially-flat strengths) | **2.24** ✓ | −0.01 — *below* ceiling ✗ | redundancy |
| every other artwork | ✓ | ✓ | both |

So the rule **"alive = strength entropy below the noise floor OR spatial coherence
above the noise ceiling"** separates **all 32 artworks from noise — at rest, and
under every benign and practical transform** (mirror, rot90, gamma, JPEG, tone
inversion, crop, vignette, resize-to-512): 32/32, tightest margin **+0.074** across
288 artwork-transform cases. It is an OR by design, and that is why it is robust: a
transform has to knock *both* an artwork's redundancy and its spatial coherence
below the noise cloud at once to misclassify it, and none does — where either axis
weakens (resize flattens `ghashghai`'s coherence, a bold tile has diffuse
strengths) the other holds.

This is the first **robust, clean** separator of the wider corpus, and it directly
addresses the caveat: the two least-repetitive pieces present — the pictorial Delft
tile and the Egyptian geometric — are carried by the *coherence* axis, which is the
one a painting would rely on. That was, until the corpus was widened again, only
evidence by proxy; the section below runs the painting test itself.

### The painting test, now run: coherence generalises at native resolution — but not yet under downscaling

The corpus was widened a second time (#34) with **twelve late-period Aubrey
Beardsley illustrations** — non-repetitive figurative line art, the "painting
against noise" case that every claim above was still short of. The corpus is now 44
CC-licensed real works (32 ornament, 12 figurative). Two things came out of it, one
confirming the direction and one marking its limit.

**At native resolution the direction holds, and coherence is what carries it.** All
12 Beardsley works separate from noise, and — the point of the whole exercise — **6
of the 12 have strength entropy *above* the noise floor**, so redundancy alone would
file them as noise, and every one is rescued by spatial coherence (its Moran's I
0.23–0.50 against a noise ceiling of 0.07, a 3–7× clearance). These are genuinely
non-repetitive images being caught by the axis built for them, not by repetition.
The whole-corpus effect is unambiguous: with the figurative works added,
`spatial_coherence` overtakes `strength_entropy` as the **single strongest** axis
(rank separation 0.977 vs 0.949), and the OR rule separates **44/44 at rest,
tightest margin +0.154** — adding non-repetitive art did not weaken it. The caveat
of the two sections above — "is it measuring wholeness, or repetitiveness?" — is
answered here: on art with little repetition to measure, the *coherence* half of the
rule does the work.

**But the non-repetitive works are not yet robust to downscaling, and that is the
honest limit.** The coherence-only works have no redundancy fallback, so they lean
entirely on the softer axis — and coherence erodes when the detector resolves fewer
centres. Re-run at 512 px, the four most marginal coherence-only Beardsleys behave
much worse than any ornament: `the_herald` and `et_in_arcadia_ego` fail the OR rule
**even at identity@512** (`the_herald`'s coherence collapses 0.269 → 0.048 on the
downscale alone), `les_liaisons_dangereuses` fails under crop, pad and perspective,
and only `erda` survives every practical transform (worst margin +0.038). This is
**not a new defect** — it is the same detection-count-versus-resolution dependence
recorded as the #9 negative finding (the detector resolves fewer centres at lower
resolution; the sigma ladder is not the cause), now shown to bite the figurative
works far harder than ornament precisely because they have only the one axis to
stand on. Ornament survives 512 because its redundancy holds when its coherence
flattens; a painting has no such second leg.

So the painting test **passes at native resolution and fails under aggressive
downscaling** — which locates the remaining work squarely on the resolution problem,
not on the discriminator. The OR rule is the right shape; making it hold for
non-repetitive art at all resolutions needs the detector to resolve a
resolution-stable set of centres first (#9, re-scoped to detection-count stability),
which is upstream of everything here.

Two honest limits therefore remain. The OR rule is **resolution-gated on
non-repetitive art**: robust on ornament at every resolution, robust on the
figurative works only near native resolution. And the floor and ceiling are
constants — taken from the noise cloud rather than fitted to art, which is
defensible, but the +0.154 tightest margin is at native resolution; downscaling
eats it for the coherence-only works. The `discrimination` stage prints this
separation and its margin on every audit run; adopting it into the *score* still
waits on the interior-optimum question below, on a non-constant floor/ceiling, and
now on the detection-count stability that the painting test just showed it needs.

### The third and strongest axis: nesting — "multiple things making one thing"

The two axes above measure *redundancy* (a few strength values recur) and *local
coherence* (neighbours resemble each other). Neither is the plain reading of
composition, which is that **many things are put together to make a single thing.**
The word that carries it is *single*, and a measure of it follows directly: walk the
containment hierarchy `assign_hierarchy` already builds — each centre nested under
the smallest centre whose extent contains it — and take the fraction of centres in
its **single largest tree** (`audit.redundancy.nesting`). One thing → most centres
nest under one whole; a heap of unrelated things → many small trees. It is
Alexander's actual claim, that a whole is one dominant centre all lesser centres
support, made countable.

It is the **strongest** discriminator of the three, on every count:

| axis | single-axis rank separation @1024 | survives downscaling? | catches `varamin`? |
|---|---:|---|---|
| `strength_entropy` (redundancy) | 0.949 | yes | no |
| `spatial_coherence` (local) | 0.977 | **no** — collapses on sparse paintings | no |
| **`nesting` (single whole)** | **1.000** | **yes** | **yes** |

Three properties make it the right measure. **It rank-separates every one of the 44
artworks from every noise field on its own** (noise nests at 0.01–0.06 — its
same-sized centres contain nothing — against art at 0.13–0.84). **It survives
downscaling** where local coherence fails: containment is measured in relative
*extent*, not pixel neighbourhoods, so the two paintings whose coherence collapses
at 512 (`the_herald`, `et_in_arcadia_ego`) keep a wide nesting margin under every
practical transform (worst +0.06). And **it catches `varamin`, which both other
axes miss** — a carpet with spatially-flat strengths is still one deeply-nested
thing.

Crucially it also **sidesteps the interior-optimum trap** that blocks the entropy
route (next section). A mechanical `regular_grid` is *flat* — identical cells, no
containment — so it nests at only 0.10, below almost every artwork. Where entropy
puts the grid at its "most alive" extreme, nesting puts it where it belongs:
middling, between noise and living order. So a **monotone** nesting reward is
usable directly; it needs no fitted interior optimum.

Its one weakness is *dense* images. When the detector resolves many hundreds of
centres the hierarchy fragments into many small trees, so nesting reads low even for
real art — `ghashghai` (0.06 at n=678) and `tile_panel_delft` (0.07 at n=463) are
the two artworks in the whole corpus that sit near the noise ceiling. That is the
detection-count/hierarchy-fragmentation limit (#9 re-scoped), and it is what keeps
even nesting — and the score fix built on it — from *fully* closing #29.

### Score integration: a wholeness gate closes ~99% of #29 — but reintroduces the count confound, and that is the real blocker

`nesting` is good enough to try in the reported score, not just the discrimination
stage. The cleanest integration is a **wholeness gate**: multiply the descriptive
sum by nesting, `L = (Σ participationₖ · qualityₖ) · nesting − barrier`. On paper it
preserves the #28 invariants — empty still scores 0, the result is still bounded in
[0, 1], cramming still does not pay — and it says the thing the theory always meant:
local structure counts toward life *only insofar as it forms one whole.*

Its effect on discrimination is large. Recomputed across the full corpus, the
reported score's own #29 rank separation goes from **0.131 to 0.99**, and the
mechanical grid — which the current score ranks *above* twelve artworks (+0.229) —
drops to +0.024, **below almost every artwork**, addressing the interior-optimum
problem §3 describes at the same time. A framework-faithful *additive* nesting term
reaches only 0.79–0.91 and does not fix the grid; the gate is decisively better on
these axes.

**But it was wired into `energy.degree_of_life` and run against the test suite, and
it fails a load-bearing invariant.** `test_degree_of_life_does_not_track_centre_count`
went from r = +0.14 to **r = −0.87**: the gate *reintroduces the centre-count
confound that #14 and #28 removed.* The reason is exact and not fixable by
recalibration — `nesting` is `(largest tree) / n`, and it is **not intensive**: on a
random field at fixed density the hierarchy fragments as `n` grows, so the ratio
decays with the centre count, and gating the score by it drags the whole score back
into tracking `n`. (`test_structure_scores_above_structurelessness` also fails, on
the shrunk absolute margin — the calibration issue — but the count confound is the
disqualifying one.) The change was reverted.

This sharpens the earlier reading. `nesting` is an excellent **rank discriminator**
(art nests more than noise at comparable counts, single-axis separation 1.000) but a
poor **score multiplier** (it is count-dependent, so it re-confounds the score).
Both failures have one root — `nesting`'s non-intensivity — which is also exactly why
the two *dense* artworks (`ghashghai` n=678, `tile_panel_delft` n=463) sit at the
noise ceiling: more centres, more fragmentation, lower nesting.

So the resolution of #29 is a wholeness term — the direction is settled — but the
specific measure the score needs is a **count-invariant** measure of "one thing made
of many," which raw `max_tree` is not.

### The count-invariant wholeness statistic, and the score fix it delivers

That statistic exists. The count confound in `max_tree` is precisely that a *random*
field fragments more as `n` grows, so its largest-tree fraction decays with count.
Subtract that null: define the **wholeness excess** as `max_tree` minus the expected
largest-tree fraction of a random field of the same `n`,

    excess(centres) = max_tree(centres) − C · n^B,   C ≈ 1.957, B ≈ −0.737

with `C, B` fit once over synthetic random fields at constant density, `n = 16..1024`
— fit to the **null**, not to any artwork, so it is not the frame-versus-artwork error
the invariant forbids. It is the exact correction that removes the count dependence:

- **Intensive.** On random fields the correlation with `n` falls from **−0.83
  (raw `max_tree`) to −0.04** (excess); its mean is ~0 at every `n`.
- **Still a perfect discriminator.** Corpus rank separation stays **1.000**, and —
  because a random field of `n = 678` fragments far more than `ghashghai` does — it
  **rescues the two dense works** the raw measure left at the noise floor: *no*
  artwork sits at or below the noise ceiling.

Mapped through `1 − exp(−max(0, excess)/S)` (S ≈ 0.15) to a [0, 1) quality and used
as the score's wholeness gate, `L = (Σ participationₖ·qualityₖ) · wholeness − barrier`,
it **fixes #29 in the reported score**: measured over the corpus, the score's own
art-over-noise rank separation goes from 0.131 to **1.000** (complete), and — the
test that killed the raw gate — `test_degree_of_life_does_not_track_centre_count`
**passes** (r ≈ −0.38, inside the |r| < 0.5 bound; the raw gate was −0.87). Empty and
structureless configurations still score exactly 0 (their excess is ≤ 0, so the gate
is 0). This is the first form that closes #29 in the score without reintroducing the
count confound.

**Shipped (2026-08-22), with the theory change it entails.** The gate multiplies a
flat lattice's wholeness toward 0, so a bare grid of equal-scale centres — reinforced
but not nested — scores ~0 rather than "alive." That is aligned with #29 and the
interior-optimum complaint (the grid should *not* score high), and it is a genuine
theory change, made deliberately: #28's stance moves from "any relationships beat
none" to "relationships that form one whole beat those that do not." The acceptance
test `test_structure_scores_above_structurelessness` is replaced by
`test_composed_whole_scores_above_structurelessness` (a nested hierarchy beats a
structureless scatter) and `test_flat_lattice_is_not_alive_under_the_wholeness_gate`
(a flat lattice now scores ~0). The gate introduces three constants (`C`, `B`, `S`)
and shrinks the score's scale, so the worked numbers in `energy.degree_of_life` and
THEORY §8 were re-derived on the 44-image corpus; a full re-fit of the per-term SCALE
constants on the wider corpus remains #16, orthogonal to the gate. Validated end to
end with `python -m audit`: the reported degree of life's own #29 rank separation is
**1.000 ("complete")**, and the count confound stays inside bounds (r(L, n) = −0.41
with controls). The `discrimination` stage still reports `nesting` as its strongest
raw axis (where count-invariance is not required, because the comparison is against a
same-scale noise cloud); the reported score is now the `wholeness`-gated form.

### Why this is an interior optimum, and why that blocks the fix (for the entropy route)

This section is why the *entropy* route is hard; the `nesting` route above avoids it,
which is the main reason nesting, not entropy, is the path to the score. The obvious
move — "reward low entropy" — is wrong, and the mechanical lattice
says why. `regular_grid` sits at the **low**-entropy extreme, *below* the carpets
(strength entropy 1.09 against 2.0–2.4): a perfect grid is maximally redundant.
Least entropy is the rigid lattice; most is the random field; **living order is
between them** — Alexander's organised complexity, and the same order/disorder
balance the `roughness` property already encodes as an interior optimum. So the
redundancy term the score is missing is an interior optimum, and **the location of
that optimum is an empirical quantity.** With a corpus of six near-identical
Persian carpets it can only be *fitted*, not measured — which is
[#34](https://github.com/brunopostle/centres/issues/34), and fitting a constant to
the corpus is exactly the frame-versus-artwork error the project invariant forbids.

A second obstacle is specific to the cleaner of the two measures. `scale_entropy`
separates more crisply at rest, but it rides the detector's **absolute-pixel**
scale ladder (`min_sigma=2, max_sigma=48`), and that ladder is
[#9](https://github.com/brunopostle/centres/issues/9). Under a resize the
detections shift along the ladder and clip at its ends, and the separation
**breaks**: the worst transformed carpet reaches scale entropy 1.743 against a
noise floor of 1.741. `strength_entropy` has no such dependence — its histogram is
ranged to the data, so it is invariant to a monotone rescaling of the strengths —
and it **holds under every practical transform**: mirror, rot90, gamma, JPEG, tone
inversion, crop, pad, vignette, perspective and a resize to 512 px all leave the
highest transformed carpet (2.39) below the noise floor (2.54).

### Where this leaves #29

The problem is now specific, and a **robust discriminator now exists** on the
corpus we have — three axes, and a score fix built on the strongest of them. Where
it stands:

1. **A separator that survives the wider corpus and the transforms:** *redundancy
   OR spatial coherence OR nesting.* A single entropy threshold does not survive the
   variety in the corpus. Adding the complementary axes closes it: the three fail on
   disjoint artworks, and at native resolution the OR rule separates all 44 from
   noise, tightest margin +0.155. The three single-axis separations are 0.949
   (entropy), 0.977 (coherence) and **1.000 (nesting)** — the "single whole" axis,
   which alone rank-separates every artwork from noise, survives downscaling where
   coherence collapses, and catches `varamin` that both others miss. See the nesting
   section above.
2. **#34 is done, and the painting test with it — with one caveat.** The corpus is
   now 44 CC-licensed real works (the owner added them out of band; this
   environment's egress policy denies image hosts): 32 ornament plus **12
   non-repetitive Beardsley illustrations**. At native resolution the figurative
   works separate 12/12, 6 of them on coherence alone — the decisive generalisation
   test passes. The caveat is that on those coherence-only works the separation is
   **resolution-gated**: downscaling to 512 collapses their coherence below the
   noise ceiling (`the_herald` and `et_in_arcadia_ego` fail even at identity@512),
   because a non-repetitive image has no redundancy fallback when the detector
   resolves fewer centres. That is the #9 detection-count problem, not a flaw in the
   rule.
3. **`scale_entropy` stays sidelined.** It separates cleanly at rest (margin 0.047)
   but a resize erases it, because it rides the absolute-pixel scale ladder. Making
   that ladder `edge_spacing`-relative (#9) was attempted to fix this and **reverted**
   (PLAN A2): the ladder is not the driver of resolution dependence — the detector
   simply resolves fewer centres at lower resolution — so the relative ladder did
   not deliver invariance and degraded sparse images. The OR rule does not need it:
   it runs on the transform-stable strength axis.
4. **#29 is fixed in the reported score.** A raw-`nesting` gate reintroduced the #14
   count confound (r → −0.87), because `(largest tree)/n` is not intensive. The fix is
   to subtract the random-field null: the **wholeness excess** `max_tree − C·n^B` is
   intensive (r → −0.04) and still a perfect discriminator (rank separation 1.000,
   rescuing the dense works). The reported degree of life is now gated by it,
   `L = (Σ participationₖ·qualityₖ)·wholeness − barrier`, and the audit confirms the
   score's own #29 separation is **1.000 ("complete")** with the count confound inside
   bounds (r = −0.41 with controls). It carries a deliberate theory change — a flat
   lattice now scores ~0 — and its acceptance tests were updated accordingly. A full
   re-fit of the per-term SCALE constants on the wider corpus (#16) is orthogonal and
   still open. See the count-invariant-wholeness section above.

What *is* wired in is the measurement — the `discrimination` stage prints the score's
separation, each of the three candidates' (entropy, coherence, nesting), and the
combined OR rule with its margin on every run, so #29 is no longer a paragraph in a
document but a number the harness reports, now over the full corpus. (The stage
reuses the corpus scores rather than re-analysing every corpus image a second time,
and `tests/test_redundancy.py`/`tests/test_discrimination.py` guard the axes'
complementarity, the nesting statistic, and the OR rule's survival of a resize, so a
pipeline change that broke the separation would fail a test rather than wait for an
audit.)


## Recommended order of work

1. **Keep the harness in front of every change.** `python -m audit` turns "this
   formulation is more elegant" into a claim that can be checked. THEORY.md
   currently records at least four rounds of unvalidated correction.
2. **Fix the front end**, and re-measure before touching any formula. Deduplicate
   detections per feature; make the scale ladder relative to image size and remove
   the ceiling; replace fixed Canny thresholds (50/150 on absolute gradient, which
   is why vignetting is so destructive) with locally adaptive ones; make the
   distance cap scale-relative rather than `min(h,w)/10`, which is why cropping
   changes everything; give `propagate_strength` a fixed point by normalising it.
   Several measures should become usable with no change to their definitions.
3. **Fix the reporting bugs**, which are independent of any of the above: make the
   energy terms consistently intensive; return "undefined" rather than 10/10 for
   the empty case; clamp the normalisers and raise the saturating reference constants.
4. **Re-run the sweeps in §12.** Whatever still fails to track its own ground
   truth after the front end is sound is a formula that is wrong on its own terms,
   and this is what tells the two failure modes apart.
5. **Then build the generators out** into a full sensitivity matrix — all fifteen
   measures against every generator parameter, not just the one each was built for
   — and require the diagonal to dominate. Off-diagonal leakage names the
   confounds. This shows which properties are recoverable from a centre-based
   representation at all, and which need the representation widened beyond
   `(x, y, scale, strength)` — as §10 argues good shape, local symmetries,
   alternating repetition and contrast all do.
6. **Then reappraise the formulas**, measure by measure, retiring the ones that
   cannot be made to track their own ground truth.

The honest end state is likely six to eight validated measures reported with
error bars, rather than fifteen reported to one decimal place. The error bars come
free: the invariance harness already estimates each measure's noise floor, so the
tool can print `boundaries 5.1 ± 0.7` and score each image as the median over a
set of benign transforms — which both stabilises the estimate and makes the
remaining instability visible instead of hidden.
