# Audit: what the 15 measures actually measure

This document records an empirical audit of the measures in `centres/properties.py`.
It was written because the mathematics in [THEORY.md](THEORY.md) had accumulated
through several rounds of revision without any test that could tell an improvement
from a regression.

Everything below is reproducible with `python -m audit`.

> ### ⚠ Measured before the phase A repairs — do not quote these numbers as current
>
> Every table in this document was measured against the pipeline as it stood when
> the audit was written. Two repairs have since landed and moved the baseline:
>
> - **#10** replaced the fixed Canny thresholds with flat-field division plus
>   percentile hysteresis. Edge density is now uniform at ~8–10% across the corpus
>   where it ranged 0.9–9.7%, so **identity centre counts changed substantially** —
>   bidjar 122 → 520, sanguszko 147 → 380, ghashghai 156 → 240, varamin 205 → 136.
>   Vignette damage fell from a worst case of 7.4 points to 1.62.
> - **#12** replaced the strength diffusion with a contraction that has a fixed
>   point. §6 below **no longer applies**: `strong_centres` and `contrast` now vary
>   by ~1e-6 across `steps ∈ {5,…,100}` rather than 1.0 → 10.0. Raw strengths now
>   lie in [0, 1.25], so the normalised values in §5 and §9 have all shifted.
>
> Known changes not yet reflected: white noise now yields 397 centres rather than 0
> (and scores ≥9.5 on 1 property rather than 7); random blobs score ≥9.5 on 1
> property rather than 5; the effective rank in §9 moved from 3 to 4.
>
> **§1's diagnosis, §11's documentation findings, and the §12 result that the
> measures do not track their ground truth are unaffected in kind** — the specific
> ρ values need re-measuring, but no repair so far addresses what they show.
>
> **Since that banner was written**, the reinforcement kernel has also been
> corrected (#28): it peaked at coincidence, rewarding two centres for being the
> same centre. That moved two ground-truth sweeps substantially — scale ratio to
> levels of scale from **−0.100 to +0.800**, void size to the void from +0.500 to
> +0.700 — and took both measures previously classified as NOISE (levels of scale,
> roughness) into the usable band. The reported scalar is now the **degree of
> life**, L = −E, and §2a's structureless minimum is fixed: empty scores exactly
> zero and collapse loses by 1.26.
>
> Refreshing every table is [#18](https://github.com/brunopostle/centres/issues/18),
> which is the pivotal task in the plan: it is what distinguishes a formula that was
> starved by a bad centre set from one that is wrong on its own terms.

Two headline results:

1. **On synthetic images where the answer is known by construction, four of five
   measures fail to track the quantity they were built to detect** — one of them
   in the wrong direction. See §12, which is the most important section here.
2. **The centre detector reports a single circle as five to seventeen centres** — not through duplication, as first supposed, but through figure/ground conflation and a saturating distance cap (§1).
   Every measure is computed over a centre set that is not a stable estimate of
   anything, and they faithfully report its instability.

These point at the front end rather than at the fifteen formulas, but the audit
cannot yet fully separate the two — see §12 for what would.

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

## 12. On stimuli with known answers, the measures do not track the answer

This is the decisive test, and it is the one that requires no interpretation.

`audit/stimuli.py` builds images in which exactly one structural quantity is
varied along a known scalar while the rest is held fixed. The generator parameter
*is* the ground truth. A valid measure must be monotone in it.

| generator | varies | measure under test | Spearman ρ |
|---|---|---|---:|
| `jittered_lattice` | positional disorder, σ = 0 → 0.45 | roughness | **+0.83** |
| `contrast_field` | figure/ground tonal separation, 0.1 → 1.0 | contrast | **+0.67** |
| `void_field` | radius of a cleared central region, 0 → 0.6 | the void | **+0.50** |
| `nested_squares` | parent:child scale ratio, 1.5 → 6.0 | levels of scale | **−0.10** |
| `alternating_tiles` | motif-size alternation, 0 → 1 | alternating repetition | **−0.60** |

Not one reaches ρ = 0.9. Two are worth dwelling on:

- **Levels of scale, ρ = −0.10.** This measure exists for one purpose: to detect
  the parent-to-child scale ratio. Given a recursive subdivision in which that
  ratio is swept from 1.5 to 6.0 — spanning, and centred on, the ideal value of 3
  the theory is built around — the measure is *uncorrelated with it*. It is blind
  to the only thing it was designed to see.
- **Alternating repetition, ρ = −0.60.** Increasing alternation *decreases* the
  score. The measure runs backwards. This is consistent with §10: it is the
  dispersion of strengths that have just been through ten steps of diffusion, and
  diffusion is a smoothing operator, so the more regular the alternation, the more
  effectively it is averaged away.

The roughness sweep is the counterpoint that makes the diagnosis interesting. At
ρ = +0.83 its formula broadly works on a clean stimulus — yet on real photographs
it is the worst measure in the suite, moved further by a mirror flip than by the
difference between any two carpets. Note also that the perfect grid at σ = 0 yields
a CV of 0.146 rather than 0 (481 centres from 225 circles), and that the centre
count collapses from 481 to 148 across the sweep: the detector changes regime
partway through the experiment.

So the two failure modes are distinguishable in principle — a sound formula
starved by a bad centre set (roughness), versus a formula that is wrong on its own
terms (alternating repetition) — but these sweeps test the pipeline end to end and
cannot by themselves say which applies to each measure. Repairing the front end
first and re-running this table is what separates them, and it is the cheapest way
to find out how much of the theory is actually salvageable.

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
