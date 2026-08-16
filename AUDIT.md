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
> measures do not track their ground truth are unaffected in kind** — no repair so
> far addresses what they show. §12 has since been fully re-measured against all
> fifteen properties at 12–13 sweep points each and is current; §2a, §2b and the
> banner below are also current. Sections 3–11 still carry pre-repair numbers.
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

1. **On synthetic images where the answer is known by construction, one of the
   fifteen measures tracks the quantity it is named for.** Five run backwards.
   `contrast` is not merely backwards but *flat* — figure/ground contrast more
   than triples while the measure moves 0.2%. See §12, which is the most
   important section here.
2. **Noise scores a higher degree of life than any of the six carpets** (§2b).
   Structure is not scarce in noise, it is abundant: 91% of white-noise centres
   are assigned a parent against the Ardabil's 83%.

A distinction that organises everything since: the repairs so far have improved
the instrument's **precision** — its scores are now stable under isometries,
bounded, independent of frame, resolution and iteration count — and **none has
improved its validity**. A precise instrument is a precondition for asking
validity questions, not an answer to them.

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

This is the decisive test, and the one that needs no interpretation.

`audit/stimuli.py` builds images in which exactly one structural quantity is
varied along a known scalar while the rest is held fixed. The generator parameter
*is* the ground truth.

**All fifteen properties now have a generator**, each swept over 12–13 points.
(The original five used 5–6 points, which was too coarse to be meaningful: a
single adjacent rank swap moved Spearman ρ by 0.1, so ρ ≥ 0.9 tested for
perfection and small movements carried no information.)

| generator | parameter swept | property | statistic |
|---|---|---|---:|
| `dominance` | dominant motif radius ×1.0→5.4 | strong centres | **+0.993** ✅ |
| `void_size` | cleared radius 0→0.6 | the void | +0.804 |
| `shape_vocabulary` | distinct shapes across 3 scales 1→12 | echoes | +0.790 |
| `element_kinds` | distinct elements at one scale 1→12 | simplicity | +0.699 |
| `jitter` | positional disorder σ 0→0.45 | roughness | +0.629 |
| `zone_width` | zone transition width | gradients | +0.517 |
| `border_band` | band thickness / motif radius | boundaries | +0.455 |
| `bleed` | figure/ground transition / radius | not-separateness | +0.406 |
| `alternation` | motif-size alternation 0→1 | alternating repetition | +0.091 |
| `ground_solidity` | interstitial solidity 0.6→1.0 | positive space | **−0.399** |
| `motif_circularity` | motif 4πA/P² 0.65→1.0 | good shape | **−0.483** |
| `interlock_depth` | interdigitation depth 0→0.8 | deep interlock | **−0.508** |
| `symmetry_order` | rotational order m = 1→12 | local symmetries | **−0.601** |
| `tonal_delta` | figure/ground separation 0.3→1.0 | contrast | **−0.882** |
| `scale_ratio` | parent:child ratio 1.5→6.0 | levels of scale | *(see below)* |

**One of fifteen tracks the quantity it is named for.** Five run *backwards*.

### Contrast does not measure contrast — it is flat

The raw values across the whole sweep, verified independently:

```
delta 0.30  raw 0.06694        delta 0.72  raw 0.06529
delta 0.44  raw 0.06620        delta 0.86  raw 0.06528
delta 0.58  raw 0.06525        delta 1.00  raw 0.06524
```

Figure/ground contrast more than triples; the measure moves by **0.2%**. The
ρ = −0.882 is that near-constant drifting imperceptibly downward, so "runs
backwards" understates it: the measure is *insensitive* to the quantity it is
named for.

The reason is architectural and was predicted in §10. `contrast` is the mean
weighted strength difference between connected centres; strengths come from the
structural field; the field is a distance transform from edges and **carries no
tone at all**. A tone-based property computed from a tone-free representation
cannot do anything else.

### Levels of scale is not extremal where the theory says it is

Given a recursive subdivision whose parent:child ratio is swept geometrically
about 3 — the value the whole hierarchy energy is built around:

```
ratio 1.500  raw 0.286      ratio 3.000  raw 0.476   <- the claimed ideal
ratio 2.381  raw 0.189  <- minimum      ratio 3.367  raw 0.224
ratio 2.673  raw 0.448      ratio 6.000  raw 0.580
```

The measure's minimum sits at 2.381, and **at exactly the ratio the theory calls
ideal it reports one of the worst deviations in the sweep**. Neither side is
monotone in the right direction. This is a stronger statement than the original
five-point ρ = −0.10: not merely uncorrelated, but not extremal anywhere near
where it claims to be.

### The normaliser invents optima the measures do not have

Every row also reports where the *normalised* 0–10 score peaks. Several peak in
the middle of a sweep that has no interior ideal — `border_band` at 0.2,
`interlock_depth` at 0.218, `zone_width` at 0.364, `motif_circularity` at 0.745.
Those peaks are artefacts of the reference constants in `normalize_all`, not
features of the measures.

### Properties that could not be isolated, which is itself a finding

- **Local symmetries.** A bounded motif with symmetry of order *m* necessarily
  has angular features no wider than 2π/m, so raising the symmetry order *is*
  refining the feature scale — as geometry, not as a defect of the construction.
  Any ρ here is shared with good shape and levels of scale.
- **Boundaries.** Band thickness is a length, and `edge_spacing` estimates the
  artwork's characteristic length from exactly such features. Boundary thickness
  cannot be separated from feature scale because it *is* feature scale.
- **Positive space and good shape** are two readings of one boundary, separable
  only by which side is the isolated shape. And `positive_space` is a deviation
  from 0.65 child-area coverage, which has nothing to do with convexity — the
  generator tests the property as Alexander states it while the formula tests
  something else.

### Three pipeline facts surfaced by building the stimuli

- A blank frame margin owns the deepest point of the distance transform and so
  sets the detection threshold for the whole artwork — a rosette lattice found 40
  centres from 121 motifs for that reason alone.
- Sharp corners reset `edge_spacing`: a star-polygon stimulus read 2.3 px against
  40 px for its own convex counterpart, a factor of 17 on the distance cap.
- Canny's percentile threshold is global, so in a two-zone image the stronger zone
  sets the bar the weaker must clear. At one `zone_width` point the weak zone lost
  its edges outright: 1198 centres → 12.


## 13. Triage of the fifteen measures

The verdict for each measure, on the evidence in this document. Required by
[#21](https://github.com/brunopostle/centres/issues/21), which gates the
reappraisal work behind the repository owner's agreement.

**ρ** is the ground-truth sweep (§12): does the measure track the quantity it is
named for, on a stimulus where the answer is known by construction?
**SNR** is signal against measurement noise (§5): is the spread between artworks
larger than the spread produced by transformations that cannot change an artwork?

The two ask different questions, and a measure needs both. ρ without SNR is a
measure that works in the laboratory and not in the field; SNR without ρ is a
measure that reports something stable that is not what it claims.

| property | ρ | SNR | verdict |
|---|---:|---:|---|
| strong centres | **+0.993** | 2.11 | **KEEP** |
| echoes | +0.790 | 6.09 | **PROVISIONAL** |
| simplicity | +0.699 | 3.65 | **PROVISIONAL** |
| roughness | +0.629 | 4.03 | **PROVISIONAL** |
| the void | +0.804 | **0.84** | **STARVED** |
| gradients | +0.517 | 1.05 | **REPAIR** |
| not-separateness | +0.406 | 2.67 | **REPAIR** |
| levels of scale | *not extremal at 3* | 3.69 | **REPAIR** |
| alternating repetition | +0.091 | 4.07 | **REDEFINE** |
| boundaries | +0.455 | 6.13 | **REDEFINE** |
| positive space | −0.399 | 1.87 | **REDEFINE** |
| good shape | −0.483 | 3.53 | **REDEFINE** |
| deep interlock | −0.508 | 1.80 | **REDEFINE** |
| local symmetries | −0.601 | 3.18 | **REDEFINE** |
| contrast | **−0.882** *(flat)* | 7.23 | **REDEFINE** |

**One measure is fit to report. Four are reportable with caveats. Ten are not.**

### What the verdicts mean

**KEEP** — tracks its ground truth and separates artworks by more than it separates
an artwork from its own mirror image. Only `strong_centres` qualifies.

**PROVISIONAL** — tracks in the right direction but loosely (0.6 ≤ ρ < 0.9).
Usable for comparison between images, not as an absolute statement about one.

**STARVED** — tracks its ground truth on a clean stimulus and is swamped by noise
on real photographs. This is the category the whole audit was built to identify,
because its fix is upstream: `the_void` is computed on the *reconstructed*
Gaussian field rather than on the image, so it measures the blob rendering. Point
it at the structural field and re-measure before touching the formula.

**REPAIR** — the current representation carries what the measure needs; the formula
is wrong. `gradients` shares `the_void`'s defect. `not_separateness` is a graph
property computed on a graph the representation does have. `levels_of_scale` reads
scale ratios, which the centre set carries — but the LoG ladder quantises them to
powers of 1.42 while the target ratio is 3, so it may be partly starved by #9;
re-measure after that lands. Its target of 3 is also unsourced (§11).

**REDEFINE** — the representation cannot carry what the property is about, so no
formula over the current centre set will do. Each needs #22:

| property | what it needs that `(x, y, scale, strength)` does not carry |
|---|---|
| contrast | tone — the field is a distance transform and carries none, which is why it is *flat* rather than merely wrong |
| good shape | region geometry — the formula is "fraction of centres with a child" and contains no shape information of any kind |
| local symmetries | image-domain symmetry — symmetry is never computed; the formula measures radial distance of children from parents |
| positive space | convexity of the interstitial ground — the formula is deviation from 0.65 child-area coverage, which is a different quantity, and 0.65 is unsourced |
| boundaries | boundary geometry in the image — currently read off the Gaussian rendering, and thickness cannot be separated from feature scale in any case (§12) |
| deep interlock | boundary geometry — spans 0.2 of 10 across the whole corpus, so it is close to a constant |
| alternating repetition | periodicity — the formula is dispersion of strengths that have just been through a diffusion, and diffusion destroys alternation |

**RETIRE** — none. Every one of the fifteen is a real property in Alexander's
sense. What should be retired is seven *formulas*, not seven properties.

### The practical recommendation

Report `strong_centres` outright, and the three provisional measures with their
uncertainty. **Withhold the other eleven from the output** — behind a flag, or
removed — until their verdicts are acted on. A tool that reports four measures it
can defend is more useful for assessing 2D artwork than one that reports fifteen
it cannot, and the present output invites a reader to compare numbers that do not
mean what their labels say.

### Two cautions on this table

**The SNR column is measured on six images.** It is the best available estimate of
each measure's noise floor, but six artworks is a small sample and the corpus is
all Persian carpets. A measure that separates carpets may not separate paintings.

**Redundancy is not evidence here.** The audit originally reported "3 principal
components explain 90% of the variance across 15 measures", computed over the six
corpus images. With 6 samples and 15 variables the correlation matrix has rank at
most 5, so that finding was largely an artefact of sample size. Recomputed over a
33-case ensemble the answer is 6 components for 90% and 8 for 95% — the measures
are *more* independent than the original figure suggested. The harness now
computes it over the full ensemble.


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
