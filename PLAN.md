# Repair plan

Work plan arising from [AUDIT.md](AUDIT.md). Each task is written to be picked up
by a session with no prior context: it states what to change, which files, and a
concrete acceptance check that can be run and either passes or does not.

**The ordering constraint that matters:** phase A must land before phase C, and
phase C before phase D. Do not reappraise a formula until the front end is sound —
the audit cannot currently distinguish a formula that is wrong from one that is
being fed a bad centre set, and phase C is what separates them.

Phase B is independent and can proceed in parallel with A at any time.

## Standing rules

1. **Every change is measured against the corpus AND the synthetic generators,
   with the full `python -m audit`, not `--quick`.** Corpus-only validation has
   hidden a regression twice — #27's normalisation change was a 0.0–0.7% no-op on
   the six carpets while collapsing the lattice generators from 481 centres to 4,
   and #11's crop criterion was passed by the buggy code and unpassable by any
   correct one. The harness now always prints generator centre counts and warns
   when a stimulus collapses, and `--quick` prints a VALIDATION INCOMPLETE notice.
2. **A claim in `THEORY.md` that is measured and found false is corrected there,
   not left standing.** Much of that document was assembled by AI tools across
   several rounds with nothing able to distinguish an improvement from a
   regression. Falsified claims are marked inline and listed in a status table at
   the top. Where the right replacement is not yet known, the claim is struck and
   marked open rather than quietly softened.

## Project invariant

**Every threshold is expressed in units of the artwork's own characteristic scale
— `edge_spacing` — never in pixels, and never as a fraction of an observed
maximum.** Pixels are a property of the photograph: how much mount or wall was in
shot, and what the file was resized to. `edge_spacing` is a property of the thing
photographed.

Every frame-derived constant this pipeline has carried turned out to be a bug:
Canny's absolute 50/150 thresholds (#10), the `min(h, w) / 10` distance cap (#11),
and dividing by `field.max()` (#27, still open — the fix was reverted because the
field's scale and the detection threshold have to be corrected together). A
*global* scale estimate is correct and intended — the characteristic scale of an artwork is a global property of it, so
measures in those units move in proportion when the artwork changes. What is not
acceptable is a scale set by the frame or by a single outlier pixel.

## How to pick up work

Open issues are the source of truth for status; this file is the overview and the
rationale. To start: list the open issues, take the lowest-numbered one whose
`Blocked by` issues are all closed, assign yourself, and close it when its
acceptance check passes.

## Dependency graph

Tracked as GitHub issues [#8–#35](https://github.com/brunopostle/centres/issues). Task IDs below link to them; statuses live on the issues, and this file is the overview and the rationale.

```
DONE   A3 #10  adaptive edge detection        B1 #14  intensive energy terms
DONE   A4 #11  scale-relative distance cap    B2 #15  undefined, not 10/10
DONE   A5 #12  propagation fixed point         G #25  graph nodes by position
DONE   C1 #18  re-run sweeps / invariance     C2 #19  generators for all 15
DONE   C3 #20  sensitivity matrix             D1 #21  triage the fifteen
DONE  D2b #26  figure/ground polarity         -- #28  degree of life
DONE   D2 #22  region layer + 8 of 11 measures redefined against the source

OPEN   A1  #8  suppress plateau/structureless detections   (not started)
OPEN   A2  #9  scale ladder from edge_spacing              (agent MemoryError, unmerged)
OPEN   A6 #13  detection exactly equivariant under isometry AND inversion (merges #30)
              (isometry Δ 5.9->0.21, inversion Δ 0.18->0.043; both accepted, exactness optional)
OPEN   A7 #27  field scale + detection threshold together  (attempt reverted)
OPEN   B3 #16  re-derive the remaining reference constants (clamping done)
OPEN   B4 #17  wire audit thresholds into CI
OPEN   D3 #23  error bars / median over benign transforms
OPEN   D4 #24  correct docs   (THEORY.md done; images/README.md stale)
OPEN      #29  noise outscores every artwork   (central problem; diagnosed, blocked on #9+#34)
DONE      #30  tone inversion: detector + measures fixed (Δ 0.18->0.043, accepted); merged into #13
OPEN   D2 #22  alternating repetition still fails (needs periodicity, no region cue)
DONE      #31  boundaries/deep_interlock tone-robust: fixed 128 -> symmetrised Otsu (gamma spread 2.9->0.25, 1.7->0.46)
DONE      #32  render stimuli as contact sheets (python -m audit.render -> docs/stimuli/)
DONE      #33  sweep verdicts direction-aware (↓ measures track, not fail) + peak-vs-valley optimum
OPEN      #34  widen the corpus - field SNR + locate the #29 organised-complexity optimum
DONE      #35  removed dead interface_complexity/boundary_ratio + _describe_interfaces
```

**The instrument is now precise and ten of fifteen measures track their ground
truth** (was one). Two problems now bound progress, and they are the right place
for a new session to start:

1. **#29 — the aggregate still ranks noise above every carpet.** Individual
   validity has not composed into a valid overall degree of life. This is the
   central open problem, and it is now **diagnosed** (AUDIT.md §17, and a
   `discrimination` stage now in `python -m audit`). The failure is *locality*:
   noise wins eleven of the fifteen individual local properties, because dense
   noise is abundant in local structure, so neither re-weighting the energy terms
   nor any per-relation measure can separate art from noise. What *does* separate
   them — cleanly (every carpet below every noise field) and stably (under every
   practical transform) — is a **global** redundancy statistic: the entropy of the
   centre population's strength distribution. The missing ingredient is an
   organised-complexity term. See `audit/redundancy.py`. **A robust discriminator
   now exists on the 32-image corpus (AUDIT §17): redundancy OR spatial coherence.**
   Strength-entropy (redundancy) alone drops to 0.997 on the wider corpus — a bold
   Egyptian tile crosses at rest. Adding `spatial_coherence` (Moran's I of strength:
   do neighbouring centres resemble each other?) closes it: the two fail on
   *disjoint* artworks, and the rule "alive = entropy below noise floor OR coherence
   above noise ceiling" separates all 32 from noise at rest AND under every transform
   (tightest margin +0.074). The coherence axis is the one a non-repetitive painting
   would rely on, and the least-repetitive pieces already depend on it. Not yet in
   the score (needs the interior-optimum/grid treatment and a non-corpus-specific
   threshold; +0.074 is thin). A strong lead ready for the painting test.
2. **#34 — partly done: corpus widened to 32 CC-licensed real artworks** (17
   carpets, 15 tile panels; the owner added them out of band, since this
   environment's egress policy denies image hosts). It tempered the finding rather
   than confirming it (point 1), and the corpus is still **ornament only** — the
   decisive test, a non-repetitive painting/portrait against noise, is still not in
   it. #9 remains the second blocker (so the crisper scale-entropy version survives
   a resize).

Then: the tone-robustness bug #31 (a clean, well-specified fix), the visualisation
#32 and harness label #33 (both small), one measure that still fails (`alternating
repetition`, needs periodicity), the optional detector-exactness push #13, and the
housekeeping in #16/#17/#23/#24/#35.

### Where the instrument stands

| | before | now | target |
|---|---:|---:|---:|
| worst property Δ under vignette | 7.4 | **1.38** | ≤1.5 ✅ |
| worst property Δ under mirror / rot90 | 5.9 | **0.21** | exact optional (#13) |
| worst property Δ under tone inversion | 0.18 | **0.043** | accepted (#13) |
| measures tracking their ground truth | 1 | **10 of 15** | — |
| crop15% like-for-like, worst | +246% | **+32%** | — |
| step-count dependence of strong_centres | 1.0 → 10.0 | **1e-6** | ✅ |
| r(score, centre count) | +0.99 | **+0.13** | \|r\| < 0.5 ✅ |
| empty canvas | 7 properties at 10/10 | **15/15 undefined** | ✅ |
| structureless configuration | the global optimum | **0.000, and collapse loses by 1.3** | ✅ |

Triage of the fifteen measures, signal against measurement noise:

| | usable | marginal | NOISE |
|---|---:|---:|---:|
| original audit | 10 | 3 | 2 |
| after the kernel fix (#28) | 7 | 6 | 2 |
| after edge symmetrisation (#13) | **11** | **3** | **1** |

Merged: #10 #11 #12 #14 #15 #18 #19 #20 #21 #22 #25 #26 #28, plus the reinforcement
kernel and the eight measure redefinitions. #27 attempted and reverted; #13
partial (0.21 vs 0.05 target). Closed on the tracker to match.

**Precision, then validity.** Every phase-A/B repair improved the instrument's
*precision* — scores are now stable, bounded, and independent of frame, resolution
and iteration count. The #22 redefinitions then improved *validity*: eight of
fifteen measures now track their ground truth, where one did — ten after the #30
detector fix lifted echoes and not-separateness. **What has not moved is the
aggregate**: the degree of life still ranks noise above every carpet
(#29). Valid individual measures have not yet composed into a valid overall score,
and that is the whole remaining problem.

A methodological caveat on the sweeps, measured: with only five sample points a
single adjacent rank swap moves Spearman ρ by 0.1, so ρ ≥ 0.9 tests for
perfection and small ρ changes carry no information. #19 is widening the sweeps
to 10–12 points and giving interior-optimum measures (levels of scale, roughness)
a test that is not monotonicity.


### Changes since the plan was first written

The audit's own diagnosis of the front end was wrong in one place, and two tasks
turned out to be misattributed. Recorded here so the history is legible:

- **#8 was rewritten.** It asked for deduplication of blob detections. There is no
  duplication — on a lattice of 225 identical circles the detector finds each
  exactly once. The count inflation comes from figure/ground conflation (split out
  as **#26**) and from plateau detections, which is all #8 now covers. It is
  consequently **blocked by #11**, since the cap is what creates the plateau.
- **#13 was re-pointed** from #8 to #10. The isometry failure is not detection
  ordering: `cv2.Canny` is not mirror-equivariant (2907 edge pixels differ, 0.5%),
  and the distance transform amplifies that to 8.6% of the field. Greyscale
  conversion and Gaussian blur are exact.
- **#9 was unblocked** — it was waiting on the deduplication premise.
- **#27 is new and constrains most of phase A.** `field / field.max()` against an
  absolute `blob_log` threshold makes every local change act globally. This turned
  out to be the actual mechanism by which vignetting destroyed the results.
- **#25 is new** — `build_graph` adds nodes by `c.id` but edges by list index, so
  any task that filters a centre list (#8, #26) will silently corrupt the graph
  until it is fixed.

---

## Phase A — the front end

Nothing downstream is trustworthy until these land. Re-run `python -m audit`
after each one; record the before/after in the commit message.

### [A1](https://github.com/brunopostle/centres/issues/8) · Suppress detections in structureless saturated regions
**Blocks:** #18 — blockers #11 and #25 are both done, so this is ready

*Rewritten. This task originally asked for deduplication; there is none — see the
note above and §1 of AUDIT.md.*

The cap saturates the distance transform wherever nothing is nearby, and
`peak_local_max` returns many degenerate maxima across the resulting flat plateau.
None are centres under any definition: there is no enclosing boundary within the
cap radius. Measured against the old `min(h, w) / 10` cap: negligible on five
carpets (0–0.9% of canvas) and **27% of the Pazyryk, where 22 of its 47 centres
sat on the plateau**. Re-measure against the current `8 × edge_spacing` cap before
starting — #11 will have reduced it, and by how much is unknown.

**Acceptance:** a single circle on a blank canvas yields exactly 1 centre; 9
circles yield 9 (±2); the Pazyryk's on-plateau count drops to ~0 with no carpet
losing centres that sit on genuine structure. Add as a test in
`tests/test_pipeline.py`. Re-check bidjar/roughness under vignette, the one cell
#10 left at 1.62 against a threshold of 1.5.

### [A2](https://github.com/brunopostle/centres/issues/9) · Make the scale ladder relative to edge spacing and remove the ceiling
**Blocks:** #18 — no longer blocked by #8, see above

`min_sigma=2, max_sigma=48` is absolute in pixels, so a 120 px circle is detected
as many blobs all pinned at 68 px (`max_sigma × √2`). Large centres cannot be
represented at all, and the excess is reported as multiplicity. This also makes
every score depend on `--max-size`.

**Re-specified.** The original text said to derive the ladder from image
dimensions — the same frame-versus-artwork error #11 fixed for the cap. Derive it
from `edge_spacing` instead, per the invariant above. That makes the ladder
framing-independent as well as resolution-independent, since resizing changes
`edge_spacing` in exact proportion.

Also fix the rung ratio while here: σ = 2 → 48 in 10 log-spaced steps gives a
ratio of 1.42, while `hierarchy_energy` targets a parent:child ratio of 3, which
falls *between* rungs — so `levels_of_scale` and `echoes` partly report the
sampling lattice. Make the lattice a function of the target ratio rather than
hard-coding either; whether 3 is right at all is a #21 question.

**Acceptance:** a circle of radius r is detected at scale ≈ r (within 25%) for
r ∈ {30, 60, 120, 200} px; scores for a corpus image at `--max-size` 512 vs 1024
agree within 1.0 on the 0–10 scale for every property; and — new — scores for an
image and the same image padded with a 10% border agree within 1.0, which the
original spec would have failed.

### [A3](https://github.com/brunopostle/centres/issues/10) · ✅ Replace fixed Canny thresholds with locally adaptive edge detection
**Blocks:** #18

`build_structural_field` in `centres/field.py` uses `cv2.Canny(…, 50, 150)` —
absolute thresholds on an 8-bit gradient. Under a vignette the corners fall below
50 and their edges vanish entirely, which is why vignetting is so destructive
(Pazyryk drops from 47 centres to 12).

Use thresholds derived from local or global image statistics, or normalise local
contrast before edge detection.

**Acceptance:** the `vignette` transform changes centre count by less than 15%,
and no property score by more than 1.5 on the 0–10 scale, for all six corpus images.

### [A4](https://github.com/brunopostle/centres/issues/11) · ✅ Make the distance cap scale-relative
**Blocks:** #18

The cap `min(h, w) / 10` in `build_structural_field` is a function of the image
frame, not of the artwork, so cropping changes the field everywhere and cropping
15% off the Ardabil moves it from 154 to 242 centres.

Tie the cap to detected structure — for example a percentile of the uncapped
distance transform — rather than to the frame.

*Resolved with `8 × edge_spacing`, the `d^-1.5`-weighted geometric mean of the
distance transform along its medial axis. **The acceptance criterion above was
badly written and unachievable by any implementation:** `crop15%` trims each side,
retaining only 59% of the pixels, so any content-tracking count must move far more
than 15% — and a perfectly constant cap still gives −32% to −57%. The old code
passed on three images only because cropping the border brightened the field and
the extra detections cancelled the lost area, which was the bug itself.*

**Acceptance, restated and met:** like-for-like — centres in the cropped image
versus centres the uncropped run places inside the same window — worst case
+246% → +32%, median +64% → +15%.

### [A5](https://github.com/brunopostle/centres/issues/12) · ✅ Give strength propagation a fixed point
**Blocks:** #18

`propagate_strength` in `centres/graph.py` has gain `(1 − β) + α = 1.15` per step
against a row-stochastic matrix, so strengths grow as 1.15^t until they hit the
clip at 10. "Strong centres" is therefore a readout of the iteration counter:
1.0 → 1.9 → 7.4 → 10.0 for 5, 10, 20, 40 steps with the image held fixed.

*Resolved with Katz/Bonacich reinforcement, `s ← s₀ + α(Ŵs)` with `Ŵ = W / max row
sum`. Note for the record that the two fixes originally suggested here — setting
`α + (1 − β) = 1`, or renormalising each step — are **both wrong**: the leading
right eigenvector of a row-stochastic matrix is uniform, so both converge to
consensus and contrast collapses to zero. They would have passed the acceptance
criterion while destroying the measure. Keeping the `s₀` source term is what makes
the fixed point informative; scaling `W` globally rather than per row is what lets
density matter.*

**Acceptance:** met — spread 1e-6 for `strong_centres` and 1.2e-5 for `contrast`
across `steps ∈ {5, 10, 20, 40, 100}`, against a threshold of 0.5.

### [A6](https://github.com/brunopostle/centres/issues/13) · Make the pipeline exactly invariant under isometries
**Blocked by:** #10 (done) · **Blocks:** #18

A mirror flip currently moves roughness by 3.4 points out of 10 — further than the
entire spread across all six carpets.

*The cause was originally attributed to order-dependent overlap pruning. It is
not.* Measured per stage on the Ardabil: greyscale conversion and Gaussian blur
are **exactly** mirror-equivariant; `cv2.Canny` is **not** (2907 edge pixels
differ, 0.536%), almost certainly through its hysteresis edge-tracking, whose
traversal order decides which weak edges get promoted; the distance transform then
amplifies that into 8.6% of the field. 24 of 154 centres fail to survive a mirror
flip, and only one of them is on a plateau.

#10 has replaced the thresholds but retains OpenCV's hysteresis, so this is
probably still open — verify first. If so, either symmetrise explicitly (compute
the edge map over the dihedral group and combine) or use an edge operator with no
order-dependent tracking stage.

**Acceptance:** `mirror` and `rot90` reproduce the `identity` scores for every
property to within 0.05 on the 0–10 scale. Add as a test.

### [A7](https://github.com/brunopostle/centres/issues/27) · Fix the field's scale and the detection threshold together, or not at all
**Blocks:** #8

`build_structural_field` ends with `field / (field.max() + 1e-8)`, and
`detect_centers` then applies an absolute `threshold=0.08`. Those do not compose:
`field.max()` is one scalar set by whichever pixel is furthest from an edge, so
**any change anywhere rescales the whole field** and the fixed threshold then
admits or rejects centres everywhere.

This is the actual mechanism by which vignetting did its damage, discovered during
#10: when corners lose edges the distance transform there runs to the cap,
`field.max()` jumps (varamin 32 → 69), and every centre count moves. The three
corpus images that failed worst were exactly the three whose `dist.max()` moved.

*Attempted and reverted.* Dividing by the cap rather than by `field.max()` is a
0.0–0.7% no-op on the corpus, where the cap always bites, and catastrophic where
it does not: on a sparse lattice `8 × spacing` is 270 px against a largest actual
distance of 74, so the field peaks at 0.27 and the absolute 0.08 threshold rejects
nearly everything. Generator counts collapsed from 481 to 4. **The instrument the
whole validation approach depends on was broken by a change that looked clean on
the corpus alone.**

The scope is therefore both halves at once: the field's scale and the detection
threshold have to be expressed in the same quantity, and `CAP_SPACINGS` re-derived
against the generators as well as the corpus.

**Acceptance:** the corpus *and* every generator in `audit/stimuli.py` keep
comparable centre counts; `field.max()` no longer divides a field read by an
absolute threshold. Run the full `python -m audit`, not `--quick`.

### [G](https://github.com/brunopostle/centres/issues/25) · ✅ Fix build_graph node/edge key mismatch
**Blocks:** #8, #26

`build_graph` adds nodes keyed by `c.id` but edges keyed by list index. Two
centres with ids 5 and 7 produce **four** nodes, the only edge joining two
phantoms, and both real centres isolated. Harmless today because every production
path assigns `id=i`, but a direct trap for any task that filters a centre list.

**Acceptance:** `build_graph` on a list with arbitrary unique ids produces exactly
`len(centers)` nodes, all carrying a `center` attribute. Add as a test.

---

## Phase B — reporting bugs

Independent of everything else. Each is small and self-contained.

### [B1](https://github.com/brunopostle/centres/issues/14) · ✅ Make the energy terms consistently intensive

`total_energy` in `centres/energy.py` mixes sums over centres (`hierarchy_energy`,
`coverage_energy`, `alignment_energy`) with means (`reinforcement_energy`,
`locality_energy`), so the total scales with N and correlates with centre count at
r = 0.993. Energies are not comparable between images. `properties.py` already
divides three of these by their pair counts — carry the same fix back.

Re-derive the six weights afterwards; the current ones were tuned against an
N-scaling total and will not be right once it is fixed.

**Acceptance:** `|r(structural_energy, centre_count)| < 0.5` across the corpus
plus the synthetic controls.

### [B2](https://github.com/brunopostle/centres/issues/15) · ✅ Return "undefined" for the degenerate case, not 10/10

A featureless grey canvas detects zero centres, the deviation-based measures
return 0, and `normalize_all` maps 0 to a perfect 10 — for levels of scale,
boundaries, positive space, local symmetries, gradients, echoes and the void.
The tool cannot distinguish *no structure* from *ideal structure*.

Have `compute_all` return `None` where a property is undefined, and have the CLI,
GUI and JSON output render that as `—` rather than a number.

**Acceptance:** all three null controls report no numeric score for any property
whose inputs are empty. Add as a test.

### [B3](https://github.com/brunopostle/centres/issues/16) · Clamp the normalisers and re-derive the reference constants

`rise(x, 0.02)` for not-separateness pins three of six carpets at exactly 10.0;
`rise(x, 0.2)` for alternating repetition pins four of six. `boundaries` uses
`10 × (1 − raw)` with no clamp and can go negative. Ceiling ties are currently
read as agreement between artworks when they only show the reference is too low.

Clamp every output to [0, 10] and set each reference from a percentile of the
observed distribution over a corpus, documenting the corpus used.

**Acceptance:** no property has more than one corpus image at exactly 0.0 or 10.0.

### [B4](https://github.com/brunopostle/centres/issues/17) · Wire the audit thresholds into the test suite
**Blocked by:** #14, #15, #16

The 87 existing tests all pass on the current code and would keep passing under
every failure in AUDIT.md — they assert that each formula computes itself.

Add a `tests/test_invariance.py` that runs the harness assertions as real tests:
isometry invariance, null controls, hyperparameter independence. Mark the slow
ones so they can be excluded locally but run in CI.

**Acceptance:** deliberately reverting any one of A5, B1 or B2 makes the suite fail.

---

## Phase C — re-measure

### [C1](https://github.com/brunopostle/centres/issues/18) · Re-run the sweeps and the invariance suite
**Blocked by:** #8–#13

Run `python -m audit` on the repaired front end and update AUDIT.md with the
before/after. **This is the step that tells the two failure modes apart:** any
measure that still fails to track its own ground truth once the centre set is
sound is a formula that is wrong on its own terms, not one that was starved.

**Acceptance:** AUDIT.md carries both columns, and each of the 15 is labelled
*starved* (now tracks) or *wrong* (still does not).

### [C2](https://github.com/brunopostle/centres/issues/19) · Extend the generators to all fifteen properties
**Blocked by:** #18

`audit/stimuli.py` currently has five parametric generators. Add one per
remaining property, each isolating a single quantity with ground truth known by
construction. Sketches: strong centres — dominance ratio of one motif over a field
of equals; boundaries — border-band thickness; positive space — convexity of the
interstitial ground; good shape — motif regularity interpolated from circle to
random star polygon; local symmetries — motif symmetry group order; deep interlock
— interdigitation depth between two regions; echoes — number of distinct motif
shapes reused across scales; simplicity — number of distinct elements;
not-separateness — figure/ground boundary sharpness.

**Acceptance:** every property has a generator, and `python -m audit` prints a ρ
for all fifteen.

### [C3](https://github.com/brunopostle/centres/issues/20) · Build the sensitivity matrix
**Blocked by:** #19

Run every measure against every generator parameter, not just the one it was built
for. The diagonal should dominate; off-diagonal leakage names the confounds, and
the current effective rank of 3–6 predicts there will be a lot of it.

**Acceptance:** a 15 × 15 matrix in AUDIT.md, with the diagonal-dominance ratio
reported per measure.

---

## Phase D — reappraisal

### [D1](https://github.com/brunopostle/centres/issues/21) · Triage all fifteen measures
**Blocked by:** #20

For each: **keep** (tracks its ground truth, survives transforms, adds independent
information), **repair** (right concept, wrong formula), **redefine** (needs a
representation the centre set does not carry), or **retire**.

Expect to keep six to eight. Fewer defensible measures is a better tool than
fifteen indefensible ones.

**Acceptance:** a table in AUDIT.md with a verdict and one-line justification per
measure, and agreement from the repository owner before D2–D4 proceed.

### [D2](https://github.com/brunopostle/centres/issues/22) · Widen the representation for the properties that need it
**Blocked by:** #21

Every property is currently derived from `(x, y, scale, strength)` tuples, which
have already discarded colour, tone, orientation, shape and texture — precisely
what several of the properties are about. Good shape needs region geometry;
local symmetries needs image-domain symmetry detection; alternating repetition
needs periodicity (autocorrelation or spectral); contrast is tonal.

Add region segmentation with shape descriptors alongside the centre set, rather
than replacing it.

### [D2b](https://github.com/brunopostle/centres/issues/26) · Give centres figure/ground polarity
**Blocked by:** #21 · groups with #22

The field is a distance transform from edges and has **no notion of figure versus
ground**. On a lattice of 225 identical circles the detector finds each motif
exactly once — and 256 further centres in the *gaps between* them, every one of
which responds more strongly to the LoG (0.134–0.310) than every motif does
(0.0847). No threshold separates them, and `skimage`'s overlap pruning discards the
*smaller* blob, so it would delete the motifs and keep the empty space.

This is not simply a bug: interstitial space genuinely is a centre, which is what
*positive space* means. The defect is that the populations are never distinguished,
so every measure assuming "centre = motif" averages over both, and their ratio
shifts with motif-to-gap ratio, crop and scale.

**Acceptance:** on `jittered_lattice(0.0)` the 225 motif and 256 gap detections are
labelled with ≥95% accuracy; each property declares which population it consumes.

### [D3](https://github.com/brunopostle/centres/issues/23) · Report error bars, and score as a median over benign transforms
**Blocked by:** #21

The invariance harness already estimates each measure's noise floor, so the tool
can print `boundaries 5.1 ± 0.7`. Scoring each image as the median across a set of
structure-preserving transforms both stabilises the estimate and yields the error
bar. This makes residual instability visible instead of hidden, and is worth doing
even if it is the only thing that ever ships.

### [D4](https://github.com/brunopostle/centres/issues/24) · Correct the documentation
**Blocked by:** #21

- `images/README.md`: withdraw the comparative analysis. It reads artefacts as art
  history — the Ardabil's low not-separateness is explained as "its multi-zone
  composition", but that number is 1.8 at `--max-size 1024` and 10.0 at 512.
- `THEORY.md` §8.4, §8.5: remove or cite the constants attributed to Alexander —
  child coverage of 0.65 and a radial band of 0.3–0.7 appear to be invented and
  then attributed.
- `THEORY.md` §11: remove the derivation of "approximately 15 levels of scale",
  which assumes `r_max/r_min ≈ 3^14` in order to conclude 14, and the claim about
  a renormalisation fixed point, which nothing in the implementation supports.
- `THEORY.md` §9: the claim that all 15 properties "arise as stable patterns when
  E is minimised" is not demonstrated anywhere. Either demonstrate it or drop it.
