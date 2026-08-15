# Repair plan

Work plan arising from [AUDIT.md](AUDIT.md). Each task is written to be picked up
by a session with no prior context: it states what to change, which files, and a
concrete acceptance check that can be run and either passes or does not.

**The ordering constraint that matters:** phase A must land before phase C, and
phase C before phase D. Do not reappraise a formula until the front end is sound —
the audit cannot currently distinguish a formula that is wrong from one that is
being fed a bad centre set, and phase C is what separates them.

Phase B is independent and can proceed in parallel with A at any time.

## How to pick up work

Open issues are the source of truth for status; this file is the overview and the
rationale. To start: list the open issues, take the lowest-numbered one whose
`Blocked by` issues are all closed, assign yourself, and close it when its
acceptance check passes.

## Dependency graph

Tracked as GitHub issues [#8–#24](https://github.com/brunopostle/centres/issues). Task IDs below link to them; statuses live on the issues, and this file is the overview.

```
A1 ─┬─ A2 ─┬────────────────┐
    └─ A6 ─┘                │
A3 ─┬──────────────────────┤
A4 ─┘                      ├─ C1 ─ C2 ─ C3 ─ D1 ─┬─ D2
A5 ────────────────────────┘                     ├─ D3
                                                 └─ D4
B1, B2, B3 ─ B4          (independent of A/C/D)
```

---

## Phase A — the front end

Nothing downstream is trustworthy until these land. Re-run `python -m audit`
after each one; record the before/after in the commit message.

### [A1](https://github.com/brunopostle/centres/issues/8) · Deduplicate blob detections per feature
**Blocks:** #9, #13, #18

`detect_centers` in `centres/pipeline.py` reports a single circle as 5–17
centres. The docstring claims a single `blob_log` call avoids duplicates; it does
not, because the default overlap pruning still admits many detections per feature
across adjacent scales.

Add a proper cross-scale non-maximum suppression: group detections whose centres
lie within some fraction of the larger blob's radius, keep the one with the
strongest LoG response, discard the rest.

**Acceptance:** `python -m audit` — a single circle on a blank canvas yields
1 centre (tolerance ±1); 9 circles yield 9 (tolerance ±2). Add these as a test in
`tests/test_pipeline.py`.

### [A2](https://github.com/brunopostle/centres/issues/9) · Make the scale ladder relative to image size and remove the ceiling
**Blocked by:** #8 · **Blocks:** #18

`min_sigma=2, max_sigma=48` is absolute in pixels, so a 120 px circle is detected
as many blobs all pinned at 68 px (`max_sigma × √2`). Large centres cannot be
represented at all, and the excess is reported as multiplicity. This also makes
every score depend on `--max-size`.

Derive the ladder from image dimensions — e.g. `max_sigma ∝ min(h, w)` — so the
same artwork at 512 px and 1024 px yields the same structure.

**Acceptance:** a circle of radius r is detected at scale ≈ r (within 25%) for
r ∈ {30, 60, 120, 200} px. Scores for a corpus image at `--max-size` 512 vs 1024
agree within 1.0 on the 0–10 scale for every property.

### [A3](https://github.com/brunopostle/centres/issues/10) · Replace fixed Canny thresholds with locally adaptive edge detection
**Blocks:** #18

`build_structural_field` in `centres/field.py` uses `cv2.Canny(…, 50, 150)` —
absolute thresholds on an 8-bit gradient. Under a vignette the corners fall below
50 and their edges vanish entirely, which is why vignetting is so destructive
(Pazyryk drops from 47 centres to 12).

Use thresholds derived from local or global image statistics, or normalise local
contrast before edge detection.

**Acceptance:** the `vignette` transform changes centre count by less than 15%,
and no property score by more than 1.5 on the 0–10 scale, for all six corpus images.

### [A4](https://github.com/brunopostle/centres/issues/11) · Make the distance cap scale-relative
**Blocks:** #18

The cap `min(h, w) / 10` in `build_structural_field` is a function of the image
frame, not of the artwork, so cropping changes the field everywhere and cropping
15% off the Ardabil moves it from 154 to 242 centres.

Tie the cap to detected structure — for example a percentile of the uncapped
distance transform — rather than to the frame.

**Acceptance:** `crop5%` and `crop15%` each change centre count by less than 15%
and no property score by more than 1.5, for all six corpus images.

### [A5](https://github.com/brunopostle/centres/issues/12) · Give strength propagation a fixed point
**Blocks:** #18

`propagate_strength` in `centres/graph.py` has gain `(1 − β) + α = 1.15` per step
against a row-stochastic matrix, so strengths grow as 1.15^t until they hit the
clip at 10. "Strong centres" is therefore a readout of the iteration counter:
1.0 → 1.9 → 7.4 → 10.0 for 5, 10, 20, 40 steps with the image held fixed.

Make the update a contraction with a stationary distribution — set `α + (1 − β) = 1`,
or renormalise after each step, or iterate to convergence instead of a fixed count.

**Acceptance:** `strong_centres` and `contrast` vary by less than 0.5 on the 0–10
scale across `steps ∈ {5, 10, 20, 40, 100}`. Add as a test.

### [A6](https://github.com/brunopostle/centres/issues/13) · Make the pipeline exactly invariant under isometries
**Blocked by:** #8 · **Blocks:** #18

A mirror flip currently moves roughness by 3.4 points out of 10 — further than the
entire spread across all six carpets — because which duplicate detections survive
pruning is order-dependent. A1 may fix this on its own; verify, and if not, make
tie-breaking and suppression order deterministic and orientation-independent.

**Acceptance:** `mirror` and `rot90` reproduce the `identity` scores for every
property to within 0.05 on the 0–10 scale. Add as a test.

---

## Phase B — reporting bugs

Independent of everything else. Each is small and self-contained.

### [B1](https://github.com/brunopostle/centres/issues/14) · Make the energy terms consistently intensive

`total_energy` in `centres/energy.py` mixes sums over centres (`hierarchy_energy`,
`coverage_energy`, `alignment_energy`) with means (`reinforcement_energy`,
`locality_energy`), so the total scales with N and correlates with centre count at
r = 0.993. Energies are not comparable between images. `properties.py` already
divides three of these by their pair counts — carry the same fix back.

Re-derive the six weights afterwards; the current ones were tuned against an
N-scaling total and will not be right once it is fixed.

**Acceptance:** `|r(structural_energy, centre_count)| < 0.5` across the corpus
plus the synthetic controls.

### [B2](https://github.com/brunopostle/centres/issues/15) · Return "undefined" for the degenerate case, not 10/10

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
