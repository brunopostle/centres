# Theory: Computable Wholeness — A Field of Mutually Reinforcing Centres

This document sets out the mathematical theory implemented in this codebase. The theory is a formalisation of Christopher Alexander's concept of *wholeness* from *The Nature of Order* (2002–2005).

---

> ## ⚠ Status: parts of this document are known to be false
>
> This theory was assembled largely by AI tools across several rounds of revision,
> with no test that could tell an improvement from a regression. [`AUDIT.md`](AUDIT.md)
> supplies that test, and it has falsified a number of the claims below. Corrections
> are marked **⚠ FALSIFIED** inline, with the measurement and the tracking issue.
>
> Standing rule for this document: **a claim that has been measured and found false
> is corrected here, not left standing.** Where the right replacement is not yet
> known, the claim is struck and marked open rather than quietly softened.
>
> Falsified so far:
>
> | § | Claim | Status |
> |---|---|---|
> | 8 | "Lower energy = greater wholeness" | **False.** The functional's minimum is the *absence* of structure ([#28](https://github.com/brunopostle/centres/issues/28)) |
> | 8 | The stated weights, incl. `50·E_L` | **Superseded.** Terms were not intensive; the total was centre count × 0.27 ([#14](https://github.com/brunopostle/centres/issues/14)) |
> | 8.4 | Child coverage of 0.65 | **Unsourced.** Attributed to Alexander with no citation ([#24](https://github.com/brunopostle/centres/issues/24)) |
> | 8.5 | Radial band of 0.3–0.7 | **Unsourced.** Attributed to Alexander with no citation ([#24](https://github.com/brunopostle/centres/issues/24)) |
> | 9 | "All 15 properties arise as stable patterns when E is minimised" | **Unsupported.** Never demonstrated; four of five measures tested fail to track their own ground truth |
> | 11 | 15 levels of scale "follows mathematically" | **Circular.** Assumes 3¹⁴ in order to conclude 14 |
> | 11 | Wholeness as a renormalisation fixed point | **Unsupported.** Nothing in the implementation bears on it |
>
> Sections 2, 4, 5 and 6 have been corrected in place as the implementation changed.

---

## 1. Background: Alexander's Concept of Centres

Alexander argues that living structure — in buildings, cities, art, and nature — arises from a recursive system of **centres**: coherent spatial regions that draw attention and reinforce each other. A centre is not just a point; it is a region of space with a degree of *strength* (salience, coherence) and a *scale*.

The key structural claim: a strong centre is one that is supported by smaller centres within it, and which in turn supports larger centres around it. This recursive mutual support constitutes *wholeness*.

Alexander identified 15 structural properties (levels of scale, strong centres, boundaries, alternating repetition, positive space, local symmetries, deep interlock, contrast, gradients, roughness, echoes, the void, simplicity, and not-separateness) as the observable signatures of wholeness. This theory treats them as emergent consequences of a single energy functional, not independent rules.

---

## 2. The Structural Field

**Definition.** Given a spatial domain D ⊂ ℝ², the structural field φ: D → [0, 1] assigns to each point an intensity measuring how much coherent structure exists there.

For an input image, φ is constructed by distance transform from detected edges:

```
φ(x) = min(dist(x, edges), d_max) + 0.1 · blur(x)
```

where `d_max = min(h, w) / 10` and `blur` is the Gaussian-smoothed image intensity normalised to [0, 1]. Edges are detected by Canny on a Gaussian-pre-blurred (σ=2) version of the image.

The distance transform assigns each pixel its distance to the nearest detected edge, so it peaks at the interiors of bounded regions. Two corrections are applied to the naive version:

1. **Pre-blur before edge detection (σ=2)** — without pre-blurring, densely textured images (fine carpet weave, natural scenes) produce Canny edges on 30–40% of pixels. Every interior point is then close to an edge, collapsing the field's dynamic range and driving the field maximum to the featureless image border. Pre-blurring with σ=2 suppresses fine texture while preserving major structural boundaries.

2. **Distance cap (d_max)** — large smooth background areas (museum mounts, white borders around the object of interest) would otherwise accumulate arbitrarily high distance values, drawing the field maximum — and hence the detected centres — to the image periphery rather than the carpets's interior. Capping at d_max = min(h, w)/10 prevents this while preserving the relative ordering of interior distances.

This formulation works well on any image where structure is expressed through boundaries: carpets, textiles, ornamental patterns, paintings, and natural scenes where regions are visually bounded. It also works on architectural plans, but only when drawn in **figure-ground form** (solid filled regions separated by boundaries), not as line drawings on a white ground, since Canny edge detection on a line drawing produces edges on the lines themselves rather than enclosing regions.

*Note on the previous formulation.* An earlier version used `edges + 0.3·blur`, which placed field maxima on edge pixels rather than at the interiors of regions. The distance transform corrects this.

---

## 3. Centres

**Definition.** A centre c_i is a local maximum of φ at a given scale, characterised by:

- **position** (x_i, y_i)
- **scale** r_i — the spatial extent of influence
- **strength** s_i — a measure of coherence, initialised from φ(x_i)
- **parent** p(i) — index of the nearest larger containing centre (may be null)

Centres are detected using Laplacian-of-Gaussian (LoG) blob detection across log-spaced scales from σ=2 to σ=48 pixels. The scale of a detected blob is r = σ√2. Log-spaced sigma values reflect the expected power-law distribution of centre sizes in coherent structures.

---

## 4. Hierarchy

Each centre has at most one parent. The parent of c_i is the nearest centre c_j satisfying:

1. r_j > r_i (the parent is larger)
2. dist(c_i, c_j) < 3 r_j (the child lies within 3× the parent's radius)

Condition 2 prevents a child in one corner of the image from being assigned a parent in the opposite corner. The multiplier of 3 (rather than 1) is necessary because LoG blobs at different scales detect different spatial features: the centre of a large region (parent) is typically separated from the small features within it (children) by 1–2 parent radii, not a fraction of a radius. The strict 1× condition fires almost never on real images and produces a degenerate flat hierarchy.

The result is a forest T = (C, P) where roots are the largest centres with no containing parent.

---

## 5. The Reinforcement Graph

**Definition.** The reinforcement graph G = (C, W) has centres as nodes and weighted edges expressing mutual support.

Edge weight between centres i and j:

```
W_ij = exp( -dist(i,j)² / (2 · (3 · r̄_ij)²) ) · exp( -(log r_i - log r_j)² )
```

where r̄_ij = (r_i + r_j)/2 is the mean scale of the pair.

The first factor is spatial proximity with a radius that scales with centre size (σ_spatial = 3 · r̄). This is critical: using a fixed pixel threshold (as in an earlier version with σ=50px) makes the graph too sparse for large images and too dense for small ones. Scale-relative interaction is consistent with the theory's multi-scale character.

The second factor rewards scale similarity: centres of similar size reinforce each other more strongly than centres of very different sizes.

Edges with W_ij < 0.1 are dropped.

---

## 6. Strength Propagation

Centre strengths are reinforced through the graph until they reach a stationary point:

```
s^(t+1) = s⁰ + α (Ŵ s^t),      Ŵ = W / max_i Σ_j W_ij
```

where s⁰ is the intrinsic strength each centre carries in from the structural field (§2), W is the raw edge-weight matrix of §5, and Ŵ is W scaled by its largest row sum. α = 0.2. The iteration is run to convergence rather than for a fixed number of steps.

This is Katz–Bonacich reinforcement seeded by the field. A centre's final strength is its own evidence plus a geometrically discounted sum of the strength reaching it along every walk in the graph:

```
s* = (I - αŴ)⁻¹ s⁰ = s⁰ + αŴs⁰ + α²Ŵ²s⁰ + …
```

Because ‖αŴ‖_∞ = α < 1 the update is a contraction, so this fixed point exists, is unique, and is approached geometrically at rate α — about 13 iterations for α = 0.2 to settle to 1e-9. Strengths are therefore bounded above by max(s⁰)/(1 - α) = 1.25 on the normalised field, and no clipping is required.

Centres embedded in clusters of strong, scale-similar neighbours end up stronger, because both the number of incident edges and the strength arriving along them enter the sum. Isolated centres keep their intrinsic strength exactly: with no edges their row of Ŵ is zero, so s* = s⁰. They are neither reinforced nor penalised, which is the honest reading of having no neighbours to be reinforced by.

**Why not a plain diffusion.** Earlier versions used `s ← (1 - β)s + α(W_norm s)` with α = 0.2, β = 0.05 for a fixed 10 steps. Against a row-stochastic W_norm this has gain (1 - β) + α = 1.15 per step on a uniform vector, so strengths grew as 1.15^t without bound until they saturated a clip at 10. There was no fixed point, and the properties that read absolute strengths — strong centres, contrast, alternating repetition, simplicity and E_R — were consequently readouts of the iteration counter rather than of the image: with the centre set held fixed, strong centres ran 1.0 → 1.9 → 7.4 → 10.0 for 5, 10, 20 and 40 steps, and at 40 steps every strength saturated and contrast collapsed to zero.

Setting α + (1 - β) = 1, or renormalising after each step, does produce a fixed point but the wrong one. The leading right eigenvector of a row-stochastic matrix is uniform, so any pure diffusion of that kind converges to *consensus*: every centre ends at the same strength and contrast collapses to zero again. Retaining the s⁰ source term is what keeps the stationary distribution informative, and leaving W scaled globally rather than normalised per row is what lets density matter — row normalisation erases degree, so a centre with ten strong neighbours would score the same as one with a single strong neighbour.

---

## 7. Field Reconstruction

After propagation, the wholeness field is reconstructed from the updated centres:

```
φ_reconstructed(x) = Σ_i s_i · exp( -|x - x_i|² / (2 r_i²) )
```

This is a superposition of Gaussian kernels, one per centre, weighted by strength and scaled by each centre's spatial extent. It provides a continuous representation of where coherent structure has been identified.

---

## 8. The Energy Functional

Structural energy E is a scalar measuring how far a configuration departs from Alexander's ideal.

> **⚠ FALSIFIED — "Lower energy = greater wholeness" is the wrong framing.**
>
> The functional's global minimum is the *absence* of structure. 36 centres at
> identical scale, spaced far apart, have no parent-child pairs (identical scales
> admit no parent), no graph edges (spacing beyond the weight threshold) and no
> overlap. Every term is a deviation penalty evaluated only over the objects it
> applies to, so all of them are zero:
>
> ```
> equal scales, far apart:  E = +0.081   edges = 0   pairs = 0   overlap = 0
> random, same canvas:      E = +1.396   edges = 23  pairs = 12  overlap = 0.0034
> real carpets:             E = +0.95 .. +1.74
> ```
>
> **Nothing in the functional rewards structure existing**, so emptiness scores
> better than any artwork. `evolve()` does not collapse to this only because its
> move set is too weak to reach it — the generative mode works by failing to
> optimise its own objective.
>
> The intended framing, per the repository owner on
> [#28](https://github.com/brunopostle/centres/issues/28): the empty case should
> score **zero**, structure should be **rewarded** (negative), and the reward must
> not be a sum over centres — that would simply cram in as many centres as possible,
> which contradicts *the void*.
>
> **The reported quantity becomes the *degree of life*, L = −E** — Alexander's own
> term, and the semantics required: zero for nothing, higher for more. `E` is kept
> as the quantity `evolve()` minimises, where the energy analogy is sound: a
> catenary or a minimal surface minimises an energy *subject to a constraint*, and
> without the constraint every such minimum is trivial. That is precisely the bug
> — nothing here constrains how much structure exists.
>
> The rename lands with the participation change, not before. Flipping the sign
> today would report the Ardabil at −1.4 life and a blank canvas at 0: the
> *ordering* is wrong, not merely the offset. Open.

The total is a weighted sum of six terms. The weights are derived, not chosen: each
is `PRIORITY / SCALE`, where `SCALE` is that term's standard deviation over a fixed
33-case reference ensemble, so each term contributes its intended share of the
total's variation. Locality is sized separately, as a barrier rather than a
descriptor.

> **⚠ SUPERSEDED — the previously stated weights.** This document gave
> `E = 0.3·E_H + 0.3·E_R + 0.2·E_C + 0.1·E_A + 0.1·E_φ + 50·E_L`. Three of those
> terms were sums over centres and two were means, so the total scaled with N and
> correlated with centre count at **r = 0.993** — the reported energy was the centre
> count times 0.27. The `50` on E_L had no traceable derivation; the claim that it
> existed to dominate an O(N²) reinforcement term was wrong, since E_R has been
> normalised since the initial commit. See
> [#14](https://github.com/brunopostle/centres/issues/14).

### 8.1 Hierarchy Energy E_H

Penalises deviation from a constant scale ratio between parent and child:

```
E_H = Σ_i (log(r_p(i) / r_i) - log 3)²
```

Natural hierarchies (trees, cities, traditional architecture) exhibit approximately constant scale ratios of 2–4 between successive levels. The target ratio of 3 is the midpoint of this range. Minimising E_H produces the power-law distribution of centre sizes P(r) ∝ r^{-γ} that Alexander repeatedly observed. This term encodes his property *levels of scale*.

### 8.2 Reinforcement Energy E_R

Rewards mutual strength between connected centres, normalised by the total number of possible pairs:

```
E_R = -(1 / (N(N-1)/2)) · Σ_{ij} W_ij · s_i · s_j
```

Lower (more negative) E_R means pairs of strongly connected centres both have high strength. This is the correct measure of *strong centres*: the energy is minimised by configurations where spatially clustered, scale-similar centres are all strong together.

Normalisation is essential: without it E_R scales as O(N²) and the global minimum is a degenerate cluster where all centres coincide. It is now divided by the **edge count** rather than by N(N-1)/2 — both kill the O(N²) sum, but a spatially local graph has O(N) edges, so the old denominator left the term decaying as O(1/N).

*Note on an earlier formulation.* A previous version used E_R = Σ W_ij (s_i - s_j)², which is minimised by equal but arbitrarily weak strengths — rewarding uniformity and blandness rather than coherent strength. The product formulation corrects this.

### 8.3 Locality Energy E_L

Penalises spatial overlap between centres, preventing the degenerate cluster minimum:

```
E_L = (1 / (N(N-1)/2)) · Σ_{i<j} exp( -d_ij² / (r_i + r_j)² )
```

E_L = 1 when all centres are coincident; E_L ≈ 0 when centres are well-separated relative to their scales.

Its weight is **5.54**, not the 50 previously stated, and is derived from what the barrier has to do rather than chosen. The sizing turns on a point that is easy to miss: at full collapse E_H, E_C and E_A all go to *zero*, because assigning a parent requires a strictly larger centre. A barrier sized only against the reinforcement gain leaves the collapsed configuration marginally *below* a random one. Requiring collapse to cost more than the worst configuration in the reference ensemble gives 5.54.

E_L is also the one term that is not intensive: it divides by all N(N-1)/2 pairs while only O(N) contribute, so it decays as 1/N when a whole scene is scaled up. Every intensive overlap measure tried was O(1) on real centre sets and so could not carry a barrier-sized weight without swamping the descriptive terms. Splitting the barrier from the descriptor is open, and entangled with [#28](https://github.com/brunopostle/centres/issues/28).

### 8.4 Coverage Energy E_C

Penalises deviation from ideal child-coverage of a parent region:

```
C_i = Σ_{j∈children(i)} r_j² / r_i²
E_C = Σ_i (C_i - 0.65)²
```

C_i ≈ 0.65 means children collectively occupy about 65% of the parent's area — filled without overcrowding. This is intended to encode Alexander's *positive space*: regions well-formed and occupied rather than fragmented or empty.

> **⚠ UNSOURCED.** The value 0.65 is not Alexander's and has no citation. It appears to have been invented and then attributed. Either source it or derive it from the corpus and say so. [#24](https://github.com/brunopostle/centres/issues/24)

### 8.5 Alignment Energy E_A

Penalises children whose radial distance from their parent deviates from the preferred range:

```
d_i = dist(c_i, c_{p(i)}) / r_{p(i)}
E_A = Σ_i (d_i - 0.5)²
```

The target 0.5 is the midpoint of a supposed 0.3–0.7 range.

> **⚠ UNSOURCED.** "Alexander observed that child centres tend to lie at 0.3–0.7 of the parent radius" carries no citation and appears to be invented. [#24](https://github.com/brunopostle/centres/issues/24)
 Too close to the parent's centre (d << 0.3) produces concentric but weakly differentiated structure; too far (d >> 0.7) breaks containment. This encodes *local symmetries* and *deep interlock*.

### 8.6 Field Energy E_φ

Penalises abrupt transitions in the reconstructed wholeness field:

```
E_φ = mean(|∇φ|²)
```

Minimising E_φ encourages *gradients* — smooth transitions between regions of varying strength — rather than hard discontinuities.

---

## 9. Connection to Alexander's 15 Properties

> **⚠ UNSUPPORTED.** The claim that all 15 properties arise as stable patterns when E is minimised has never been demonstrated, and nothing in the repository tests it. Five different properties are attributed to the same driver (E_R). Against it: on synthetic stimuli where the answer is known by construction, **four of the five measures tested fail to track the quantity they were built to detect** — levels of scale is uncorrelated (ρ = −0.10) with a swept parent:child scale ratio, and alternating repetition runs backwards (ρ = −0.60). See AUDIT.md §12.

The intended claim was that all 15 properties arise as stable patterns when E is minimised. Each can also be computed directly as a scalar score from the centre set, field, and reinforcement graph — see `centres/properties.py` and the `compute_all()` function. The CLI displays all 15 scores after every analysis.

| Property | Energy driver | Score (from `properties.py`) |
|---|---|---|
| Levels of scale | E_H | E_H directly; lower = better |
| Strong centres | E_R | mean strength of top-quartile centres |
| Boundaries | E_φ | field value at midpoints between connected centres; lower = clearer |
| Alternating repetition | E_R | std of neighbour strengths; higher = more alternation *(approx)* |
| Positive space | E_C | E_C directly; lower = better coverage |
| Good shape | E_C | fraction of centres with at least one child |
| Local symmetries | E_A | E_A directly; lower = better |
| Deep interlock | E_A | fraction of parent-child pairs in zone 0.3–0.7 × r_parent |
| Contrast | E_R | mean weighted strength difference between connected centres |
| Gradients | E_φ | E_φ directly; lower = smoother |
| Roughness | perturbative minima | CV of nearest-neighbour distances *(approx)* |
| Echoes | E_H | std of log-scale ratios; lower = more consistent |
| The void | E_φ | mean gradient magnitude inside strongest centre; lower = calmer |
| Simplicity | E_R | Gini coefficient of strengths; higher = more concentrated |
| Not-separateness | E_R | Fiedler value of graph Laplacian; higher = more integrated |

Two properties — alternating repetition and roughness — are marked as approximations because they require spectral or spatial-regularity analysis that is not yet fully implemented.

---

## 10. Generative Mode

`evolve()` in `pipeline.py` generates designs with high wholeness by minimising E from a random initial configuration using **simulated annealing**:

1. Start with n random centres at random positions and scales.
2. At each iteration, perturb all centre positions (Gaussian noise σ=2px) and scales (log-normal noise σ=0.02).
3. Recompute hierarchy, graph, propagate strengths, reconstruct field, evaluate E.
4. Accept the new configuration if E decreases, or with Metropolis probability exp(-ΔE/T) if it increases.
5. Temperature T decays exponentially from T_start to T_end.

The result tends toward configurations with hierarchical scaling, strong local clusters, and smooth field variation — the same structural properties observable in coherent images of any kind.

---

## 11. Scale-Invariance and the 15-Level Observation

> **⚠ CIRCULAR.** The derivation below assumes `r_max/r_min ≈ 3^14` in order to conclude that there are ≈14 levels. Substituting any other ratio yields any other number of levels, so it demonstrates nothing. Retained only as a record of what was claimed.

Alexander observed approximately 15 levels of scale in highly coherent structures. This was claimed to follow mathematically from the hierarchy energy: if the minimum-energy scale ratio is k ≈ 3, and the ratio of largest to smallest centre is r_max/r_min ≈ 3^14 ≈ 5×10^6 (roughly the ratio of a city to a brick), then:

```
L = log(r_max / r_min) / log(k) ≈ 14
```

Near the minimum of E_H, centre sizes follow a power law P(r) ∝ r^{-γ}, which is the signature of a scale-invariant (fractal) system.

> **⚠ UNSUPPORTED.** The claim that "wholeness corresponds to configurations near a fixed point of a renormalisation operator" is not supported by anything in the implementation. No renormalisation operator is defined, constructed or tested anywhere in the codebase. It is an analogy, and should be labelled as one or removed.
>
> The power-law claim is also untested, and is in tension with the measured behaviour of the detector: the LoG scale ladder runs σ = 2 → 48 in 10 log-spaced steps, a rung ratio of 1.42, so detected scale ratios are quantised to powers of 1.42 while the target ratio of 3 falls between rungs. Levels of scale and echoes partly report the sampling lattice rather than the image ([#9](https://github.com/brunopostle/centres/issues/9)).
