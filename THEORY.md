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
> | 8 | "Lower energy = greater wholeness" | **Was false, now corrected.** The functional's minimum was the *absence* of structure. Replaced by the degree of life L = −E, participation × quality, zero for nothing ([#28](https://github.com/brunopostle/centres/issues/28)) |
> | 8 | The reported score orders artwork above noise | **False, and now diagnosed.** Noise outscores all six carpets, equally under the functional this replaced. AUDIT.md §17: this is a *locality* failure — noise wins eleven of the fifteen individual local properties, because dense noise is abundant in local structure, not short of it, so no per-relation measure and no re-weighting can separate the two. The reported score still fails #29 for all 32 artworks of the widened corpus, so the problem is robust. What separates art from noise is *global* structure the local aggregate discards: **redundancy OR spatial coherence.** Strength-entropy (redundancy) alone drops to 0.997 on the wider corpus, but adding Moran's I of strength (spatial coherence — "do neighbouring centres resemble each other?") closes it: the two fail on disjoint artworks, and the OR rule separates all 32 from noise at rest and under every transform (tightest margin +0.074). Not yet in the score (needs the interior-optimum treatment and a non-corpus-specific threshold; corpus still ornament-only, so the painting test is #34). A strong lead ready for that test ([#29](https://github.com/brunopostle/centres/issues/29)) |
> | 8 | The stated weights, incl. `50·E_L` | **Superseded twice.** Terms were not intensive; the total was centre count × 0.27 ([#14](https://github.com/brunopostle/centres/issues/14)). The weights they became belonged to the additive framing and are gone with it ([#28](https://github.com/brunopostle/centres/issues/28)) |
> | 8.5 | Child coverage of 0.65 | **Unsourced.** Attributed to Alexander with no citation ([#24](https://github.com/brunopostle/centres/issues/24)) |
> | 8.6 | Radial band of 0.3–0.7 | **Unsourced.** Attributed to Alexander with no citation ([#24](https://github.com/brunopostle/centres/issues/24)) |
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
W_ij = exp( -log(d_ij / (r_i + r_j))² / (2 · 0.5²) ) · exp( -(log r_i - log r_j)² )
```

The first factor is a Gaussian in *log* separation, measured in units of the pair's own combined radius, so it **peaks where the two centres touch** and falls to zero both as they coincide and as they separate. Working in log separation rather than in pixels does two things: it sends the weight to zero as d → 0, and it makes the kernel depend only on separation relative to the pair's own size, per the scale invariant of §2.

The second factor rewards scale similarity: centres of similar size reinforce each other more strongly than centres of very different sizes.

*Note on the previous formulation, and why it mattered.* The weight was `exp(-d² / (2·(3·r̄)²))`, a Gaussian in raw distance, which is **maximal at d = 0**. It rewarded two centres for being the same centre, and made the collapsed configuration — every centre at one point, at one scale — the global optimum of the energy: measured on 40 centres in 300×300, collapse had the most negative reinforcement of any configuration tested (−0.391 against −0.154 for a lattice), and only a large locality penalty held it up. Alexander's centres reinforce one another by adjacency, nesting and interlock; two superimposed centres are one centre, and there is nothing there to reinforce. With the peak at contact, a lattice earns more reinforcement than a collapse (−0.216 against −0.161), which is what the term was always meant to express.

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

## 8. The Degree of Life, and the Energy It Negates

The reported quantity is the **degree of life** L — Alexander's own term — and it has
the semantics the theory needs: **zero for a configuration with no structure, higher
for more**. `E = −L` is retained as the quantity `evolve()` minimises, because
simulated annealing descends. Nothing else in the tool reports E.

> **Corrected — "lower energy = greater wholeness" was false, and is no longer the
> framing.** [#28](https://github.com/brunopostle/centres/issues/28)
>
> The old functional's global minimum was the *absence* of structure. 36 centres at
> identical scale, spaced far apart, have no parent-child pairs (identical scales
> admit no parent), no graph edges (spacing beyond the weight threshold) and no
> overlap. Every term was a deviation penalty evaluated only over the objects it
> applies to, so all of them were zero:
>
> ```
> equal scales, far apart:  E = +0.081   edges = 0   pairs = 0   overlap = 0
> random, same canvas:      E = +1.396   edges = 23  pairs = 12  overlap = 0.0034
> real carpets:             E = +0.95 .. +1.74
> ```
>
> **Nothing in the functional rewarded structure existing**, so emptiness scored
> better than any artwork. `evolve()` did not collapse to it only because its move
> set — ±2 px, ×exp(N(0, 0.02)) — is too weak to reach it: the generative mode
> worked by failing to optimise its own objective.
>
> The energy analogy is sound only *with a constraint*. A catenary minimises energy
> subject to fixed endpoints; a soap film minimises area subject to a fixed boundary.
> Without the constraint every such minimum is trivial, and nothing here constrained
> how much structure existed. That is why the sign, not merely the offset, was wrong.

### 8.1 Participation × quality

Three constraints, set by the repository owner on #28: the empty case must score
**zero**, for the optimiser's sake; structure must be **rewarded**; and the reward
must **not** be a sum over centres, since that would cram in as many centres as the
optimiser can fit, contradicting *the void*.

Constraints 2 and 3 together rule out the obvious moves. A sum over centres violates
3. A plain mean over centres satisfies 3 but violates 2, because a mean over an empty
set is undefined rather than zero — and one perfect pair would then score as well as
two hundred.

Each descriptive term is therefore factored into two halves:

```
L_term = participation × quality
L      = Σ_k w_k · p_k · q_k  −  w_L · E_L²
```

**Participation** `p_k ∈ [0, 1]` is the fraction of the centres taking part in that
kind of structure — the fraction whose per-object mean the term is actually taken
over.

| term | the mean is over | participants |
|---|---|---|
| E_H hierarchy | parent-child pairs, one per child | centres with a parent |
| E_A alignment | parent-child pairs, one per child | centres with a parent |
| E_C coverage | parents | centres that are a parent |
| E_R reinforcement | graph edges | centres with at least one edge |
| E_φ field | pixels of the reconstructed field | centres with at least one edge |

**Quality** `q_k ∈ [0, 1]` maps the term's per-object mean deviation D so that ideal
is 1 and poor is 0, as `q = exp(−D / S_k)`; reinforcement, the one term that was
already a reward rather than a deviation, uses the mirror image `q = 1 − exp(−m/S_R)`.
The exponential rather than a clamped `max(0, 1 − D/S)` because a clamp is flat above
S, and a flat region of the objective is one the annealer cannot descend.

The three constraints then hold by construction:

- **Empty scores exactly zero** — participation is zero, so the product is zero. Not
  because a sum has no terms, but because nothing participates.
- **Structure is rewarded** — only structure makes p and q positive.
- **Cramming does not pay** — p is a fraction and q is a mean, so adding weak centres
  raises the denominator of both. Measured over random configurations at *fixed
  spatial density* with N = 32…256, r(L, N) = **+0.14**.
- ***The void* survives** — participation is measured over the centres that exist, so
  a deliberately empty region costs nothing. Only failing to relate the centres you
  *do* have costs.

Because every term is a product of two numbers in [0, 1] and the priorities sum to 1,
**L is bounded in [0, 1]** before the barrier: 0 is "no structure of any kind", 1 is
"every centre participates in every kind of structure, perfectly". Neither end is
reachable by a real image; the six carpets sit at 0.32–0.41.

The priorities `w = {H: 0.3, R: 0.3, C: 0.2, A: 0.1, φ: 0.1}` are the emphasis the
original weights were trying to express and are carried over unchanged. The scale
constants `S_k` are measured, not chosen: each is the **median** of that term over a
fixed 33-case reference ensemble — the six carpets, the null controls that yield
centres, and every frame of the five parametric sweeps, at `--max-size 1024`. Under
the old additive total these constants were the ensemble *standard deviation*, because
their job was unit conversion before addition. Nothing is added across units any more,
so their job is now "the deviation at which this term stops counting as good", and the
median is what fixes that: it puts the typical measured artwork at q = 1/e, where the
map is steepest and discriminates best. The standard deviation would have put the
corpus at q = 0.003–0.05 on three of the five terms — every real image
indistinguishably bad.

> **Measured ordering, and one part of it fails.**
>
> ```
> empty canvas, 0 centres      L = +0.0000
> 36 equal scales, far apart   L = −0.0000
> 40 centres collapsed         L = −1.0000
> concentric 3:1 ladder        L = −0.8041
> lattice of 36                L = +0.2994
> six carpets                  L = +0.3237 .. +0.4047
> white noise                  L = +0.4957
> random blobs                 L = +0.4833
> ```
>
> The first four lines are what #28 asked for: emptiness and collapse are the floor,
> not the optimum, and collapse now loses to a lattice by 1.30 rather than by 0.004.
>
> **The last two are a failure. Noise outscores every carpet.** It is not a failure
> this change introduced: read the functional it replaces as −E, and white noise
> (−0.79) and random blobs (−0.83) also beat all six carpets (−1.71…−0.92). The cause
> is in the measures, not in the framing. Term by term, the carpets beat white noise
> on exactly one of the five — field smoothness — and lose on the other four, chiefly
> because `assign_hierarchy` gives 91% of white-noise centres a parent against 67–83%
> for a carpet, and because dense noise earns more reinforcement per edge. No
> re-weighting of these five terms can order art above noise. That is
> [#21](https://github.com/brunopostle/centres/issues/21)'s territory, not #28's, and
> it is recorded here so the claim is not left standing.

### 8.2 Hierarchy Energy E_H

The deviation from a constant scale ratio between parent and child:

```
E_H = Σ_i (log(r_p(i) / r_i) - log 3)²
D_H = E_H / (number of parent-child pairs)     q_H = exp(-D_H / 0.317)
```

> **⚠ WRONG TARGET, AND WRONG SHAPE.** Salingaros (2025) —
> `docs/salingaros-2025-fifteen-properties.pdf`, the detailed expansion of the
> fifteen properties — gives "optimal magnification factors range between
> approximately **2 to 5**", with 1.5 too close to distinguish and 10 disengaging.
> That is a **band**, not a point, so a quadratic penalty about a single ratio is
> the wrong shape whatever the constant. The "2–4" stated here was unsourced.
> AUDIT.md §12 measures this term's minimum at ratio 2.381 — inside the sourced
> band — so the measure may be less wrong than the target it is scored against.
>
> The source also specifies that scales are "measured **independently in the
> vertical and horizontal directions**". Nothing in this implementation is
> directional.

Natural hierarchies were claimed here to exhibit constant scale ratios of 2–4, with the target ratio of 3 as the midpoint of that range. Minimising E_H produces the power-law distribution of centre sizes P(r) ∝ r^{-γ} that Alexander repeatedly observed. This term encodes his property *levels of scale*.

### 8.3 Reinforcement Energy E_R

Rewards mutual strength between connected centres, normalised by the number of edges:

```
E_R = -(1 / |edges|) · Σ_{ij ∈ edges} W_ij · s_i · s_j
q_R = 1 - exp(-|E_R| / 0.0866)
```

This is the one term that was already a reward rather than a deviation, so its quality map is the mirror image of the others: no reinforcement gives 0, strong reinforcement approaches 1. Higher q_R means pairs of connected centres are both strong. This is the intended measure of *strong centres*: configurations where spatially adjacent, scale-similar centres are all strong together.

Normalisation is essential: without it E_R scales as O(N²). It is divided by the **edge count** rather than by N(N-1)/2 — both kill the O(N²) sum, but a spatially local graph has O(N) edges, so the old denominator left the term decaying as O(1/N).

E_R can no longer be made arbitrarily large by collapsing every centre onto one point, because the kernel of §5 peaks at adjacency rather than at coincidence and coincident centres share no edge at all.

*Note on an earlier formulation.* A previous version used E_R = Σ W_ij (s_i - s_j)², which is minimised by equal but arbitrarily weak strengths — rewarding uniformity and blandness rather than coherent strength. The product formulation corrects this.

### 8.4 Locality Energy E_L — the one barrier

Mean pairwise Gaussian overlap:

```
E_L = (1 / (N(N-1)/2)) · Σ_{i<j} exp( -d_ij² / (r_i + r_j)² )
```

E_L = 1 when all centres are coincident, ≈ 0 when centres are well-separated relative to their scales, and about 1e-3 on a real image. It is the one term subtracted from the degree of life rather than added to it, and the one term that is not a participation × quality descriptor.

It enters **squared, with weight 1**:

- The weight is what the new bound makes of it. The descriptive terms are products of two numbers in [0, 1] weighted by priorities summing to 1, so they contribute at most 1; a unit weight at full coincidence is exactly enough to cancel the best score any configuration could earn. The 5.54 that preceded it — the reinforcement lower bound plus the worst ensemble energy — belonged to the additive framing and went with it, as the 50 before that went with the framing before.
- The square is what keeps the barrier flat where the constraint is not active. E_L divides by all N(N−1)/2 pairs while only O(N) of them overlap, so at fixed spatial density it decays as 1/N. Entering linearly it therefore smuggles the centre-count dependence that #14 removed back into the total: measured over random configurations at fixed density with N = 32…256 it took r(L, N) from +0.11 to **+0.55**, and that was the barrier alone. Squared, it still costs the full 1.0 at coincidence and 0.51 for a tight cluster (E_L = 0.71), but 1e-6 on a real image, and r(L, N) returns to +0.14.

**Why a barrier is still needed at all.** Since the reinforcement kernel was moved from coincidence to adjacency (§5), coincident centres share no edge, so a *plain* collapse earns nothing from any term and needs no barrier. What still needs one is the **concentric** collapse: centres stacked on one point at scales in a 3:1 ladder do have parent-child pairs, and would otherwise collect hierarchy and alignment reward for a configuration with no spatial extent at all — +0.196 barrier-free against a lattice's +0.299, and inside the corpus range once slightly jittered.

**What it should probably become.** E_L as a penalty is arguably backwards theory: Alexander's *deep interlock and ambiguity* says centres should interpenetrate, so a term whose minimum is maximal separation encodes the opposite. Two intensive alternatives were measured as candidate descriptors — mean largest-overlap-per-centre reads 0.20–0.26 on the carpets and 0.56–0.62 on random configurations at fixed density, so it is N-independent and informative, but at barrier weight it would push every random configuration below the empty one. Re-deriving E_L as a descriptor rather than a barrier is the third and last step of [#28](https://github.com/brunopostle/centres/issues/28).

### 8.5 Coverage Energy E_C

Penalises deviation from ideal child-coverage of a parent region:

```
C_i = Σ_{j∈children(i)} r_j² / r_i²
E_C = Σ_i (C_i - 0.65)²
```

C_i ≈ 0.65 means children collectively occupy about 65% of the parent's area — filled without overcrowding. This is intended to encode Alexander's *positive space*: regions well-formed and occupied rather than fragmented or empty.

> **⚠ UNSOURCED, AND THE PROPERTY IS ABOUT SOMETHING ELSE.** The value 0.65 does
> not appear in Salingaros (2025), and neither does any child-area coverage
> figure. What the source says *positive space* means is convexity: "The
> experienced space itself … is typically **convex** … while the enclosing solid
> boundary is mostly **concave**." Area coverage is a different quantity.
> [#24](https://github.com/brunopostle/centres/issues/24)

### 8.6 Alignment Energy E_A

Penalises children whose radial distance from their parent deviates from the preferred range:

```
d_i = dist(c_i, c_{p(i)}) / r_{p(i)}
E_A = Σ_i (d_i - 0.5)²
```

The target 0.5 is the midpoint of a supposed 0.3–0.7 range.

> **⚠ UNSOURCED.** No radial-distance figure appears in Salingaros (2025). What
> the source says *local symmetries* means is **bilateral symmetry about the
> vertical axis**, nested so that one acts on every distinct scale of the
> hierarchy. Radial distance of children from parents is not that.
> [#24](https://github.com/brunopostle/centres/issues/24)
 Too close to the parent's centre (d << 0.3) produces concentric but weakly differentiated structure; too far (d >> 0.7) breaks containment. This encodes *local symmetries* and *deep interlock*.

### 8.7 Field Energy E_φ

Abrupt transitions in the reconstructed wholeness field:

```
E_φ = mean(|∇φ|²)
D_φ = E_φ · r_rms²      q_φ = exp(-D_φ / 0.0252)
```

High q_φ means *gradients* — smooth transitions between regions of varying strength — rather than hard discontinuities. The multiplication by the squared rms centre radius is a units correction: every other deviation here is dimensionless, but mean(|∇φ|²) carries units of 1/pixel², so without it the score would change under a pure resize of the image.

This is the awkward term in the participation × quality scheme, because it is a property of the reconstructed field rather than of any set of centres, and so has no set to take a fraction of. It is given the *reinforcement* participation — the fraction of centres with at least one graph edge. That is the honest reading of what it measures: a lone centre's Gaussian bump has a gradient, but that gradient is an artefact of the reconstruction rather than evidence of structure, and an isolated centre is exactly one with no neighbour whose field meets its own. Without this, the field term alone would keep #28's spread-out, unconnected, equal-scale configuration off zero.

---

## 9. Connection to Alexander's 15 Properties

> **⚠ UNSUPPORTED.** The claim that all 15 properties arise as stable patterns when E is minimised has never been demonstrated, and nothing in the repository tests it. Against it, two ways. First, on synthetic stimuli where the answer is known by construction, the measures were rebuilt against the source (#22) until ten of fifteen now track their own ground truth (AUDIT.md §12/§13) — but that was done by defining each measure *directly*, not by minimising E, so it is evidence the properties can be *computed*, not that they *emerge* from the functional. Second, and directly against the claim: a configuration that minimises E (equivalently, maximises the degree of life) does **not** exhibit the fifteen properties more than dense noise does — noise scores a higher degree of life than every carpet and beats the carpets on eleven of the fifteen individual properties (AUDIT.md §17, [#29](https://github.com/brunopostle/centres/issues/29)). If the properties emerged from E-minimisation, the thing that minimises E would show them; it shows disorder instead.
>
> The **table below is also stale.** It maps each property to an "energy driver" and gives a score formula, but the #22 redefinitions rebuilt eleven of the measures on a region layer read from the image (compactness, solidity, tone, image-domain boundaries), so most of the "energy driver" and "score" cells no longer describe `centres/properties.py`. Read that module and AUDIT.md §15 for what each measure now computes.

The intended claim was that all 15 properties arise as stable patterns when E is minimised. Each can also be computed directly as a scalar score from the centre set, field, and reinforcement graph — see `centres/properties.py` and the `compute_all()` function. The CLI displays all 15 scores after every analysis. *(The specific "energy driver" and formula cells below predate the #22 redefinitions and are no longer accurate; see the note above.)*

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
