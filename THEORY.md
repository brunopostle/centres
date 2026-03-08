# Theory: Computable Wholeness — A Field of Mutually Reinforcing Centres

This document sets out the mathematical theory implemented in this codebase. The theory is a formalisation of Christopher Alexander's concept of *wholeness* from *The Nature of Order* (2002–2005).

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

Centre strengths evolve by diffusion through the reinforcement graph:

```
s^(t+1) = (1 - β) s^t + α (W_norm s^t)
```

where W_norm is the row-normalised adjacency matrix, α = 0.2, β = 0.05, run for 10 steps.

This is a damped random walk: each centre decays slightly (factor 1-β) and is reinforced by its neighbours (factor α). Centres embedded in clusters of strong, scale-similar neighbours grow stronger; isolated or weakly-connected centres decay. Strengths are clipped to [0, 10].

---

## 7. Field Reconstruction

After propagation, the wholeness field is reconstructed from the updated centres:

```
φ_reconstructed(x) = Σ_i s_i · exp( -|x - x_i|² / (2 r_i²) )
```

This is a superposition of Gaussian kernels, one per centre, weighted by strength and scaled by each centre's spatial extent. It provides a continuous representation of where coherent structure has been identified.

---

## 8. The Energy Functional

Structural energy E is a scalar measuring how far a configuration departs from Alexander's ideal. **Lower energy = greater wholeness.** The total is a weighted sum of six terms:

```
E = 0.3·E_H + 0.3·E_R + 0.2·E_C + 0.1·E_A + 0.1·E_φ + 50·E_L
```

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

Normalisation by N(N-1)/2 is essential: without it E_R scales as O(N²), and the global minimum is a degenerate cluster where all centres coincide (maximising every W_ij to 1 and every s_i to 10 after propagation), outweighing all other terms by orders of magnitude.

*Note on an earlier formulation.* A previous version used E_R = Σ W_ij (s_i - s_j)², which is minimised by equal but arbitrarily weak strengths — rewarding uniformity and blandness rather than coherent strength. The product formulation corrects this.

### 8.3 Locality Energy E_L

Penalises spatial overlap between centres, preventing the degenerate cluster minimum:

```
E_L = (1 / (N(N-1)/2)) · Σ_{i<j} exp( -d_ij² / (r_i + r_j)² )
```

E_L = 1 when all centres are coincident; E_L ≈ 0 when centres are well-separated relative to their scales. The large coefficient (50) ensures this penalty dominates over the O(1) reinforcement gain from clustering, while still allowing nearby centres of similar scale to form genuine local clusters.

### 8.4 Coverage Energy E_C

Penalises deviation from ideal child-coverage of a parent region:

```
C_i = Σ_{j∈children(i)} r_j² / r_i²
E_C = Σ_i (C_i - 0.65)²
```

C_i ≈ 0.65 means children collectively occupy about 65% of the parent's area — filled without overcrowding. This encodes Alexander's *positive space*: regions are well-formed and occupied rather than fragmented or empty.

### 8.5 Alignment Energy E_A

Penalises children whose radial distance from their parent deviates from the preferred range:

```
d_i = dist(c_i, c_{p(i)}) / r_{p(i)}
E_A = Σ_i (d_i - 0.5)²
```

Alexander observed that child centres tend to lie at 0.3–0.7 of the parent radius from the parent centre. The target 0.5 is the midpoint of this range. Too close to the parent's centre (d << 0.3) produces concentric but weakly differentiated structure; too far (d >> 0.7) breaks containment. This encodes *local symmetries* and *deep interlock*.

### 8.6 Field Energy E_φ

Penalises abrupt transitions in the reconstructed wholeness field:

```
E_φ = mean(|∇φ|²)
```

Minimising E_φ encourages *gradients* — smooth transitions between regions of varying strength — rather than hard discontinuities.

---

## 9. Connection to Alexander's 15 Properties

All 15 properties arise as stable patterns when E is minimised. Each can also be computed directly as a scalar score from the centre set, field, and reinforcement graph — see `centres/properties.py` and the `compute_all()` function. The CLI displays all 15 scores after every analysis.

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

Alexander observed approximately 15 levels of scale in highly coherent structures. This follows mathematically from the hierarchy energy: if the minimum-energy scale ratio is k ≈ 3, and the ratio of largest to smallest centre is r_max/r_min ≈ 3^14 ≈ 5×10^6 (roughly the ratio of a city to a brick), then:

```
L = log(r_max / r_min) / log(k) ≈ 14
```

Near the minimum of E_H, centre sizes follow a power law P(r) ∝ r^{-γ}, which is the signature of a scale-invariant (fractal) system. This connects the theory to statistical physics: wholeness corresponds to configurations near a **fixed point of a renormalisation operator** — structures that look similar at every scale of observation.
