"""Run the validation suite and print a triage table for the 15 measures."""

import argparse
import os

import cv2
import numpy as np

from centres.pipeline import analyze
from centres.properties import compute_all, normalize_all

from . import stimuli, transforms
from . import redundancy as rdcy

KEYS = [
    "levels_of_scale", "strong_centres", "boundaries", "alternating_repetition",
    "positive_space", "good_shape", "local_symmetries", "deep_interlock",
    "contrast", "gradients", "roughness", "echoes", "the_void",
    "simplicity", "not_separateness",
]

IMAGES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")


def _redundancy(centers):
    """The three global-redundancy statistics of a centre set (see audit.redundancy).

    Bundled here so the one ``analyze`` call ``score`` already makes yields them
    too, and the ``discrimination`` stage can reuse the corpus scores instead of
    re-analysing every corpus image a second time.
    """
    return {
        "strength_entropy": rdcy.strength_entropy(centers),
        "scale_entropy": rdcy.scale_entropy(centers),
        "spatial_coherence": rdcy.spatial_coherence(centers),
    }


def score(img, max_size=1024):
    """Run the full pipeline and return (n_centres, life, raw, normalised, redundancy).

    ``analyze`` returns the energy E, which is what ``evolve()`` minimises. The
    reported quantity is the degree of life L = -E — zero for a configuration
    with no structure, higher for more (#28) — so the sign is flipped here, once,
    at the point where the harness reads it. The fifth value is the redundancy
    dict, computed from the same centre set so no caller has to analyse twice.
    """
    h, w = img.shape[:2]
    s = min(max_size / max(h, w), 1.0)
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    field, centers, G, energy = analyze(img)
    raw = compute_all(field, centers, G, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return len(centers), -energy, raw, normalize_all(raw), _redundancy(centers)


def _header(title):
    print(f"\n{title}\n" + "=" * len(title))


def _cell(v):
    """A measure may be None — undefined for these inputs, not zero."""
    return "   -" if v is None else f"{v:4.1f}"


def _defined(values):
    return [v for v in values if v is not None]


def _spread(values):
    """Range over the defined values; 0 when fewer than two are defined."""
    d = _defined(values)
    return float(np.ptp(d)) if len(d) > 1 else 0.0


def _row(label, n, norm, width=18):
    print(f"  {label:<{width}}{n:>6}  " + " ".join(_cell(norm[k]) for k in KEYS))


def _cols(width=18):
    abbr = ["LoS", "Str", "Bnd", "Alt", "Pos", "Shp", "Sym", "Int",
            "Con", "Grd", "Rgh", "Ech", "Voi", "Sim", "NSp"]
    print(f"  {'':<{width}}{'n':>6}  " + " ".join(f"{a:>4}" for a in abbr))


def null_controls():
    """Structureless and precisely-structured inputs. High scores here are false."""
    _header("Null controls  (scores should be low or explicitly undefined)")
    _cols()
    out = {}
    for name, fn in stimuli.NULL_CONTROLS.items():
        n, life, _, norm, _ = score(fn())
        out[name] = (n, life, norm)
        _row(name, n, norm)
    print("\n  degree of life:  " + "  ".join(
        f"{k} {v[1]:+.4f}" for k, v in out.items()))
    blank = out["flat_grey"][2]
    perfect = [k for k in KEYS if blank[k] is not None and blank[k] >= 9.5]
    undefined = [k for k in KEYS if blank[k] is None]
    if perfect:
        print(f"\n  A featureless grey canvas scores >= 9.5/10 on {len(perfect)} of 15 "
              f"properties:\n    {', '.join(perfect)}")
    print(f"\n  flat_grey: {len(undefined)} of 15 properties undefined, "
          f"{15 - len(undefined)} scored")
    return out


def corpus(controls=None):
    """Baseline scores for the reference images.

    The correlation between the reported score and the centre count is reported
    over the corpus alone and over the corpus plus the synthetic controls. The
    acceptance criterion on #14, restated on #28, is |r| < 0.5 over the second
    of those: a score that tracks the centre count is a readout of the
    detector's sensitivity, not a measure of the image, and the corpus on its
    own spans too narrow a range of counts to show it.
    """
    _header("Reference corpus")
    _cols()
    out = {}
    for f in sorted(os.listdir(IMAGES)):
        if not f.endswith((".jpg", ".png")):
            continue
        n, life, _, norm, red = score(cv2.imread(os.path.join(IMAGES, f)))
        out[f] = (n, life, norm, red)
        _row(f, n, norm)
    print("\n  degree of life:  " + "  ".join(
        f"{k.split('.')[0]} {v[1]:+.4f}" for k, v in out.items()))
    ns = [v[0] for v in out.values()]
    ls = [v[1] for v in out.values()]
    if len(ns) > 2:
        print(f"\n  Pearson r(degree of life, centre count), corpus       = "
              f"{np.corrcoef(ns, ls)[0, 1]:+.4f}")
    if controls:
        # flat_grey detects nothing at all, so it carries no centre count to
        # correlate against and is excluded rather than imputed as zero.
        extra = [(v[0], v[1]) for v in controls.values() if v[0] > 0]
        ns2 = ns + [x[0] for x in extra]
        ls2 = ls + [x[1] for x in extra]
        print(f"  Pearson r(degree of life, centre count), + controls   = "
              f"{np.corrcoef(ns2, ls2)[0, 1]:+.4f}   (target |r| < 0.5)")
    return out


#: The three dense-noise controls the score must be ranked *below*. ``flat_grey``
#: detects nothing and ``regular_grid`` is a mechanical lattice, not noise; both
#: are reported by ``null_controls`` and neither is the #29 comparison, which is
#: specifically composed structure against a dense random field.
NOISE_CONTROLS = ("white_noise", "smooth_noise", "random_blobs")


def _analyze_scaled(img, max_size=1024):
    """Resize exactly as ``score`` does, then return life and the redundancy stats.

    Kept separate from ``score`` because the discrimination stage needs the centre
    set (to take the distribution entropies) and ``score`` deliberately returns
    only scalars. The resize must match ``score`` so the degree of life reported
    here is the same number the corpus table shows.
    """
    h, w = img.shape[:2]
    s = min(max_size / max(h, w), 1.0)
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    _, centers, _, energy = analyze(img)
    return (-energy,
            rdcy.strength_entropy(centers),
            rdcy.scale_entropy(centers),
            rdcy.spatial_coherence(centers))


def discrimination(corpus_scores):
    """Does the reported score rank artworks above dense noise?  (#29)

    The central open problem. Section 2b showed uniform noise scoring a higher
    degree of life than every carpet, and it is not an artefact of one functional:
    no *local* measure separates the two, because dense noise is not short of
    local structure — it has more edges, more parents and more neighbours than any
    carpet. This stage states that failure as a number the harness prints on every
    run, and alongside it measures the global-redundancy and spatial-coherence
    discriminators from ``audit.redundancy`` — and the combination of them that
    *does* separate the corpus from noise. None is yet in the score, for the
    reasons that module and #29/#9/#34 record.

    The metric is rank separation: the fraction of (artwork, noise) pairs ordered
    the way life demands. 1.0 is complete separation, 0.5 none, 0.0 fully inverted.

    The artwork side is read from ``corpus_scores`` — the scores ``corpus`` already
    computed — rather than re-analysed here, so the corpus is analysed once per run,
    not twice.
    """
    _header("Discrimination: does the score rank artworks above noise?  (#29)")

    # (life, strength_entropy, scale_entropy, spatial_coherence) per artwork, taken
    # from the corpus stage's results (index 1 is life, index 3 the redundancy dict).
    art = {f: (v[1], v[3]["strength_entropy"], v[3]["scale_entropy"],
               v[3]["spatial_coherence"])
           for f, v in corpus_scores.items()}
    # Several seeds per noise generator, not one. The noise floor is itself a
    # sampled quantity — a smooth-noise strength entropy ranges 2.54–2.59 across
    # seeds — and a single realisation gives an optimistic separation: the
    # Egyptian geometric tile (2.55) sits below seed 0's floor and above seed 1's,
    # so one seed reports a clean 1.000 and three seeds report the honest 0.99x.
    noise = {f"{name}{seed}": _analyze_scaled(getattr(stimuli, name)(seed=seed))
             for name in NOISE_CONTROLS for seed in range(3)}
    grid = _analyze_scaled(stimuli.regular_grid())  # a mechanical reference point

    def col(d, i):
        return [v[i] for v in d.values()]

    def line(name, art_vals, noise_vals, higher_is_life, note="", is_score=False):
        a, z = _defined(art_vals), _defined(noise_vals)
        sep = rdcy.pairwise_order(a, z, higher_is_life)
        arrow = "higher" if higher_is_life else "lower"
        # The pass/fail-of-#29 verdict belongs only to the reported score; a
        # candidate's separation being below 1.0 is information about the candidate,
        # not a statement that the tool fails #29.
        tag = ("(complete)" if sep == 1.0 else "(FAILS #29)") if is_score else ""
        print(f"  {name:<22} art [{min(a):+.3f},{max(a):+.3f}]  "
              f"noise [{min(z):+.3f},{max(z):+.3f}]  ({arrow}=life)")
        print(f"  {'':<22} rank separation art-over-noise = {sep:.3f}   {tag}{note}")
        return sep

    print("  reported degree of life:")
    line("  degree of life", col(art, 0), col(noise, 0), True, is_score=True)
    print("\n  candidate discriminators (NOT in the score; see audit/redundancy.py):")
    se = line("  strength_entropy", col(art, 1), col(noise, 1), False,
              note="   redundancy, transform-stable")
    line("  scale_entropy", col(art, 2), col(noise, 2), False,
         note="   redundancy, transform-FRAGILE (#9)")
    line("  spatial_coherence", col(art, 3), col(noise, 3), True,
         note="   Moran's I of strength; complements entropy")

    # The combination that actually separates the wider corpus: alive if redundant
    # OR spatially coherent. The two axes fail on disjoint artworks, so the OR
    # clears every one where neither single axis does. See AUDIT.md §17.
    frac, margin = rdcy.combined_separation(col(art, 1), col(art, 3),
                                            col(noise, 1), col(noise, 3))
    print(f"\n  COMBINED — alive if (strength_entropy < noise floor) OR "
          f"(spatial_coherence > noise ceiling):")
    print(f"  {'':<22} {frac * len(art):.0f}/{len(art)} artworks separated from noise"
          f"   tightest margin {margin:+.3f}"
          f"   {'(all separated)' if frac == 1.0 else '(some cross both axes)'}")

    print(f"\n  mechanical reference: regular_grid  "
          f"strength_entropy {grid[1]:.3f}, scale_entropy {grid[2]:.3f}, "
          f"spatial_coherence {grid[3]:.3f}")
    return {"life_sep": rdcy.pairwise_order(col(art, 0), col(noise, 0), True),
            "strength_entropy_sep": se, "combined_sep": frac}


def invariance(images, group=transforms.BENIGN, label="benign"):
    """Spread of each measure under transformations that preserve structure."""
    _header(f"Invariance under {label} transformations  (spread, 0-10 scale)")
    noise = {k: 0.0 for k in KEYS}
    flaky = set()
    for path in images:
        img = cv2.imread(path)
        vals = {k: [] for k in KEYS}
        for tname, fn in group.items():
            _, _, _, norm, _ = score(fn(img))
            for k in KEYS:
                vals[k].append(norm[k])
        for k in KEYS:
            noise[k] = max(noise[k], _spread(vals[k]))
            if 0 < len(_defined(vals[k])) < len(vals[k]):
                flaky.add(k)
        _row(os.path.basename(path), 0, {k: noise[k] for k in KEYS})
    if flaky:
        print(f"\n  defined for some transforms and undefined for others: "
              f"{', '.join(sorted(flaky))}")
    return noise


def triage(corpus_scores, noise):
    """Compare between-image signal against within-image noise."""
    _header("Triage: is the spread between artworks larger than the measurement noise?")
    signal = {k: _spread([v[2][k] for v in corpus_scores.values()]) for k in KEYS}
    print(f"  {'property':<24}{'signal':>8}{'noise':>8}{'SNR':>8}   verdict")
    print("  " + "-" * 62)
    for k in KEYS:
        snr = signal[k] / (noise[k] + 1e-9)
        v = "usable" if snr >= 2 else ("marginal" if snr >= 1 else "NOISE")
        print(f"  {k:<24}{signal[k]:8.1f}{noise[k]:8.1f}{snr:8.2f}   {v}")


def redundancy(corpus_scores, extra=()):
    """Are the 15 measures 15 independent quantities?

    Computed over the corpus *and* every other scored stimulus. Over the six
    corpus images alone the question cannot be asked: with 6 samples and 15
    variables the correlation matrix has rank at most 5, so a finding that "3 or 4
    components explain 90%" is close to vacuous -- there are at most 5 non-zero
    components to begin with. The original audit reported exactly that, and it was
    largely an artefact of the sample size. Over a 33-case ensemble the answer is
    6 components for 90% of the variance and 8 for 95%, so the measures are
    somewhat more independent than the corpus-only figure suggested.
    """
    _header("Redundancy")
    rows = [[v[2][k] for k in KEYS] for v in corpus_scores.values()]
    rows += [[n[k] for k in KEYS] for n in extra]
    complete = [r for r in rows if all(x is not None for x in r)]
    if len(complete) < len(rows):
        print(f"  ({len(rows) - len(complete)} of {len(rows)} images dropped: "
              f"some properties undefined)")
    M = np.array(complete, dtype=float)
    if len(M) < 4:
        print("  (needs at least 4 images with all 15 properties defined)")
        return
    C = np.corrcoef(M.T)
    pairs = sorted(((abs(C[i, j]), KEYS[i], KEYS[j])
                    for i in range(15) for j in range(i + 1, 15)), reverse=True)
    print("  most redundant pairs:")
    for v, a, b in pairs[:5]:
        print(f"    {v:.2f}  {a} ~ {b}")
    ev = np.linalg.eigvalsh(C)[::-1]
    cum = np.cumsum(ev) / ev.sum()
    print(f"\n  {int(np.argmax(cum > 0.90)) + 1} principal components explain 90% "
          f"of the variance across 15 'independent' measures")


def measure_generators():
    """Score every generator stimulus once, and return the results.

    Both the centre-count stage and the sweep stage need the same ~180 scores, and
    each score is several seconds. They are computed here and shared rather than
    computed twice. The sensitivity matrix needs every measure on every stimulus,
    so the whole score dicts are kept rather than only the generator's target.
    """
    out = {}
    for name, (fn, values, target, test) in stimuli.SWEEPS.items():
        rows = []
        for v in values:
            n, _, raw, norm, _ = score(fn(v))
            rows.append((v, n, raw, norm))
        out[name] = (target, test, rows)
    return out


def generators(measured):
    """Centre counts for every synthetic stimulus.

    Cheap relative to the rest, and always run. A front-end change can look clean
    on the six carpets and still destroy the synthetic stimuli that every later
    stage depends on: dividing the field by the cap rather than by its own maximum
    was a 0.0-0.7% no-op on the corpus while collapsing the lattice generators
    from 481 centres to 4, because the cap does not bite on sparse images and the
    absolute detection threshold then rejects almost everything (#27).

    That regression was invisible from the corpus and obvious here, so this stage
    exists to make it impossible to miss again.
    """
    _header("Generator centre counts  (the instrument every later stage depends on)")
    collapsed = []
    for name, (target, _, rows) in measured.items():
        counts = [n for _, n, _, _ in rows]
        for v, n, _, _ in rows:
            if n < 10:
                collapsed.append(f"{name}={v:g} ({n})")
        print(f"  {name:<18} param " + " ".join(f"{v:>5g}" for v, _, _, _ in rows))
        print(f"  {'':<18} n     " + " ".join(f"{c:>5}" for c in counts))
    if collapsed:
        print(f"\n  WARNING — {len(collapsed)} stimuli yield fewer than 10 centres:")
        print(f"    {', '.join(collapsed)}")
        print("  A stimulus the detector cannot see cannot validate anything. Some of")
        print("  these are genuine (a zero-contrast field has nothing to detect); a")
        print("  sudden change in this table after a front-end edit is not.")
    return collapsed


#: A measure tracks its ground truth if its rank correlation with the parameter is
#: at least this strong, in the *expected direction*. 0.5 is a genuine relationship
#: at n = 12 (p < 0.1), and it is the honest bar: the old code flagged anything
#: below 0.9 as "does not track its own ground truth", but 0.9 is a test for
#: perfection — one rank swap costs 0.03 at n = 12 — and it mislabelled measures
#: that clearly move the right way. echoes runs at -0.85 and local symmetries at
#: -0.72 *by design* (more distinct shapes, less echo; more shear, less symmetry),
#: and calling those "does not track" was simply wrong. The rho is still printed in
#: full, so the difference between a perfect tracker (0.99) and a moderate one
#: (0.66) is visible; the flag is reserved for real failures.
TRACKS = 0.5


def _monotone_verdict(name, target, pairs, dropped, expect):
    """Does the measure move with the ground truth, in the direction it should?

    ``expect`` is +1 if the measure should rise with the parameter, -1 if it should
    fall. Three outcomes: it tracks (correct direction, |rho| >= TRACKS), it runs
    backwards (wrong direction, not negligibly), or it does not track (|rho| below
    TRACKS, no monotone relationship either way).
    """
    from scipy.stats import spearmanr
    rho = spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic
    arrow = "↑" if expect >= 0 else "↓"   # the direction it should move
    if abs(rho) < TRACKS:
        flag = "   <-- does not track its own ground truth"
    elif expect * rho < 0:
        flag = f"   <-- runs BACKWARDS (should move {arrow})"
    else:
        flag = ""
    note = f"  ({dropped} undefined)" if dropped else ""
    print(f"  {name:<18} -> {target:<24} monotone {arrow}  rho = {rho:+.3f}{flag}{note}")
    return rho


def _optimum_verdict(name, target, pairs, claimed, peak, dropped):
    """For a measure with an interior ideal, report where the ideal actually is.

    The measure should be *extremal* at the parameter value the theory names and
    fall away on both sides, so a high rho over the whole sweep would be evidence
    *against* it — rho is the wrong statistic and is not printed. What is printed is
    the location of the extremum against the claimed one, and the one-sided rank
    correlations, which is what the shape actually asserts.

    ``peak`` says which extremum. The #22 redefinitions made these measures
    ``exp(-deviation)``, *maximal* at the ideal (``boundaries`` peaks where the
    band is 1/3 of what it bounds; ``levels_of_scale`` where the ratio is in band),
    so ``peak`` is True and the extremum is the *maximum*. A raw squared deviation
    would be *minimal* there instead. The previous code always took the minimum,
    which put a peak measure's extremum at a sweep end and mislabelled it as "not
    where the measure claims" — a harness bug, not a measure failure.
    """
    from scipy.stats import spearmanr
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    at = xs[int(np.argmax(ys) if peak else np.argmin(ys))]
    kind = "maximum" if peak else "minimum"
    left = [(x, y) for x, y in pairs if x <= claimed]
    right = [(x, y) for x, y in pairs if x >= claimed]
    lo = (spearmanr([p[0] for p in left], [p[1] for p in left]).statistic
          if len(left) > 2 else None)
    hi = (spearmanr([p[0] for p in right], [p[1] for p in right]).statistic
          if len(right) > 2 else None)
    ok = abs(at - claimed) <= 0.5 * (max(xs) - min(xs)) / (len(xs) - 1)
    flag = "" if ok else f"   <-- {kind} is not where the measure claims it is"
    note = f"  ({dropped} undefined)" if dropped else ""
    print(f"  {name:<18} -> {target:<24} optimum   {kind} at {at:g}, "
          f"claimed {claimed:g}{flag}{note}")
    # A peak rises then falls (+ then -); a valley falls then rises (- then +).
    want = "+ then -" if peak else "- then +"
    print(f"  {'':<18}    {'':<24} rising below rho = "
          f"{'  n/a' if lo is None else f'{lo:+.3f}'}, falling above rho = "
          f"{'  n/a' if hi is None else f'{hi:+.3f}'}   (this shape wants {want})")
    return at


def sweeps(measured):
    """Do measures track the ground truth they were built to detect?

    Two tests, chosen per generator in ``stimuli.SWEEPS`` and named in each row:

    *monotone* — the measure should move one way with the parameter throughout,
    and Spearman's rho over the whole sweep is the statistic.

    *optimum* — the measure claims an ideal in the *middle* of the sweep, so it
    should be extremal there rather than monotone, and a high rho would be
    evidence against it. See ``_optimum_verdict``.

    Every row also reports where the *normalised* 0-10 score peaks, which is the
    parameter value the tool would call ideal if asked. For a monotone property
    that should be an end of the sweep; anywhere else means the normaliser has
    put an optimum where the measure does not have one.
    """
    _header("Ground-truth sweeps  (against the generator parameter; test named per row)")
    for name, (target, (kind, arg), rows) in measured.items():
        # A sweep point where the measure is undefined carries no rank
        # information, so it is dropped rather than imputed.
        pairs = [(v, raw[target]) for v, _, raw, _ in rows if raw[target] is not None]
        scored = [(v, norm[target]) for v, _, _, norm in rows if norm[target] is not None]
        dropped = len(rows) - len(pairs)
        if len(pairs) < 3:
            print(f"  {name:<18} -> {target:<24} undefined at "
                  f"{dropped}/{len(rows)} sweep points; no statistic")
            continue
        if kind == "optimum":
            claimed, peak = arg
            _optimum_verdict(name, target, pairs, claimed, peak, dropped)
        else:
            _monotone_verdict(name, target, pairs, dropped, expect=arg)
        peak = max(scored, key=lambda p: p[1])[0]
        print(f"  {'':<18}    {'':<24} normalised score peaks at {peak:g}"
              f"  (sweep spans {min(v for v, _ in pairs):g} to "
              f"{max(v for v, _ in pairs):g})")



def sensitivity(measured):
    """Every measure against every generator: does each respond to its own?

    Knowing that a measure tracks *its own* generator is necessary but nowhere
    near sufficient. A measure that responds just as strongly to every other
    generator is not isolating anything -- it is reporting some general property
    of the stimulus, and the name on it is decoration.

    The diagonal is each measure against the generator built for it. Dominance is
    |rho_own| / max |rho_other|: above 1 the measure responds most to its own
    parameter, below 1 something else moves it more.

    Costs nothing extra -- the scores are already computed for the sweep stage.
    """
    from scipy.stats import spearmanr

    gens = list(measured.keys())
    targets = {name: measured[name][0] for name in gens}

    rho = {}
    for gname in gens:
        _, _, rows = measured[gname]
        for m in KEYS:
            pairs = [(v, raw[m]) for v, _, raw, _ in rows if raw[m] is not None]
            rho[(m, gname)] = (
                float(spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic)
                if len(pairs) >= 3 else float("nan")
            )

    _header("Sensitivity matrix  (|Spearman rho|, measure x generator)")
    abbr = {g: g[:6] for g in gens}
    print(f"  {'measure':<24}" + "".join(f"{abbr[g]:>7}" for g in gens))
    print("  " + "-" * (24 + 7 * len(gens)))
    for m in KEYS:
        own = next((g for g in gens if targets[g] == m), None)
        cells = []
        for g in gens:
            v = rho[(m, g)]
            cells.append("     ." if np.isnan(v) else
                         (f"[{abs(v):4.2f}]" if g == own else f"{abs(v):7.2f}"))
        print(f"  {m:<24}" + "".join(cells))
    print("\n  [x] marks the generator built for that measure.")

    print(f"\n  {'measure':<24}{'own':>7}{'strongest other':>17}{'dominance':>11}")
    print("  " + "-" * 60)
    weak = []
    for m in KEYS:
        own = next((g for g in gens if targets[g] == m), None)
        o = abs(rho[(m, own)])
        others = [(abs(rho[(m, g)]), g) for g in gens if g != own and not np.isnan(rho[(m, g)])]
        best, bg = max(others) if others else (float("nan"), "-")
        dom = o / best if best else float("inf")
        if not (dom > 1.0):
            weak.append(m)
        print(f"  {m:<24}{o:7.2f}{best:10.2f} {bg:<6}{dom:11.2f}")
    if weak:
        print(f"\n  {len(weak)} of {len(KEYS)} measures respond more strongly to some other")
        print(f"  generator than to their own:\n    {', '.join(weak)}")
    return rho



def count_confound(measured):
    """Is the centre count the common driver behind the sensitivity matrix?

    Section 14 found that all fifteen measures respond more strongly to some
    other generator than to their own, and that two generators - dominance and
    void_size - drive most of them. Both make large changes to gross composition,
    and gross composition changes the number of detected centres: across the
    generators the count swings from 37 to 1169.

    So the obvious hypothesis is that there is one underlying quantity, the
    centre count, and the fifteen measures are fifteen views of it.

    Two tests. First, the pooled rank correlation between each measure and the
    count over every generator stimulus. Second - the one that decides it - the
    *partial* correlation between each generator's parameter and its own target
    measure, holding the count fixed. If the count is the whole story the raw and
    partial figures diverge sharply; if the diagonal survives partialling, the
    measures are seeing something the count does not carry.
    """
    from scipy.stats import spearmanr

    pooled_n, pooled_m = [], {k: [] for k in KEYS}
    for _, _, rows in measured.values():
        for v, n, raw, _ in rows:
            pooled_n.append(n)
            for k in KEYS:
                pooled_m[k].append(raw[k])

    _header("Centre-count confound  (is the count the common driver?)")
    print(f"  {'measure':<24}{'rho vs count':>14}   pooled over "
          f"{len(pooled_n)} stimuli, counts {min(pooled_n)}-{max(pooled_n)}")
    print("  " + "-" * 60)
    for k in KEYS:
        pairs = [(a, b) for a, b in zip(pooled_n, pooled_m[k]) if b is not None]
        r = spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic
        flag = "  <-- tracks the count" if abs(r) >= 0.7 else ""
        print(f"  {k:<24}{r:+14.3f}{flag}")

    def _partial(x, y, z):
        """Spearman partial correlation of x and y controlling for z.

        Returns the raw correlation when z is constant: there is nothing to
        control for, because the generator already holds the count fixed by
        construction. That is the strongest possible position for a sweep to be
        in, not a failure -- ``tonal_delta`` and ``zone_width`` both hold the
        count exactly constant, so their figures are count-free already.
        """
        if len(set(z)) < 2:
            return spearmanr(x, y).statistic
        rx = spearmanr(x, y).statistic
        rz1 = spearmanr(x, z).statistic
        rz2 = spearmanr(y, z).statistic
        denom = np.sqrt((1 - rz1**2) * (1 - rz2**2))
        return float("nan") if denom == 0 else (rx - rz1 * rz2) / denom

    print(f"\n  Each generator against its own target, before and after holding "
          f"the count fixed:")
    print(f"  {'generator -> measure':<40}{'raw':>8}{'partial':>10}{'change':>9}")
    print("  " + "-" * 68)
    survived = 0
    for gname, (target, _, rows) in measured.items():
        trip = [(v, raw[target], n) for v, n, raw, _ in rows if raw[target] is not None]
        if len(trip) < 4:
            print(f"  {gname + ' -> ' + target:<40}{'too few defined points':>27}")
            continue
        xs = [t[0] for t in trip]
        ys = [t[1] for t in trip]
        ns = [t[2] for t in trip]
        raw_r = spearmanr(xs, ys).statistic
        par_r = _partial(xs, ys, ns)
        fixed = "  (count constant)" if len(set(ns)) < 2 else ""
        if abs(par_r) >= abs(raw_r) - 0.1:
            survived += 1
        print(f"  {gname + ' -> ' + target:<40}{raw_r:+8.2f}{par_r:+10.2f}"
              f"{abs(par_r) - abs(raw_r):+9.2f}{fixed}")
    print(f"\n  {survived} of {len(measured)} diagonals survive partialling out the "
          f"count (lose < 0.1).")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true",
                   help="Skip the invariance, triage and sweep stages. Saves less "
                        "than it used to — the sweeps now reuse the generator scores "
                        "rather than recomputing them, so only invariance is skipped. "
                        "NOT sufficient to validate a change — see the notice printed "
                        "at the end.")
    args = p.parse_args()

    nulls = null_controls()
    cs = corpus(nulls)
    discrimination(cs)
    measured = measure_generators()
    generators(measured)
    if not args.quick:
        imgs = [os.path.join(IMAGES, f) for f in sorted(os.listdir(IMAGES))
                if f.endswith((".jpg", ".png"))][:3]
        noise = invariance(imgs)
        triage(cs, noise)
        sweeps(measured)
        sensitivity(measured)
        count_confound(measured)
    extra = [norm for _, _, rows in measured.values() for _, _, _, norm in rows]
    extra += [v[2] for v in nulls.values()]
    redundancy(cs, extra)

    print()
    if args.quick:
        print("  " + "=" * 68)
        print("  VALIDATION INCOMPLETE — --quick skipped invariance, triage and the")
        print("  ground-truth sweeps. Do not report a change as verified on this run.")
        print("  Every change is measured against the corpus AND the generators, with")
        print("  the full `python -m audit`. Corpus-only validation has hidden a")
        print("  regression twice (#27, and the crop criterion on #11).")
        print("  " + "=" * 68)
    else:
        print("  Full run: corpus, generators, invariance, triage and sweeps.")


if __name__ == "__main__":
    main()
