"""Run the validation suite and print a triage table for the 15 measures."""

import argparse
import os

import cv2
import numpy as np

from centres.pipeline import analyze
from centres.properties import compute_all, normalize_all

from . import stimuli, transforms

KEYS = [
    "levels_of_scale", "strong_centres", "boundaries", "alternating_repetition",
    "positive_space", "good_shape", "local_symmetries", "deep_interlock",
    "contrast", "gradients", "roughness", "echoes", "the_void",
    "simplicity", "not_separateness",
]

IMAGES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")


def score(img, max_size=1024):
    """Run the full pipeline and return (n_centres, life, raw, normalised).

    ``analyze`` returns the energy E, which is what ``evolve()`` minimises. The
    reported quantity is the degree of life L = -E — zero for a configuration
    with no structure, higher for more (#28) — so the sign is flipped here, once,
    at the point where the harness reads it.
    """
    h, w = img.shape[:2]
    s = min(max_size / max(h, w), 1.0)
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    field, centers, G, energy = analyze(img)
    raw = compute_all(field, centers, G)
    return len(centers), -energy, raw, normalize_all(raw)


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
        n, life, _, norm = score(fn())
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
        n, life, _, norm = score(cv2.imread(os.path.join(IMAGES, f)))
        out[f] = (n, life, norm)
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


def invariance(images, group=transforms.BENIGN, label="benign"):
    """Spread of each measure under transformations that preserve structure."""
    _header(f"Invariance under {label} transformations  (spread, 0-10 scale)")
    noise = {k: 0.0 for k in KEYS}
    flaky = set()
    for path in images:
        img = cv2.imread(path)
        vals = {k: [] for k in KEYS}
        for tname, fn in group.items():
            _, _, _, norm = score(fn(img))
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


def redundancy(corpus_scores):
    """Are the 15 measures 15 independent quantities?"""
    _header("Redundancy")
    rows = [[v[2][k] for k in KEYS] for v in corpus_scores.values()]
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


def sweeps():
    """Do measures track the ground truth they were built to detect?"""
    from scipy.stats import spearmanr
    _header("Ground-truth sweeps  (Spearman rho against the generator parameter)")
    for name, (fn, values, target) in stimuli.SWEEPS.items():
        got = []
        for v in values:
            _, _, raw, _ = score(fn(v))
            got.append(raw[target])
        # A sweep point where the measure is undefined carries no rank
        # information, so it is dropped rather than imputed.
        pairs = [(v, g) for v, g in zip(values, got) if g is not None]
        dropped = len(values) - len(pairs)
        if len(pairs) < 3:
            print(f"  {name:<14} -> {target:<24} undefined at "
                  f"{dropped}/{len(values)} sweep points; no rho")
            continue
        rho = spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic
        flag = "" if abs(rho) > 0.9 else "   <-- does not track its own ground truth"
        note = f"  ({dropped} undefined)" if dropped else ""
        print(f"  {name:<14} -> {target:<24} rho = {rho:+.3f}{flag}{note}")


def generators():
    """Centre counts for every synthetic stimulus.

    Cheap, and always run. A front-end change can look clean on the six carpets
    and still destroy the synthetic stimuli that every later stage depends on:
    dividing the field by the cap rather than by its own maximum was a 0.0-0.7%
    no-op on the corpus while collapsing the lattice generators from 481 centres
    to 4, because the cap does not bite on sparse images and the absolute
    detection threshold then rejects almost everything (#27).

    That regression was invisible from the corpus and obvious here, so this
    stage exists to make it impossible to miss again.
    """
    _header("Generator centre counts  (the instrument every later stage depends on)")
    collapsed = []
    for name, (fn, values, target) in stimuli.SWEEPS.items():
        counts = []
        for v in values:
            n, _, _, _ = score(fn(v))
            counts.append(n)
            if n < 10:
                collapsed.append(f"{name}={v:g} ({n})")
        joined = " ".join(f"{c:>5}" for c in counts)
        params = " ".join(f"{v:>5g}" for v in values)
        print(f"  {name:<14} param {params}")
        print(f"  {'':<14} n     {joined}")
    if collapsed:
        print(f"\n  WARNING — {len(collapsed)} stimuli yield fewer than 10 centres:")
        print(f"    {', '.join(collapsed)}")
        print("  A stimulus the detector cannot see cannot validate anything. Some of")
        print("  these are genuine (a zero-contrast field has nothing to detect); a")
        print("  sudden change in this table after a front-end edit is not.")
    return collapsed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true",
                   help="Skip the slower invariance and sweep stages. NOT sufficient "
                        "to validate a change — see the notice printed at the end.")
    args = p.parse_args()

    nulls = null_controls()
    cs = corpus(nulls)
    generators()
    if not args.quick:
        imgs = [os.path.join(IMAGES, f) for f in sorted(os.listdir(IMAGES))
                if f.endswith((".jpg", ".png"))][:3]
        noise = invariance(imgs)
        triage(cs, noise)
        sweeps()
    redundancy(cs)

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
