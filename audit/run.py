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
    """Run the full pipeline and return (n_centres, energy, raw, normalised)."""
    h, w = img.shape[:2]
    s = min(max_size / max(h, w), 1.0)
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    field, centers, G, energy = analyze(img)
    raw = compute_all(field, centers, G)
    return len(centers), energy, raw, normalize_all(raw)


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
        n, _, _, norm = score(fn())
        out[name] = norm
        _row(name, n, norm)
    blank = out["flat_grey"]
    perfect = [k for k in KEYS if blank[k] is not None and blank[k] >= 9.5]
    undefined = [k for k in KEYS if blank[k] is None]
    if perfect:
        print(f"\n  A featureless grey canvas scores >= 9.5/10 on {len(perfect)} of 15 "
              f"properties:\n    {', '.join(perfect)}")
    print(f"\n  flat_grey: {len(undefined)} of 15 properties undefined, "
          f"{15 - len(undefined)} scored")
    return out


def corpus():
    """Baseline scores for the reference images."""
    _header("Reference corpus")
    _cols()
    out = {}
    for f in sorted(os.listdir(IMAGES)):
        if not f.endswith((".jpg", ".png")):
            continue
        n, e, _, norm = score(cv2.imread(os.path.join(IMAGES, f)))
        out[f] = (n, e, norm)
        _row(f, n, norm)
    ns = [v[0] for v in out.values()]
    es = [v[1] for v in out.values()]
    if len(ns) > 2:
        print(f"\n  Pearson r(structural energy, centre count) = "
              f"{np.corrcoef(ns, es)[0, 1]:.4f}")
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true",
                   help="Skip the slower sweep and invariance stages.")
    args = p.parse_args()

    nulls = null_controls()
    cs = corpus()
    if not args.quick:
        imgs = [os.path.join(IMAGES, f) for f in sorted(os.listdir(IMAGES))
                if f.endswith((".jpg", ".png"))][:3]
        noise = invariance(imgs)
        triage(cs, noise)
        sweeps()
    redundancy(cs)


if __name__ == "__main__":
    main()
