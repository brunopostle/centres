import argparse
import json
import sys
import cv2

from .pipeline import analyze, evolve
from .graph import build_graph
from .visualize import visualize
from .properties import compute_all, normalize_all


_PROPERTY_LABELS = [
    ("levels_of_scale", "levels of scale"),
    ("strong_centres", "strong centres"),
    ("boundaries", "boundaries"),
    ("alternating_repetition", "alternating repetition"),
    ("positive_space", "positive space"),
    ("good_shape", "good shape"),
    ("local_symmetries", "local symmetries"),
    ("deep_interlock", "deep interlock"),
    ("contrast", "contrast"),
    ("gradients", "gradients"),
    ("roughness", "roughness"),
    ("echoes", "echoes"),
    ("the_void", "the void"),
    ("simplicity", "simplicity"),
    ("not_separateness", "not-separateness"),
]

_BAR_WIDTH = 10
_BAR_CHARS = " ▏▎▍▌▋▊▉█"


def _bar(score):
    """Render a score in [0, 10] as a fixed-width block bar."""
    score = max(0.0, min(10.0, score))
    total_eighths = round(score * _BAR_WIDTH * 8 / 10)
    full = total_eighths // 8
    frac = total_eighths % 8
    empty = _BAR_WIDTH - full - (1 if frac else 0)
    return "█" * full + (_BAR_CHARS[frac] if frac else "") + " " * empty


def print_properties(raw_scores):
    norm = normalize_all(raw_scores)
    print()
    print("  Alexander's 15 structural properties  (0–10, higher = more present)")
    print(f"  {'':>2}  {'property':<24}  {'score':>5}  {'':10}  raw value")
    print("  " + "─" * 68)
    for n, (key, label) in enumerate(_PROPERTY_LABELS, 1):
        s = norm[key]
        print(f"  {n:>2}  {label:<24}  {s:>5.1f}  {_bar(s)}  {raw_scores[key]:.4g}")
    print()


def load_and_rescale(path: str, max_size: int):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Error: cannot read image '{path}'")
    h, w = img.shape[:2]
    scale = min(max_size / max(h, w), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"Rescaled {w}x{h} -> {new_w}x{new_h}")
    return img


def _emit(args, n_centers, energy, raw_scores):
    """Print or emit JSON results depending on --json flag."""
    norm = normalize_all(raw_scores)
    if args.json:
        out = {
            "centres": n_centers,
            "structural_energy": round(energy, 4),
            "properties": {
                key: {"score": round(norm[key], 2), "raw": round(raw_scores[key], 6)}
                for key, _ in _PROPERTY_LABELS
            },
        }
        print(json.dumps(out, indent=2))
    else:
        print(f"Centers: {n_centers}")
        print(f"Structural energy: {energy:.4f}")
        print_properties(raw_scores)


def cmd_analyse(args):
    img = load_and_rescale(args.image, args.max_size)
    field, centers, G, energy = analyze(img)
    raw = compute_all(field, centers, G)
    _emit(args, len(centers), energy, raw)
    if not args.json and (not args.no_display or args.save):
        visualize(field, centers, G, img, save_path=args.save if args.save else None)


def cmd_evolve(args):
    initial_centers = None
    if args.image:
        img = load_and_rescale(args.image, args.max_size)
        h, w = img.shape[:2]
        shape = (h, w)
        if not args.json:
            print(f"Seeding from image ({w}x{h}) ...")
        _, initial_centers, _, _ = analyze(img)
        if not args.json:
            print(f"Detected {len(initial_centers)} initial centres")
    else:
        shape = (args.size, args.size)

    n = len(initial_centers) if initial_centers else args.n_centres
    if not args.json:
        print(f"Evolving {n} centres for {args.iterations} iterations ...")

    def progress(t, total, energy, T, accepted):
        if args.json:
            return
        mark = "+" if accepted else " "
        print(
            f"\r  [{mark}] {t:4d}/{total}  T={T:.3f}  E={energy:+.2f}",
            end="",
            flush=True,
        )
        if t == total:
            print()

    field, centers = evolve(
        shape=shape,
        iterations=args.iterations,
        n_centers=args.n_centres,
        progress=progress,
        initial_centers=initial_centers,
    )
    if not args.json:
        print()
    G = build_graph(centers)
    from .graph import propagate_strength
    from .energy import total_energy

    G = propagate_strength(G)
    energy = total_energy(field, centers, G)
    raw = compute_all(field, centers, G)
    _emit(args, len(centers), energy, raw)
    if not args.json and (not args.no_display or args.save):
        visualize(field, centers, G, save_path=args.save if args.save else None)


def main():
    parser = argparse.ArgumentParser(
        description="Computable wholeness: structural centre field analysis after Christopher Alexander.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- analyse ---
    p_analyse = subparsers.add_parser(
        "analyse",
        help="Detect centres and measure structural energy of an existing image.",
    )
    p_analyse.add_argument("image", help="Path to input image")
    p_analyse.add_argument(
        "--max-size",
        type=int,
        default=1024,
        metavar="PX",
        help="Downscale so the longest dimension is at most PX pixels (default: 1024).",
    )
    p_analyse.add_argument(
        "--no-display", action="store_true", help="Skip interactive visualisation."
    )
    p_analyse.add_argument(
        "--save",
        metavar="PATH",
        help="Save visualisation to this file instead of displaying.",
    )
    p_analyse.add_argument(
        "--json",
        action="store_true",
        help="Emit results as JSON and suppress all other output.",
    )
    p_analyse.set_defaults(func=cmd_analyse)

    # --- evolve ---
    p_evolve = subparsers.add_parser(
        "evolve",
        help="Generate a high-wholeness centre configuration from scratch via simulated annealing.",
    )
    p_evolve.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Optional input image. If given, seed the search from its detected "
        "centres rather than from a random configuration.",
    )
    p_evolve.add_argument(
        "--max-size",
        type=int,
        default=1024,
        metavar="PX",
        help="Downscale input image so its longest dimension is at most PX pixels "
        "(default: 1024). Only used when an image is provided.",
    )
    p_evolve.add_argument(
        "--size",
        type=int,
        default=512,
        metavar="PX",
        help="Canvas size in pixels when no image is provided (square, default: 512).",
    )
    p_evolve.add_argument(
        "--n-centres",
        type=int,
        default=40,
        metavar="N",
        help="Number of centres when no image is provided (default: 40).",
    )
    p_evolve.add_argument(
        "--iterations",
        type=int,
        default=200,
        metavar="N",
        help="Number of simulated annealing iterations (default: 200).",
    )
    p_evolve.add_argument(
        "--no-display", action="store_true", help="Skip interactive visualisation."
    )
    p_evolve.add_argument(
        "--save",
        metavar="PATH",
        help="Save visualisation to this file instead of displaying.",
    )
    p_evolve.add_argument(
        "--json",
        action="store_true",
        help="Emit results as JSON and suppress all other output.",
    )
    p_evolve.set_defaults(func=cmd_evolve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
