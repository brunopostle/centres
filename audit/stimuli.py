"""Synthetic stimuli with structure known by construction.

Two kinds:

*Null controls* contain no coherent centre structure (or, for ``regular_grid``,
structure of a precisely known kind). A measure that reports high wholeness
for these is not measuring wholeness.

*Parametric generators* vary one structural property along a known scalar
while holding the rest fixed, so a measure can be checked for monotonicity
and calibration against ground truth rather than merely for plausibility.
"""

import cv2
import numpy as np

CANVAS = 1024
_BG = 245
_FG = (30, 30, 30)


# --- null controls -------------------------------------------------------

def flat_grey(n=CANVAS):
    """No edges, no structure. Every property is undefined."""
    return np.full((n, n, 3), 128, np.uint8)


def white_noise(n=CANVAS, seed=0):
    """Maximal high-frequency energy, no structure at any scale."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (n, n, 3), dtype=np.uint8)


def smooth_noise(n=CANVAS, seed=0, sigma=8):
    """Band-limited noise: smooth gradients, no repeating or nested motifs."""
    rng = np.random.default_rng(seed)
    a = cv2.GaussianBlur(rng.normal(0, 1, (n, n)), (0, 0), sigma)
    a = (a - a.min()) / (np.ptp(a) + 1e-9)
    return np.dstack([(a * 255).astype(np.uint8)] * 3)


def random_blobs(n=CANVAS, k=200, seed=0):
    """Circles at uniformly random positions, radii and colours.

    The key discrimination control: no hierarchy, no repetition, no symmetry.
    A tool that scores this like a Persian carpet is not ranking wholeness.
    """
    rng = np.random.default_rng(seed)
    img = np.full((n, n, 3), 240, np.uint8)
    for _ in range(k):
        x, y = rng.integers(0, n, 2)
        col = tuple(int(v) for v in rng.integers(0, 200, 3))
        cv2.circle(img, (int(x), int(y)), int(rng.integers(5, 40)), col, -1)
    return img


def regular_grid(n=CANVAS, step=64):
    """A perfect lattice: zero roughness and maximal repetition, by definition."""
    return jittered_lattice(0.0, n=n, step=step)


# --- parametric generators -----------------------------------------------

def jittered_lattice(sigma, n=CANVAS, step=64, radius=16, seed=0):
    """Identical motifs on a lattice, displaced by Gaussian noise.

    ``sigma`` is the displacement standard deviation as a fraction of the
    lattice spacing, and is the ground truth for roughness: 0.0 is a perfect
    grid, ~0.5 approaches a random point process. A valid roughness measure
    must be monotone in ``sigma``.
    """
    rng = np.random.default_rng(seed)
    img = np.full((n, n, 3), _BG, np.uint8)
    for y in range(step, n - step + 1, step):
        for x in range(step, n - step + 1, step):
            dx, dy = rng.normal(0, sigma * step, 2)
            cv2.circle(img, (int(x + dx), int(y + dy)), radius, _FG, -1)
    return img


def nested_squares(ratio, n=CANVAS, depth=4, fill=0.5):
    """Recursive subdivision with a fixed parent:child scale ratio.

    ``ratio`` is the ground truth for levels of scale: each generation is
    ``ratio`` times smaller than its parent. A valid levels-of-scale measure
    should be extremal at whatever ratio it claims is ideal, and monotone
    on either side of it.
    """
    img = np.full((n, n, 3), _BG, np.uint8)

    def draw(cx, cy, r, d):
        if d == 0 or r < 3:
            return
        v = int(_BG - (_BG - 40) * (0.4 + 0.6 * d / depth))
        cv2.rectangle(img, (int(cx - r), int(cy - r)), (int(cx + r), int(cy + r)),
                      (v, v, v), -1)
        cv2.rectangle(img, (int(cx - r), int(cy - r)), (int(cx + r), int(cy + r)),
                      _FG, 2)
        child = r / ratio
        off = r * fill
        for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            draw(cx + sx * off, cy + sy * off, child, d - 1)

    draw(n / 2, n / 2, n / 2.6, depth)
    return img


def alternating_tiles(amplitude, n=CANVAS, step=64, radius=20):
    """Checkerboard of two motif sizes differing by ``amplitude``.

    ``amplitude`` is the ground truth for alternating repetition: 0.0 is a
    uniform grid of identical motifs (repetition without alternation), 1.0 is
    maximal size alternation between neighbours.
    """
    img = np.full((n, n, 3), _BG, np.uint8)
    for i, y in enumerate(range(step, n - step + 1, step)):
        for j, x in enumerate(range(step, n - step + 1, step)):
            r = radius * (1.0 - amplitude * 0.7 * ((i + j) % 2))
            cv2.circle(img, (x, y), max(int(r), 2), _FG, -1)
    return img


def contrast_field(delta, n=CANVAS, step=64, radius=20):
    """Lattice of motifs whose tonal separation from the ground is ``delta``.

    ``delta`` in [0, 1] is the ground truth for contrast. At 0 the motifs are
    invisible; at 1 they are black on near-white.
    """
    img = np.full((n, n, 3), _BG, np.uint8)
    v = int(round(_BG - delta * (_BG - 20)))
    for y in range(step, n - step + 1, step):
        for x in range(step, n - step + 1, step):
            cv2.circle(img, (x, y), radius, (v, v, v), -1)
    return img


def void_field(void_fraction, n=CANVAS, step=56, radius=16):
    """A lattice cleared of motifs within a central disc.

    ``void_fraction`` is the void radius as a fraction of the half-canvas,
    the ground truth for 'the void'.
    """
    img = np.full((n, n, 3), _BG, np.uint8)
    cx = cy = n / 2
    vr = void_fraction * n / 2
    for y in range(step, n - step + 1, step):
        for x in range(step, n - step + 1, step):
            if np.hypot(x - cx, y - cy) < vr:
                continue
            cv2.circle(img, (x, y), radius, _FG, -1)
    return img


NULL_CONTROLS = {
    "flat_grey": flat_grey,
    "white_noise": white_noise,
    "smooth_noise": smooth_noise,
    "random_blobs": random_blobs,
    "regular_grid": regular_grid,
}

# generator -> (callable, sweep values, the property it is built to isolate)
SWEEPS = {
    "jitter": (jittered_lattice, [0.0, 0.05, 0.10, 0.20, 0.30, 0.45], "roughness"),
    "scale_ratio": (nested_squares, [1.5, 2.0, 3.0, 4.0, 6.0], "levels_of_scale"),
    "alternation": (alternating_tiles, [0.0, 0.25, 0.5, 0.75, 1.0], "alternating_repetition"),
    "tonal_delta": (contrast_field, [0.1, 0.3, 0.5, 0.7, 1.0], "contrast"),
    "void_size": (void_field, [0.0, 0.15, 0.3, 0.45, 0.6], "the_void"),
}
