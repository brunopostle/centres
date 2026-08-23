"""Synthetic stimuli with structure known by construction.

Two kinds:

*Null controls* contain no coherent centre structure (or, for ``regular_grid``,
structure of a precisely known kind). A measure that reports high wholeness
for these is not measuring wholeness.

*Parametric generators* vary one structural property along a known scalar
while holding the rest fixed, so a measure can be checked against ground truth
rather than merely for plausibility. There is one per property, all fifteen,
registered in ``SWEEPS`` as::

    name: (callable, sweep values, property isolated, how to judge it)

The fourth field matters: monotonicity is the right test only for a measure
that has no interior ideal. ``levels_of_scale`` is a squared deviation from a
parent:child ratio of 3, so on a sweep that straddles 3 it should be *minimal*
in the middle and a high Spearman rho would be evidence against it. That entry
is marked ``optimum(3.0)`` and ``audit.run.sweeps`` reports where the minimum
actually falls, plus monotonicity on each side, instead of a single rho.

Twelve or thirteen sample points per sweep, not the five the first generators
carried: at n = 5 one adjacent rank swap costs 0.100 of Spearman's rho, which
makes the usual 0.9 threshold a test for perfection and every value below it
uninterpretable. At n = 12 one swap costs 0.007.

``tests/test_stimuli.py`` recovers every generator's parameter from the rendered
image by direct measurement, using nothing from ``centres/``. A ground truth
that can only be checked by running the pipeline it exists to test is not one.
"""

import cv2
import numpy as np

CANVAS = 1024
_BG = 245
_FG = (30, 30, 30)


# --- shape primitives ----------------------------------------------------
#
# Every generator below that varies a motif's *shape* holds the motif's *area*
# fixed, so that the swept quantity is the shape alone and not the amount of ink
# on the canvas. These helpers exist to make that constraint exact rather than
# approximate: polygons are built about the origin in arbitrary units and then
# scaled to a stated area before being drawn.

def _poly_area(pts):
    """Shoelace area of a closed polygon given as an (N, 2) array."""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _fit_area(pts, area):
    """Scale a polygon about the origin so its area is exactly ``area``."""
    return pts * np.sqrt(area / _poly_area(pts))


def _ngon(k, phase=0.0, r=1.0):
    t = np.arange(k) * 2 * np.pi / k + phase
    return np.stack([r * np.cos(t), r * np.sin(t)], 1)


def _star(k, inner, phase=0.0, r=1.0):
    """A 2k-gon alternating between radius ``r`` and radius ``inner * r``.

    Convex exactly when ``inner >= cos(pi / k)``; below that the edges cave in
    and the polygon becomes a k-pointed star.
    """
    t = np.arange(2 * k) * np.pi / k + phase
    rad = np.where(np.arange(2 * k) % 2 == 0, r, r * inner)
    return np.stack([rad * np.cos(t), rad * np.sin(t)], 1)


def _plus(arm=0.34):
    """A Greek cross whose arms are ``arm`` of the full width."""
    a, b = arm, 1.0
    return np.array([
        (-a, -b), (a, -b), (a, -a), (b, -a), (b, a), (a, a),
        (a, b), (-a, b), (-a, a), (-b, a), (-b, -a), (-a, -a),
    ], float)


def _rosette(m, base=0.45, samples=192):
    """A closed curve with rotational symmetry of order exactly ``m``.

    ``r(theta) = base + (1 - base) * (1 + cos(m * theta)) / 2``. The mean square
    of that profile does not depend on ``m``, so at fixed scale every member of
    the family encloses the same area — the symmetry order is varied without
    varying how much ink is on the canvas. See :func:`symmetric_motifs`.
    """
    t = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    rad = base + (1 - base) * (0.5 + 0.5 * np.cos(m * t))
    return np.stack([rad * np.cos(t), rad * np.sin(t)], 1)


def _radial(profile):
    """Close a radial profile sampled at equal angles into a polygon."""
    t = np.linspace(0, 2 * np.pi, len(profile), endpoint=False)
    return np.stack([profile * np.cos(t), profile * np.sin(t)], 1)


def _solidity(pts):
    hull = cv2.convexHull(pts.astype(np.float32).reshape(-1, 1, 2))
    return _poly_area(pts) / _poly_area(hull.reshape(-1, 2))


def _circularity(pts):
    """Isoperimetric ratio 4*pi*A / P**2: 1 for a circle, less for anything else."""
    p = float(np.hypot(*(np.roll(pts, -1, 0) - pts).T).sum())
    return 4 * np.pi * _poly_area(pts) / (p * p)


def _solve_amplitude(shape_at, target, hi, statistic, iters=48):
    """Bisect the deformation amplitude that gives ``statistic`` = ``target``.

    ``statistic`` (solidity or circularity) is 1 at zero amplitude and decreases
    monotonically with it, so a plain bisection is exact to machine tolerance and
    deterministic. This is what lets the generators below take a *named shape
    statistic* as their parameter rather than an arbitrary interpolation weight:
    the sweep value is the answer, not a proxy for it.
    """
    lo = 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if statistic(shape_at(mid)) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


#: Twelve visually distinct outlines, used as a vocabulary by the generators that
#: vary *how many kinds of thing* a composition contains (``shape_vocabulary``,
#: ``element_vocabulary``). Each is scaled to a common area before it is drawn,
#: so swapping one for another changes the shape and nothing else.
#:
#: Distinct means distinguishable from the rendered image, not merely different
#: in this list. Each outline's normalised radial signature has its energy at a
#: different angular harmonic, or at the same harmonic with a clearly different
#: amplitude (pentagon against five-pointed star), which is what
#: ``tests/test_stimuli.py`` clusters on. Near-circular polygons are deliberately
#: absent above five sides: a hexagon sits 0.030 from a circle in signature
#: space while the same shape drawn at 64 px and at 256 px differs by up to
#: 0.027, so a vocabulary containing both a circle and a hexagon cannot be
#: counted from the image at all. Dropping the hexagon lifts the closest pair to
#: 0.051 (circle against pentagon) and leaves a factor of two of margin. Counting
#: shapes the image cannot distinguish would be a claim the stimulus does not
#: support.
#: Every outline is also kept compact — none spans more than 1.76 times the
#: square root of its own area — so that at the fill fractions the vocabulary
#: generators use, no shape overruns its cell and merges with its neighbour. A
#: shape vocabulary whose members touch each other is not a vocabulary of that
#: many elements.
SHAPE_PALETTE = [
    _ngon(32),                             # circle
    _ngon(4, np.pi / 4),                   # square
    _ngon(3, np.pi / 2),                   # triangle
    _star(5, 0.50),                        # five-pointed star
    _plus(0.36),                           # cross
    _star(8, 0.62),                        # eight-pointed star
    _ngon(5, np.pi / 2),                   # pentagon
    _star(6, 0.58),                        # six-pointed star
    _rosette(9, base=0.58),                # nine-lobed gear
    np.array([(-1.15, -0.55), (1.15, -0.55),
              (0.45, 0.55), (-0.45, 0.55)]),  # trapezoid
    _ngon(4, np.pi / 4) * (1.55, 0.645),   # rectangle, 2.4:1
    _ngon(32) * (1.42, 0.704),             # ellipse, 2:1
]


def _fill(img, pts, cx, cy, colour=_FG):
    cv2.fillPoly(img, [np.round(pts + (cx, cy)).astype(np.int32)], colour)


def _lattice(n, step):
    """The lattice the original generators use: ``step`` apart, inset by ``step``."""
    return [(x, y)
            for y in range(step, n - step + 1, step)
            for x in range(step, n - step + 1, step)]


def _full_lattice(n, step):
    """Lattice sites covering the whole canvas, leaving no empty margin.

    The inset lattice above leaves a blank border between the outermost motifs
    and the frame — up to 144 px at ``step`` = 80 — and that border, not the
    artwork, then contains the deepest point of the distance transform. Because
    ``build_structural_field`` normalises by its own maximum and
    ``detect_centers`` thresholds the result at an absolute 0.08 (the coupling
    ``centres/field.py`` documents as its one known violation of the project
    invariant, #27), a blank margin sets the detection threshold for everything
    inside it: a rosette lattice detected 40 centres from 121 motifs because the
    motifs stood at 0.18 of a maximum owned by the corner of the frame.

    A margin is a property of the photograph and not of the thing photographed,
    which is exactly the class of quantity the project invariant forbids, so the
    generators added for #19 tile edge to edge and leave the frame out of it.
    """
    return [(x, y)
            for y in range(step // 2, n, step)
            for x in range(step // 2, n, step)]


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


#: Full figure/ground separation at ``delta`` = 1, in grey levels. The two tones
#: straddle mid-grey, so at every delta the *area-weighted mean* of the canvas is
#: 128 — see contrast_field.
_CONTRAST_SPAN = 220.0

#: Disc radius and lattice step for contrast_field, chosen together so the discs
#: cover about 42% of the canvas — near enough to half that both tones can sit
#: well inside the 8-bit range at delta = 1.
_CONTRAST_STEP = 64
_CONTRAST_RADIUS = 25


def contrast_field(delta, n=CANVAS, step=_CONTRAST_STEP, radius=_CONTRAST_RADIUS):
    """Lattice of motifs whose tonal separation from the ground is ``delta``.

    ``delta`` in [0, 1] is the ground truth for contrast: the figure and the
    ground are ``delta * 220`` grey levels apart, placed about mid-grey in
    proportion to how much of the canvas each covers, so the *mean* tone is 128
    for every delta and only the separation varies. The span stops at 220 rather
    than the full 255 because that is the widest separation the two tones can
    straddle mid-grey without one of them clipping, and a clipped tone would make
    the swept separation smaller than the parameter says it is.

    That symmetry is a repair, not decoration. The original construction put
    near-black discs covering 31% of a near-white ground, so the canvas mean was
    227-239 rather than 128. ``_flat_field`` divides the image by a blurred copy
    of itself, which rescales local contrast by 128/mean — roughly 0.54 here — so
    the contrast actually presented to the edge detector was little more than
    half the delta being swept. The 0.3 point fell just below Canny's absolute
    floor and detected 8 centres against 481 for the rest of the sweep; the
    detector changed regime in the middle of the experiment, which is exactly
    what a ground-truth sweep must not do. Centring the two tones on mid-grey
    makes the flat-field gain unity, so the swept parameter reaches the detector
    undivided.

    ``delta = 0.1`` still detects nothing, and correctly so. ``_EDGE_FLOOR`` in
    ``centres/field.py`` is an absolute 80 in Canny's |dx| + |dy| units — the
    smallest step the pipeline is willing to call a boundary at all. A sharp step
    of ``220 * delta`` levels, pre-blurred at sigma 2, produces about
    ``350 * delta``, so the pipeline's contrast floor sits at delta ~= 0.23. Below
    it the pipeline is asserting "there is no structure here", which is its
    designed behaviour and not a defect of the stimulus. The sweep keeps the 0.1
    point so that the floor stays visible in the audit table.
    """
    img = np.full((n, n, 3), _BG, np.uint8)
    sites = _lattice(n, step)
    fraction = len(sites) * np.pi * radius**2 / float(n * n)
    lo = int(round(128.0 - (1.0 - fraction) * _CONTRAST_SPAN * delta))
    hi = int(round(128.0 + fraction * _CONTRAST_SPAN * delta))
    img[:] = int(np.clip(hi, 0, 255))
    for x, y in sites:
        cv2.circle(img, (x, y), radius, (int(np.clip(lo, 0, 255)),) * 3, -1)
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


# --- generators for the remaining ten properties -------------------------
#
# Design constraints these all try to meet, in order of importance:
#
#   1. The parameter is the answer. Each is a quantity that can be recovered
#      from the rendered image by direct measurement — a component-area ratio, a
#      solidity, a ramp width — with no reference to the pipeline under test.
#      ``tests/test_stimuli.py`` recovers each one and checks it.
#   2. Vary one thing. Motif area, lattice pitch, motif count and canvas mean
#      tone are held fixed wherever the swept quantity does not force them to
#      move; where something else must move, the docstring says what and why.
#   3. Hold the gap population fixed. The structural field is a distance
#      transform and detects the gaps between motifs as readily as the motifs
#      themselves (#26), so a sweep that changes how many interstitial regions
#      there are is not isolating anything — it is varying the centre count.

#: Lattice pitch and motif radius for ``dominant_motif``, and the radius of the
#: clearing at the centre. The clearing is the same at every sweep point, so the
#: lattice population does not change as the dominant motif grows.
_DOM_STEP = 64
_DOM_RADIUS = 14
_DOM_CLEARING = 176


def dominant_motif(dominance, n=CANVAS, step=_DOM_STEP, radius=_DOM_RADIUS,
                   clearing=_DOM_CLEARING):
    """A field of equal motifs with one motif ``dominance`` times their radius.

    ``dominance`` is the ground truth for strong centres: the radius of the one
    distinguished motif as a multiple of the radius shared by all the others. At
    1.0 nothing is distinguished — the composition is a field of equals with a
    hole in it; at 5.4 a single centre is five times the linear size of anything
    else and unambiguously dominates.

    The isolation comes from the clearing. A dominant motif has to occupy space,
    and if it simply grew into the lattice it would delete a different number of
    its neighbours at each sweep point — the centre count would then track the
    parameter for a reason that has nothing to do with dominance. So the clearing
    is cut once, at a fixed radius large enough for the largest motif in the
    sweep, and every sweep point is drawn into the same hole in the same lattice.
    Between the first and last point the *only* difference in the image is the
    radius of one disc.
    """
    img = np.full((n, n, 3), _BG, np.uint8)
    cx = cy = n / 2.0
    for x, y in _full_lattice(n, step):
        if np.hypot(x - cx, y - cy) < clearing:
            continue
        cv2.circle(img, (x, y), radius, _FG, -1)
    cv2.circle(img, (int(cx), int(cy)), int(round(radius * dominance)), _FG, -1)
    return img


#: Outer radius and pitch for ``bordered_motifs``. The outer radius is what is
#: held fixed as the band thickens.
_BORDER_STEP = 112
_BORDER_RADIUS = 40


def bordered_motifs(band, n=CANVAS, step=_BORDER_STEP, radius=_BORDER_RADIUS):
    """Motifs of fixed outer radius ringed by a border band of thickness ``band``.

    ``band`` is the ground truth for boundaries: the thickness of the dark border
    band as a fraction of the motif's outer radius. Each motif is a dark disc of
    radius ``radius`` with a light core of radius ``radius * (1 - band)``, so at
    0.15 the motif is a thin ring and at 0.7 it is a heavy annulus around a small
    core.

    What is held fixed is the *outer* radius, which is what makes the
    interstitial ground identical at every sweep point: the gap between motif
    footprints is ``step - 2 * radius`` throughout, so the ground-polarity centre
    population does not move with the parameter. The one thing that necessarily
    does move is the light core, which shrinks from 0.85 to 0.30 of the radius — a
    border band has to be made of something, and the alternative (fixing the core
    and growing the band outward) would have consumed the interstitial ground
    instead. Fixing the core keeps *the centre* constant and lets its boundary
    thicken, which is the direction Alexander's property is stated in.

    A second thing moves that no lattice construction can prevent: the band
    thickness is itself a length, and ``edge_spacing`` estimates the artwork's
    characteristic length from exactly such features. Between the ends of this
    sweep the pipeline's own scale estimate roughly quadruples. Boundary
    thickness cannot be separated from feature scale, because it is one.
    """
    img = np.full((n, n, 3), _BG, np.uint8)
    inner = int(round(radius * (1.0 - band)))
    for x, y in _full_lattice(n, step):
        cv2.circle(img, (x, y), radius, _FG, -1)
        if inner > 0:
            cv2.circle(img, (x, y), inner, (_BG, _BG, _BG), -1)
    return img


#: Cell size for ``interstitial_shape``, the area of the light region inside each
#: cell as a fraction of the cell, and how many lobes its outline has. The area
#: is what is held fixed while the shape's convexity is swept.
_INTER_CELL = 128
_INTER_FILL = 0.26
_INTER_LOBES = 5
_INTER_SAMPLES = 192


def _inter_outline(amplitude, lobes=_INTER_LOBES, samples=_INTER_SAMPLES):
    t = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    return _radial(1.0 + amplitude * np.cos(lobes * t))


def interstitial_shape(solidity, n=CANVAS, cell=_INTER_CELL, fill=_INTER_FILL):
    """A dark web whose interstitial ground has exactly the given ``solidity``.

    ``solidity`` — the region's area divided by the area of its convex hull — is
    the ground truth for positive space: 1.0 is a well-formed convex courtyard,
    0.6 is a ragged leftover of the same area in the same place, with two fifths
    of its hull unoccupied.

    The ground is the closed curve ``r(theta) = 1 + a * cos(5 * theta)``, whose
    solidity falls monotonically from 1 as ``a`` rises from 0, so ``a`` is
    bisected for each sweep value and the parameter is the shape statistic
    itself rather than a proxy for it. The curve is then scaled so the enclosed
    area is identical at every sweep point.

    Smooth lobes rather than a star polygon, and this matters more than it
    looks. A polygonal star has sharp tips, and ``edge_spacing`` weights its
    medial-axis samples by ``d ** -1.5`` — deliberately, so that one large empty
    area cannot outvote many small ones. A handful of needle-sharp corners is
    read as the artwork's characteristic scale: the tipped version of this
    stimulus estimated the spacing at 2.3 px against 40 px for its own convex
    endpoint, which moved the distance cap by a factor of 17 and the centre count
    from 32 to 972 across a sweep that was supposed to hold everything but
    convexity fixed. Rounded lobes hold the spacing estimate steady.

    Isolation: the figure is a single connected dark web at every sweep point,
    and the ground is the same number of isolated regions, of the same area, in
    the same places. Only their convexity changes. Note that this is the same
    boundary that ``motif_regularity`` sweeps, read from the other side — there
    the isolated shapes are the figure and the ground is one connected web, here
    it is the reverse. Positive space and good shape are two readings of one
    boundary, and running both generators is the cheapest way to find out whether
    the two measures tell them apart.
    """
    a = _solve_amplitude(_inter_outline, float(solidity), 0.9, _solidity)
    pts = _fit_area(_inter_outline(a), fill * cell * cell)
    img = np.full((n, n, 3), _FG[0], np.uint8)
    for x, y in _full_lattice(n, cell):
        _fill(img, pts, x, y, (_BG, _BG, _BG))
    return img


#: Pitch and motif area for ``motif_regularity``, and the harmonics of the
#: deformation that takes its motif from a disc to a ragged blob.
_SHAPE_STEP = 112
_SHAPE_AREA = np.pi * 35.0**2
_SHAPE_HARMONICS = (2, 3, 4, 5)
_SHAPE_PHASES = (0.0, 1.9, 3.6, 5.1)
_SHAPE_WEIGHTS = (1.0, 0.8, 0.6, 0.45)
_SHAPE_SAMPLES = 192


def _motif_outline(amplitude, samples=_SHAPE_SAMPLES):
    t = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    p = sum(w * np.cos(h * t + ph)
            for h, ph, w in zip(_SHAPE_HARMONICS, _SHAPE_PHASES, _SHAPE_WEIGHTS))
    return _radial(1.0 + amplitude * p / np.abs(p).max())


def motif_regularity(circularity, n=CANVAS, step=_SHAPE_STEP, area=_SHAPE_AREA):
    """A lattice of identical motifs of exactly the given ``circularity``.

    ``circularity`` — the isoperimetric ratio ``4 * pi * A / P**2`` — is the
    ground truth for good shape: 1.0 is a disc, the shape with the least boundary
    for its area and the canonical well-formed figure; 0.65 is a blob of the same
    area whose outline is a quarter longer than it needs to be.

    The motif is a disc deformed by four fixed harmonics of its radial profile,
    with the deformation amplitude bisected until the outline's circularity is
    the sweep value exactly. Four harmonics rather than one so that the ragged
    end of the sweep is genuinely irregular rather than an n-lobed rosette, which
    would be a symmetry sweep wearing a different label; smooth rather than a
    vertex-pulled polygon so that sharp corners do not silently reset the
    pipeline's scale estimate (see :func:`interstitial_shape`).

    The lower end of the sweep is set by that family rather than chosen: four
    smooth harmonics with radii bounded below by zero cannot reach a circularity
    under about 0.60, and reaching lower would mean spikes.

    Isolation rests on two things. The outline is rescaled so its *area* is the
    same at every sweep point, so the ink on the canvas, its mean tone and the
    flat-field gain do not move. And every motif in a given image is the same
    shape, so the composition's repetition structure is untouched and the only
    thing that changes is the shape being repeated. The lattice, the motif count
    and the tone are fixed.
    """
    a = _solve_amplitude(_motif_outline, float(circularity), 0.9, _circularity)
    pts = _fit_area(_motif_outline(a), area)
    img = np.full((n, n, 3), _BG, np.uint8)
    for x, y in _full_lattice(n, step):
        _fill(img, pts, x, y)
    return img


#: Pitch and scale for ``symmetric_motifs``. The rosette radius is set so the
#: motif fills the same fraction of its cell as the other lattice generators.
_SYM_STEP = 144
_SYM_RADIUS = 56.0


def symmetric_motifs(order, n=CANVAS, step=_SYM_STEP, radius=_SYM_RADIUS):
    """A lattice of rosettes with rotational symmetry of order exactly ``order``.

    ``order`` is the ground truth for local symmetries: the order of the motif's
    rotational symmetry group, m = 1 to 12. Each motif is the closed
    curve ``r(theta) = R * (base + (1 - base) * (1 + cos(m * theta)) / 2)``, whose
    symmetry group is C_m for every m — an egg at m = 1, a peanut at 2, a rosette
    with m lobes above that.

    Area is constant by construction rather than by rescaling: the mean square of
    that radial profile does not depend on m, so at fixed R every member of the
    family encloses the same area. Position, pitch, count and tone are fixed too.

    **The confound that cannot be removed, and it is worth stating plainly.**
    A motif of bounded size with symmetry of order m necessarily has angular
    features no wider than ``2*pi/m``: raising the symmetry order *is* refining
    the feature scale, as a matter of geometry rather than of this construction.
    The lobes here are 360/m degrees wide, so between m = 1 and m = 12 the
    motif's internal feature scale falls twelvefold and its perimeter rises. The
    only way to hold feature scale fixed is to grow the motif in proportion to m,
    which trades the confound for a worse one (motif size, which is levels of
    scale and strong centres). Local symmetry order is therefore *not*
    independently isolable at fixed motif size, and any rho measured on this
    sweep is shared with good shape and levels of scale.
    """
    img = np.full((n, n, 3), _BG, np.uint8)
    pts = _rosette(int(order)) * radius
    for x, y in _full_lattice(n, step):
        _fill(img, pts, x, y)
    return img


#: Band height and tooth period for ``interdigitated_bands``.
_INTERLOCK_BAND = 128
_INTERLOCK_PERIOD = 128


def interdigitated_bands(depth, n=CANVAS, band=_INTERLOCK_BAND,
                         period=_INTERLOCK_PERIOD):
    """Alternating bands whose shared boundaries interdigitate to ``depth``.

    ``depth`` is the ground truth for deep interlock: the amplitude of the
    square-wave boundary between two adjacent regions, as a fraction of the band
    half-height. At 0 the bands are straight stripes that merely touch; at 0.8
    each band's fingers reach four fifths of the way into its neighbour's
    territory and the two regions are woven together.

    Successive boundaries carry the same square wave in antiphase, so a band is
    thick where its neighbours are thin. That is what makes this interlock rather
    than corrugation: giving every boundary the same phase would translate the
    bands without interleaving them at all. Because the wave is symmetric about
    the nominal boundary, the mean band thickness — and therefore the area of
    each region and the mean tone of the canvas — is independent of ``depth``.
    The number of bands and the tooth period are fixed; only the amplitude moves.

    The sweep stops at 0.8 because at 1.0 the thin sections pinch to zero width
    and the bands stop being two regions at all.
    """
    a = depth * band / 2.0
    x = np.arange(n)
    half = max(period // 2, 1)
    Y = np.arange(n)[:, None]
    wave = np.where((x // half) % 2 == 0, 1.0, -1.0)
    mask = np.zeros((n, n), bool)
    # Offset by half a band so that the frame cuts through a light region rather
    # than through a toothed boundary. Without it the top and bottom boundaries
    # are clipped asymmetrically and the ink on the canvas drifts by 3% across
    # the sweep — a confound introduced by the frame, exactly the class of thing
    # these generators exist to keep out.
    for i in range(1, int(n // band), 2):
        top = (i - 0.5) * band + a * wave
        bot = (i + 0.5) * band - a * wave
        mask |= (Y >= top[None, :]) & (Y < bot[None, :])
    img = np.full((n, n, 3), _BG, np.uint8)
    img[mask] = _FG
    return img


#: Band heights for ``shape_vocabulary``, in order down the canvas. Each band is
#: a row of motifs on a square grid of that pitch, so the bands tile the frame
#: and three scales are interleaved. Fixed for every sweep point — only which
#: shape lands in each slot changes.
_ECHO_BANDS = (
    256, 128, 64, 256, 128, 64, 128,
)

#: Motif area within a band cell, as a fraction of the cell.
_ECHO_FILL = 0.22


def shape_vocabulary(distinct, n=CANVAS, bands=_ECHO_BANDS, fill=_ECHO_FILL):
    """Motifs at three scales drawn from a vocabulary of ``distinct`` shapes.

    ``distinct`` is the ground truth for echoes: how many different outlines the
    composition uses across its three scale bands. At 1 every motif in the image
    is the same shape at three sizes — the family resemblance is total, which is
    what an echo is. At 12 no shape has a relative and the resemblance is gone.

    The canvas is divided into horizontal bands of height 256, 128 or 64 px, each
    holding a row of motifs on a square grid of that pitch, so the three scales
    are interleaved down the image and the bands tile the frame exactly. Motif
    area is 22% of its cell at every scale.

    Isolation is as complete as this project gets. The slot layout — positions,
    counts and areas per band — is identical at every sweep point, because each
    palette shape is rescaled to its band's area before it is drawn. The only
    difference between the first image and the last is which outline sits in
    which slot. A centre count therefore cannot track the parameter through the
    geometry, which is the usual way a sweep fools itself.

    ``element_vocabulary`` is the flat-composition sibling of this generator: the
    same question asked of one scale instead of three. The pair is deliberate —
    #19 sketches echoes and simplicity in nearly the same words, and running both
    is how you find out whether the two measures tell them apart.
    """
    k = max(int(distinct), 1)
    img = np.full((n, n, 3), _BG, np.uint8)
    y0 = 0
    for level, cell in enumerate(bands):
        for j in range(n // cell):
            pts = _fit_area(SHAPE_PALETTE[(level + j) % k], fill * cell * cell)
            _fill(img, pts, (j + 0.5) * cell, y0 + 0.5 * cell)
        y0 += cell
    return img


#: Pitch and motif area for ``element_vocabulary``.
_VOCAB_STEP = 96
_VOCAB_AREA = 0.22 * _VOCAB_STEP**2


def element_vocabulary(kinds, n=CANVAS, step=_VOCAB_STEP, area=_VOCAB_AREA):
    """One flat lattice built from ``kinds`` distinct elements.

    ``kinds`` is the ground truth for simplicity: the size of the vocabulary a
    viewer would need in order to describe the composition. At 1 the whole canvas
    is one element repeated 121 times — simplicity and inner calm in the literal
    sense. At 12 the same 121 slots hold twelve different things in a fixed
    interleaved order, and the composition has to be described twelve ways.

    Everything except identity is held fixed: one lattice, one motif count, one
    scale, one tone, and every palette shape rescaled to the same area before it
    is drawn, so the ink on the canvas is unchanged by the parameter.

    Deliberately *not* varied: the number of elements drawn. Sweeping that at
    constant ink would shrink the elements as it multiplied them, which is a
    uniform change of scale — a self-similar zoom rather than a change in
    complexity — and would confound simplicity with levels of scale for no gain.
    """
    k = max(int(kinds), 1)
    img = np.full((n, n, 3), _BG, np.uint8)
    for x, y in _full_lattice(n, step):
        # Indexed by row *plus* column rather than by position in the flat list.
        # A flat index aliases against the row length — at 11 columns and 11
        # kinds, every row repeats the same sequence, so two of the kinds only
        # ever appear in the edge columns. The diagonal index puts every kind in
        # every row.
        which = ((x // step) + (y // step)) % k
        _fill(img, _fit_area(SHAPE_PALETTE[which], area), x, y)
    return img


#: Pitch and radius for ``bleeding_motifs``. The pitch is wide enough that
#: neighbouring ramps never meet, even at the widest bleed in the sweep.
_BLEED_STEP = 112
_BLEED_RADIUS = 34
_BLEED_INK = 15
_BLEED_GROUND = 240


def bleeding_motifs(bleed, n=CANVAS, step=_BLEED_STEP, radius=_BLEED_RADIUS):
    """A lattice of motifs whose edges dissolve into the ground over ``bleed``.

    ``bleed`` is the ground truth for not-separateness: the width of the
    figure/ground transition as a fraction of the motif radius. At 0 the motifs
    are hard-edged discs, as separate from their surround as a shape can be; at
    0.6 each motif fades into the ground over a band 20 pixels wide and there is
    no line at which the motif stops.

    The transition is an explicit linear ramp in tone between ``radius - w/2``
    and ``radius + w/2`` rather than a blur, for two reasons: the ramp width is
    then exactly the parameter and directly measurable from the rendered image,
    and the maximum gradient it presents is exactly ``contrast / w``, which can
    be checked against the pipeline's edge floor instead of discovered by
    collapse. The motif's half-tone contour stays at ``radius``, so the ink on
    the canvas, the motif positions and the interstitial ground are all
    independent of the parameter.

    **A ceiling worth recording.** ``_EDGE_FLOOR`` = 80 in Canny's |dx| + |dy|
    units means the pipeline cannot see a full-contrast transition wider than
    about ``8 * 225 / 80`` = 22 pixels *at all*: past that the figure and the
    ground stop being separated by anything the detector will call a boundary,
    and the centre count collapses rather than degrading. The sweep is bounded
    below that ceiling on purpose. Not-separateness beyond it is not measured
    poorly, it is invisible.
    """
    w = max(float(bleed) * radius, 1e-6)
    img = np.full((n, n), float(_BLEED_GROUND))
    reach = int(np.ceil(radius + w / 2.0)) + 1
    for x, y in _full_lattice(n, step):
        x0, x1 = max(x - reach, 0), min(x + reach + 1, n)
        y0, y1 = max(y - reach, 0), min(y + reach + 1, n)
        Y, X = np.mgrid[y0:y1, x0:x1]
        t = np.clip((np.hypot(X - x, Y - y) - (radius - w / 2.0)) / w, 0.0, 1.0)
        img[y0:y1, x0:x1] = np.minimum(
            img[y0:y1, x0:x1], _BLEED_INK + (_BLEED_GROUND - _BLEED_INK) * t)
    return np.dstack([np.round(img).astype(np.uint8)] * 3)


#: Pitch, radius and the two zone tones for ``zone_transition``. Both tones are
#: well clear of the ground, so every motif is visible at every position.
_ZONE_STEP = 96
_ZONE_RADIUS = 30
_ZONE_DARK = 20
_ZONE_LIGHT = 100


def zone_transition(width, n=CANVAS, step=_ZONE_STEP, radius=_ZONE_RADIUS):
    """One lattice, two tonal zones, and a transition ``width`` between them.

    ``width`` is the ground truth for gradients: the width of the band over which
    the composition changes from its dark zone to its light zone, as a fraction
    of the canvas. At 0 the two zones abut along a hard seam; at 1.0 the change is
    spread over the whole canvas and there are no zones left, only a gradient.

    The ramp is linear in motif tone with x, so the transition width recovered
    from the rendered image is exactly ``width * n``. The geometry is completely
    untouched by the parameter: the same lattice, the same radii, the same count,
    the same ground. The only thing that changes is which grey each motif is
    painted, and the ramp is antisymmetric about the canvas centre so the mean
    tone is fixed as well.

    Both zone tones are dark against a light ground — 20 and 100 against 245 —
    rather than one zone fading into the background. A zone that fades out would
    delete its own motifs at one end of the sweep and turn a gradient measurement
    into a centre-count measurement.

    The light zone's tone is 100 rather than the 120 first tried, and the reason
    is instructive. Canny's upper hysteresis threshold is the 92nd percentile of
    the gradient magnitude over the *whole* frame, so the zone with the stronger
    contrast sets the threshold that the weaker zone then has to clear. At a tone
    of 120 the ratio was steep enough that at one sweep point — width 0.5, and
    only there — the weak zone's edges failed hysteresis, half the frame lost its
    edges entirely, and the count fell from 1198 centres to 12. A global
    percentile makes a two-zone image a single population; the cure is to keep
    the two zones' contrasts within about a factor of 1.5 of each other. This is
    worth knowing about the pipeline, and it is not the only stimulus it will
    bite.

    Note that ``_flat_field`` divides out a Gaussian illumination estimate at
    sigma = 0.05 of the short side, about 51 px here, which is itself a
    transition-width filter: a zone boundary much wider than that is partly
    removed before any measurement happens. That is a property of the pipeline
    under test, not of the stimulus, and it is one of the things this sweep is
    for.
    """
    img = np.full((n, n, 3), _BG, np.uint8)
    half = max(float(width) * n, 1e-6) / 2.0
    for x, y in _full_lattice(n, step):
        t = float(np.clip((x - (n / 2.0 - half)) / (2.0 * half), 0.0, 1.0))
        v = int(round(_ZONE_DARK + (_ZONE_LIGHT - _ZONE_DARK) * t))
        cv2.circle(img, (x, y), radius, (v, v, v), -1)
    return img



def bilateral_asymmetry(amount, n=CANVAS, step=72, radius=30):
    """Motifs sheared away from bilateral symmetry about the vertical axis.

    ``amount`` is the ground truth for *local symmetries*: 0 is a motif mirror-
    symmetric about its own vertical axis, 1 is the same motif sheared so that
    the two halves no longer correspond.

    This exists because ``symmetry_order`` cannot test the property. The source
    is specific that the axis is the vertical one — "bilateral symmetry about the
    vertical axis respects gravitational stability" — and a regular m-gon is
    bilaterally symmetric at *every* rotational order, measured at 0.986 to 0.999
    for m = 3 to 12 at any phase. Sweeping rotational order therefore holds the
    sourced quantity almost constant while varying something else.

    The shear is horizontal and increases with height, which destroys the mirror
    correspondence about a vertical axis while leaving area, convexity and the
    number of motifs unchanged.

    Motifs are packed close to touching. At a wider spacing the detector finds
    only the interstitial ground -- every centre came back with negative polarity
    and the figure population was empty -- because a sparse lattice of small
    motifs makes the gaps the larger local maxima of the distance field.

    The sweep stops at 0.6. Beyond that the sheared motifs begin to merge into
    their neighbours, the figure population collapses (367 centres to 53) and the
    measure becomes undefined -- so the stimulus stops isolating the quantity it
    was built to vary, and points past that would be measuring the merge.
    """
    img = np.full((n, n, 3), _BG, np.uint8)
    base = np.array(
        [[np.cos(a) * radius, np.sin(a) * radius]
         for a in np.linspace(-np.pi / 2, 3 * np.pi / 2, 7, endpoint=False)]
    )
    for y in range(step, n - step + 1, step):
        for x in range(step, n - step + 1, step):
            pts = base.copy()
            pts[:, 0] += amount * pts[:, 1]          # shear x by height
            poly = (pts + [x, y]).astype(np.int32)
            cv2.fillPoly(img, [poly], _FG)
    return img


NULL_CONTROLS = {
    "flat_grey": flat_grey,
    "white_noise": white_noise,
    "smooth_noise": smooth_noise,
    "random_blobs": random_blobs,
    "regular_grid": regular_grid,
}


def _span(lo, hi, k=12, digits=3):
    """``k`` evenly spaced sweep values from ``lo`` to ``hi`` inclusive."""
    return [round(float(v), digits) for v in np.linspace(lo, hi, k)]


def _geom(mid, factor, half, digits=3):
    """Sweep values geometric about ``mid``, ``half`` steps each side.

    For a parameter that is a *ratio*, equal steps in the ratio are not equal
    steps in anything the measures care about — ``levels_of_scale`` is a squared
    deviation in log-ratio — so a sweep meant to straddle an optimum has to be
    geometric if the two sides are to be comparable at all.
    """
    return [round(float(mid * factor**k), digits)
            for k in range(-half, half + 1)]


#: How a sweep is to be judged, and — for monotone measures — which way it should
#: move. ``MONOTONE`` means the measure should *rise* with the parameter and
#: ``MONOTONE_DOWN`` that it should *fall*: some properties are the inverse of the
#: quantity swept (more distinct shapes means *less* echo, more shear means *less*
#: symmetry), and a measure that correctly runs the other way is tracking its
#: ground truth, not failing it. The second field is the expected sign, so the
#: verdict can tell "runs backwards" from "tracks in the right direction" instead
#: of assuming every measure should increase.
#:
#: ``optimum(x)`` means the measure claims an *interior* ideal at parameter value
#: ``x`` and should be extremal there, so a high rho over the whole sweep would be
#: evidence *against* it. ``peak`` says which extremum: True when the raw measure
#: is *maximal* at the ideal — which every current optimum measure is, since the
#: #22 redefinitions made them ``exp(-deviation)``, one at the ideal — and False
#: for a raw squared deviation that is *minimal* there. ``audit.run.sweeps`` looks
#: for the right extremum accordingly.
MONOTONE = ("monotone", +1)
MONOTONE_DOWN = ("monotone", -1)


def optimum(at, peak=True):
    return ("optimum", (at, peak))


#: Twelve sample points per sweep, which is a deliberate change from the five or
#: six the first five generators carried.
#:
#: At n = 5 Spearman's rho takes only 21 distinct values and a *single* adjacent
#: rank swap costs 0.100, so the conventional 0.9 acceptance threshold is a test
#: for perfection and nothing else, and any rho below it is uninterpretable —
#: three swaps and total independence score alike. At n = 12 one swap costs
#: 0.007, which is the difference between a statistic that measures the measure
#: and one that measures the sampling. The concrete case: the dihedral
#: symmetrisation of the edge detector improved precision across the whole audit
#: (triage 7 usable / 6 marginal / 2 noise -> 11 / 3 / 1) while `scale_ratio`
#: moved +0.800 -> +0.500 and `alternation` -0.300 -> -0.100. Those are three and
#: two rank swaps on five points, and nothing in the numbers could say whether
#: either was real.
#:
#: The cost is runtime: 15 sweeps x 12 points is 180 stimuli, and the audit now
#: scores each one once and shares the result between the generator-count stage
#: and the sweep stage rather than scoring it twice.

# generator -> (callable, sweep values, property isolated, how to judge it)
#
# One entry per property, all fifteen. The ordering follows ``audit.run.KEYS``.
SWEEPS = {
    # E_H is (log(r_parent / r_child) - log 3)**2, so this measure claims its
    # ideal at a scale ratio of exactly 3 and the sweep spans and straddles it.
    "scale_ratio": (nested_squares, _geom(3.0, 2 ** (1 / 6), 6), "levels_of_scale", optimum(3.0)),
    "dominance": (dominant_motif, _span(1.0, 5.4), "strong_centres", MONOTONE),
    # An interior optimum, not monotone: the score peaks where the band is 1/3
    # of what it bounds, which measurement puts at param ~0.3, and falls away
    # symmetrically on both sides (0.47 -> 0.96 -> 0.23 across the sweep). A high
    # monotone rho would be evidence *against* the measure, exactly as for
    # scale_ratio.
    "border_band": (bordered_motifs, _span(0.15, 0.7), "boundaries", optimum(0.3)),
    "alternation": (alternating_tiles, _span(0.0, 1.0), "alternating_repetition", MONOTONE),
    "ground_solidity": (interstitial_shape, _span(0.6, 1.0), "positive_space", MONOTONE),
    "motif_circularity": (motif_regularity, _span(0.65, 1.0), "good_shape", MONOTONE),
    # Bilateral symmetry about the vertical axis, which is what the source names.
    # symmetric_motifs sweeps *rotational* order instead, and a regular m-gon is
    # bilaterally symmetric at every order (0.986 to 0.999 for m = 3..12, at any
    # phase), so that sweep holds this property almost constant while varying
    # something else. Kept below, unregistered, as the record of a stimulus that
    # could not test what it was built for.
    "bilateral_asymmetry": (bilateral_asymmetry, _span(0.0, 0.6), "local_symmetries", MONOTONE_DOWN),
    "interlock_depth": (interdigitated_bands, _span(0.0, 0.8), "deep_interlock", MONOTONE),
    # 0.1 is below the pipeline's absolute edge floor and detects nothing; it is
    # kept as the one sub-floor probe, and the rest of the sweep starts above it.
    "tonal_delta": (contrast_field, [0.1] + _span(0.3, 1.0, 11), "contrast", MONOTONE),
    "zone_width": (zone_transition, _span(0.0, 1.0), "gradients", MONOTONE),
    # The raw measure is the coefficient of variation of nearest-neighbour
    # spacing, which rises monotonically with disorder. The interior ideal at
    # CV = 0.5 that ``normalize_all`` applies lives in the *normaliser*, in score
    # space, not in this parameter, so the raw sweep is a monotonicity test.
    # ``audit.run.sweeps`` prints where the normalised score peaks regardless.
    "jitter": (jittered_lattice, _span(0.0, 0.45), "roughness", MONOTONE),
    "shape_vocabulary": (shape_vocabulary, list(range(1, 13)), "echoes", MONOTONE_DOWN),
    "void_size": (void_field, _span(0.0, 0.6), "the_void", MONOTONE),
    "element_kinds": (element_vocabulary, list(range(1, 13)), "simplicity", MONOTONE),
    "bleed": (bleeding_motifs, _span(0.0, 0.6), "not_separateness", MONOTONE),
}
