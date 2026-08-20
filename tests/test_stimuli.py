"""Do the ground-truth generators vary what they claim to vary? (#19)

A parametric generator is only worth anything if its parameter really is the
answer. These tests recover each generator's parameter from the rendered image
by direct measurement — connected components, contour solidity, tone profiles —
using nothing from ``centres/``. If a generator's claim can only be checked by
running the pipeline it exists to test, it is not a ground truth.

Each test also states what the generator holds *fixed*, because that is the half
of the claim that is easy to get wrong: a sweep that quietly changes the motif
count or the mean tone alongside its parameter is measuring the confound.
"""

import cv2
import numpy as np
import pytest
from scipy.stats import spearmanr

from audit import stimuli
from audit.run import KEYS


# --- helpers -------------------------------------------------------------

def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _ink(img):
    """Fraction of the canvas darker than mid-grey."""
    return float((_gray(img) < 128).mean())


def _components(img, dark=True):
    """Connected components of the dark (or light) part, largest first."""
    g = _gray(img)
    mask = (g < 128) if dark else (g >= 128)
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    areas = sorted(stats[1:, cv2.CC_STAT_AREA], reverse=True)
    return [float(a) for a in areas]


def _contours(img, dark=True):
    """Whole outlines of the dark (or light) regions.

    Regions touching the frame are dropped: they are cut off by it, so their
    shape is the frame's rather than the generator's, and counting them would
    make a shape-vocabulary count report the number of ways a motif can be
    clipped.
    """
    g = _gray(img)
    h, w = g.shape
    mask = ((g < 128) if dark else (g >= 128)).astype(np.uint8)
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        if cv2.contourArea(c) <= 50:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        if x == 0 or y == 0 or x + cw >= w or y + ch >= h:
            continue
        out.append(c)
    return out


def _monotone(param, measured):
    """Spearman rho of a recovered quantity against the parameter."""
    return float(spearmanr(param, measured).statistic)


# --- the registry itself -------------------------------------------------

def test_every_property_has_exactly_one_generator():
    targets = [t for _, _, t, _ in stimuli.SWEEPS.values()]
    assert sorted(targets) == sorted(KEYS)
    assert len(set(targets)) == 15


@pytest.mark.parametrize("name", sorted(stimuli.SWEEPS))
def test_generators_are_deterministic(name):
    """A sweep that moves between runs cannot support a rank correlation."""
    fn, values, _, _ = stimuli.SWEEPS[name]
    v = values[len(values) // 2]
    assert np.array_equal(fn(v), fn(v))


@pytest.mark.parametrize("name", sorted(stimuli.SWEEPS))
def test_sweeps_have_enough_points_for_a_rank_test(name):
    """Twelve points, because at five a single adjacent rank swap costs 0.100.

    With the conventional 0.9 threshold that makes rho a test for perfection and
    makes every failing value uninterpretable — three swaps and no relationship
    at all score alike. At twelve points one swap costs 0.007.
    """
    _, values, _, _ = stimuli.SWEEPS[name]
    assert len(values) >= 10
    assert len(set(values)) == len(values)
    assert list(values) == sorted(values)


@pytest.mark.parametrize("name", sorted(stimuli.SWEEPS))
def test_every_sweep_declares_how_it_is_to_be_judged(name):
    """Monotonicity is the wrong test for a measure with an interior ideal, and a
    measure that runs the opposite way is not a failure. Each sweep therefore
    declares its kind and, for a monotone sweep, the direction it should move; for
    an optimum sweep, the ideal parameter and which extremum marks it."""
    _, _, _, (kind, arg) = stimuli.SWEEPS[name]
    assert kind in ("monotone", "optimum")
    if kind == "monotone":
        assert arg in (+1, -1)                 # the direction it should move
    else:
        at, peak = arg                         # interior ideal, and peak vs valley
        assert isinstance(at, float) and isinstance(peak, bool)


def test_levels_of_scale_is_judged_on_where_its_optimum_falls():
    """``levels_of_scale`` peaks (after #22, ``exp(-deviation)``) at a scale ratio
    in the 2–5 band, so a high Spearman rho over a sweep spanning 3 would be
    evidence against it — the right test is where its maximum falls."""
    _, values, target, (kind, (at, peak)) = stimuli.SWEEPS["scale_ratio"]
    assert (target, kind, at, peak) == ("levels_of_scale", "optimum", 3.0, True)
    assert min(values) < at < max(values)


@pytest.mark.parametrize("name", sorted(stimuli.SWEEPS))
def test_every_sweep_point_renders_a_canvas_with_structure(name):
    """No sweep point may be a blank canvas — except where that is the answer.

    ``tonal_delta`` = 0.1 is a real zero: the figure and the ground are 25 grey
    levels apart, which is below the pipeline's declared edge floor. It still
    renders two tones, so it is not blank here; it is blank downstream, and that
    is the finding.
    """
    fn, values, _, _ = stimuli.SWEEPS[name]
    for v in values:
        img = fn(v)
        assert img.shape == (stimuli.CANVAS, stimuli.CANVAS, 3)
        assert img.dtype == np.uint8
        assert np.ptp(_gray(img)) > 5, f"{name}={v} is featureless"


# --- strong centres: dominance ------------------------------------------

def test_dominance_is_recoverable_as_an_area_ratio():
    """The distinguished motif's radius, relative to the field of equals."""
    _, values, _, _ = stimuli.SWEEPS["dominance"]
    got = []
    for v in values:
        areas = _components(stimuli.dominant_motif(v))
        biggest, rest = areas[0], np.median(areas[1:])
        got.append(np.sqrt(biggest / rest))
    assert _monotone(values, got) == 1.0
    # the recovered radius ratio is the parameter, to within rasterisation
    assert np.allclose(got, values, rtol=0.05)


def test_dominance_leaves_the_field_of_equals_untouched():
    """The clearing is cut once, so the same lattice motifs are present
    at every sweep point — only one disc differs between images."""
    _, values, _, _ = stimuli.SWEEPS["dominance"]
    counts = [len(_components(stimuli.dominant_motif(v))) for v in values]
    assert len(set(counts)) == 1


# --- boundaries: border band --------------------------------------------

def test_border_band_thickness_is_recoverable():
    """Band thickness as a fraction of the motif's outer radius."""
    _, values, _, _ = stimuli.SWEEPS["border_band"]
    r, step = stimuli._BORDER_RADIUS, stimuli._BORDER_STEP
    got = []
    for v in values:
        g = _gray(stimuli.bordered_motifs(v))
        # one motif, sampled along a radius from its centre
        cx = cy = step // 2
        row = g[cy, cx:cx + r + 1]
        got.append(float((row < 128).sum()) / r)
    assert _monotone(values, got) == 1.0
    assert np.allclose(got, values, atol=0.05)


def test_border_band_holds_the_interstitial_ground_fixed():
    """The outer radius is what is held constant, so the gap between motif
    footprints — and hence the ground-polarity population — does not move."""
    _, values, _, _ = stimuli.SWEEPS["border_band"]
    outer = []
    for v in values:
        biggest = max(_contours(stimuli.bordered_motifs(v)),
                      key=lambda c: cv2.contourArea(c))
        (_, _), rad = cv2.minEnclosingCircle(biggest)
        outer.append(rad)
    assert np.ptp(outer) <= 1.0


# --- positive space: interstitial solidity -------------------------------

def test_ground_solidity_is_solved_for_exactly():
    """The bisection hits the requested solidity on the outline itself."""
    _, values, _, _ = stimuli.SWEEPS["ground_solidity"]
    for v in values:
        a = stimuli._solve_amplitude(
            stimuli._inter_outline, v, 0.9, stimuli._solidity)
        assert abs(stimuli._solidity(stimuli._inter_outline(a)) - v) < 1e-3


def test_ground_solidity_is_recoverable_from_the_light_regions():
    """Solidity = area / convex-hull area of the interstitial ground."""
    _, values, _, _ = stimuli.SWEEPS["ground_solidity"]
    got = []
    for v in values:
        cs = _contours(stimuli.interstitial_shape(v), dark=False)
        sol = [cv2.contourArea(c) / cv2.contourArea(cv2.convexHull(c)) for c in cs]
        got.append(float(np.median(sol)))
    assert _monotone(values, got) == 1.0
    assert np.allclose(got, values, atol=0.04)


def test_ground_solidity_holds_the_ground_area_and_count_fixed():
    """Only convexity is swept: the same number of interstitial regions,
    each of the same area, in the same places."""
    _, values, _, _ = stimuli.SWEEPS["ground_solidity"]
    counts, areas = [], []
    for v in values:
        cs = _contours(stimuli.interstitial_shape(v), dark=False)
        counts.append(len(cs))
        areas.append(float(np.median([cv2.contourArea(c) for c in cs])))
    assert len(set(counts)) == 1
    assert np.ptp(areas) / np.mean(areas) < 0.03


# --- good shape: motif circularity ---------------------------------------

def test_motif_circularity_is_solved_for_exactly():
    """The bisection hits the requested circularity on the outline itself."""
    _, values, _, _ = stimuli.SWEEPS["motif_circularity"]
    for v in values:
        a = stimuli._solve_amplitude(
            stimuli._motif_outline, v, 0.9, stimuli._circularity)
        assert abs(stimuli._circularity(stimuli._motif_outline(a)) - v) < 1e-3


def test_motif_circularity_is_recoverable_from_the_rendered_motifs():
    """Isoperimetric ratio 4*pi*A/P**2, measured on the rasterised outline.

    Rasterisation adds staircase length to the perimeter, so the recovered value
    sits below the parameter; what has to hold is that the ordering survives.
    """
    _, values, _, _ = stimuli.SWEEPS["motif_circularity"]
    got = []
    for v in values:
        c = max(_contours(stimuli.motif_regularity(v)), key=cv2.contourArea)
        a, p = cv2.contourArea(c), cv2.arcLength(c, True)
        got.append(4 * np.pi * a / (p * p))
    assert _monotone(values, got) == 1.0
    assert got[-1] > got[0] + 0.2


def test_motif_circularity_holds_motif_area_and_count_fixed():
    _, values, _, _ = stimuli.SWEEPS["motif_circularity"]
    inks = [_ink(stimuli.motif_regularity(v)) for v in values]
    counts = [len(_components(stimuli.motif_regularity(v))) for v in values]
    assert len(set(counts)) == 1
    assert np.ptp(inks) / np.mean(inks) < 0.03


# --- local symmetries: rotational order ----------------------------------

_SYM_SAMPLES = 720


def _radial_profile(img, cx, cy, reach, samples=_SYM_SAMPLES):
    mask = _gray(img) < 128
    theta = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    d = np.arange(1, reach)
    rad = np.zeros(samples)
    for i, t in enumerate(theta):
        xs = np.round(cx + d * np.cos(t)).astype(int)
        ys = np.round(cy + d * np.sin(t)).astype(int)
        inside = mask[ys, xs]
        rad[i] = d[inside].max() if inside.any() else 0.0
    return rad


def _rotational_order(rad, candidates=range(2, 13), tol=0.97):
    """Largest m for which rotating by 2*pi/m maps the outline onto itself.

    Tested as self-coincidence rather than as harmonic content, because harmonic
    content is not a reliable reading of symmetry *order*: a shape whose group is
    trivial can still have all its energy at one harmonic, and re-centring on the
    centroid moves the first harmonic around. Self-coincidence is what C_m
    actually asserts, and it gives 1 for a shape with no symmetry at all.
    """
    theta = np.linspace(0, 2 * np.pi, len(rad), endpoint=False)
    best = 1
    for m in candidates:
        turned = np.interp(theta + 2 * np.pi / m, theta, rad, period=2 * np.pi)
        if np.corrcoef(rad, turned)[0, 1] > tol:
            best = max(best, m)
    return best


def test_bilateral_asymmetry_reduces_vertical_mirror_correspondence():
    """The generator that replaced symmetry_order for local symmetries.

    A regular m-gon is bilaterally symmetric at *every* rotational order, so
    sweeping order held the sourced property almost constant. This sweeps it
    directly, and the shear must actually reduce it.
    """
    from centres.regions import _describe

    fn, values, target, _ = stimuli.SWEEPS["bilateral_asymmetry"]
    assert target == "local_symmetries"
    scores = []
    for v in (values[0], values[-1]):
        gray = cv2.cvtColor(fn(v), cv2.COLOR_BGR2GRAY)
        mask = gray < 128
        num, labels = cv2.connectedComponents(mask.astype(np.uint8))
        biggest = max(range(1, num), key=lambda k: (labels == k).sum())
        scores.append(_describe(labels == biggest, gray).vertical_symmetry)
    assert scores[0] > scores[-1] + 0.05, "shear must break vertical symmetry"


def test_symmetry_order_is_recoverable_from_the_motif_outline():
    values = list(range(1, 13))  # symmetric_motifs is no longer registered
    step = stimuli._SYM_STEP
    for m in values:
        img = stimuli.symmetric_motifs(m)
        # the motif at m = 1 is an egg, whose centroid is offset from the
        # drawing centre; measure about the centroid so the recovery does not
        # depend on knowing how the stimulus was constructed.
        cell = _gray(img)[:step, :step]
        ys, xs = np.nonzero(cell < 128)
        rad = _radial_profile(img, xs.mean(), ys.mean(), step // 2)
        assert _rotational_order(rad) == m


def test_symmetry_order_holds_motif_area_fixed():
    """The rosette family has an m-independent mean square radius, so the ink
    on the canvas does not move with the symmetry order."""
    values = list(range(1, 13))  # symmetric_motifs is no longer registered
    inks = [_ink(stimuli.symmetric_motifs(m)) for m in values]
    # 4% rather than 0: the *outline* encloses an m-independent area exactly,
    # but rasterising it does not, because the perimeter grows with m and so
    # does the number of boundary pixels. That is the same confound the
    # generator's docstring names — symmetry order and feature scale cannot be
    # separated inside a bounded motif — showing up in the ink.
    assert np.ptp(inks) / np.mean(inks) < 0.04


# --- deep interlock: interdigitation depth -------------------------------

def test_interlock_depth_is_recoverable_from_band_thickness():
    """A band is thick where its neighbours are thin.

    Thickness runs between ``band - 2a`` and ``band + 2a``, so its range over the
    canvas is ``4a``, and the parameter — the amplitude over the band half-height
    — is that range divided by twice the band.
    """
    _, values, _, _ = stimuli.SWEEPS["interlock_depth"]
    band = stimuli._INTERLOCK_BAND
    got = []
    for v in values:
        g = _gray(stimuli.interdigitated_bands(v))
        col = (g[:2 * band, :] < 128).sum(axis=0)  # the first dark band
        got.append(float(np.ptp(col)) / (2.0 * band))
    assert _monotone(values, got) == 1.0
    assert np.allclose(got, values, atol=0.02)


def test_interlock_depth_holds_the_ink_fixed():
    """The square wave is symmetric about the nominal boundary, so each region
    keeps its area however deeply the two interpenetrate."""
    _, values, _, _ = stimuli.SWEEPS["interlock_depth"]
    inks = [_ink(stimuli.interdigitated_bands(v)) for v in values]
    assert np.ptp(inks) < 0.01


# --- contrast: tonal delta ----------------------------------------------

def test_tonal_delta_is_recoverable_as_a_tone_difference():
    _, values, _, _ = stimuli.SWEEPS["tonal_delta"]
    got = []
    for v in values:
        g = _gray(stimuli.contrast_field(v)).astype(float)
        got.append((g.max() - g.min()) / stimuli._CONTRAST_SPAN)
    assert _monotone(values, got) == 1.0
    assert np.allclose(got, values, atol=0.03)


def test_tonal_delta_is_centred_on_mid_grey():
    """The repair on #19: the area-weighted mean is 128 at every sweep point, so
    the flat-field gain is unity and the swept contrast reaches the detector
    undivided. Before this, the mean sat at 227-239 and the gain near 0.54."""
    _, values, _, _ = stimuli.SWEEPS["tonal_delta"]
    for v in values:
        assert abs(_gray(stimuli.contrast_field(v)).mean() - 128) < 3


# --- gradients: zone transition width ------------------------------------

def test_zone_width_is_recoverable_from_the_motif_tone_profile():
    """Width of the band over which motif tone runs from dark zone to light."""
    _, values, _, _ = stimuli.SWEEPS["zone_width"]
    step, n = stimuli._ZONE_STEP, stimuli.CANVAS
    xs = np.array(sorted({x for x, _ in stimuli._full_lattice(n, step)}))
    got = []
    for v in values:
        g = _gray(stimuli.zone_transition(v)).astype(float)
        tone = np.array([g[step // 2, x] for x in xs])
        lo, hi = stimuli._ZONE_DARK, stimuli._ZONE_LIGHT
        ramping = (tone > lo + 1) & (tone < hi - 1)
        got.append((np.ptp(xs[ramping]) + step) / n if ramping.any() else 0.0)
    # not exactly 1: the recovery can only resolve the transition to the nearest
    # lattice pitch, so adjacent sweep points occasionally tie. The construction
    # is exact; the readout is quantised.
    assert _monotone(values, got) >= 0.99


def test_zone_width_holds_the_geometry_fixed():
    """Only the tone assignment moves; the lattice is identical throughout."""
    _, values, _, _ = stimuli.SWEEPS["zone_width"]
    counts = [len(_components(stimuli.zone_transition(v))) for v in values]
    inks = [_ink(stimuli.zone_transition(v)) for v in values]
    assert len(set(counts)) == 1
    assert np.ptp(inks) < 1e-9


# --- echoes and simplicity: vocabulary size ------------------------------

def _outline_signature(contour, samples=128, harmonics=13):
    """Scale- and rotation-invariant signature of a closed outline.

    The radial distance from the centroid, resampled at equal angles and divided
    by its own mean, is invariant to size and position; the magnitude spectrum of
    that profile is invariant to rotation as well. Hu moments were tried first and
    are useless here — a circle, a pentagon and a hexagon differ in log-Hu space
    by about 0.02, far inside any threshold that separates genuinely different
    shapes, so they cannot count a vocabulary.
    """
    pts = contour.reshape(-1, 2).astype(float)
    cx, cy = pts.mean(0)
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    rad = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    order = np.argsort(ang)
    grid = np.linspace(-np.pi, np.pi, samples, endpoint=False)
    prof = np.interp(grid, ang[order], rad[order], period=2 * np.pi)
    prof = prof / prof.mean()
    return np.abs(np.fft.rfft(prof))[1:harmonics + 1] / samples


def _distinct_outlines(img, tol=0.038):
    """Count distinct shapes by clustering their outline signatures."""
    clusters = []
    for c in _contours(img):
        s = _outline_signature(c)
        if not any(np.linalg.norm(s - k) < tol for k in clusters):
            clusters.append(s)
    return len(clusters)


def test_shape_vocabulary_size_is_recoverable():
    """Distinct outlines, counted from the image by shape descriptor."""
    _, values, _, _ = stimuli.SWEEPS["shape_vocabulary"]
    got = [_distinct_outlines(stimuli.shape_vocabulary(k)) for k in values]
    assert got == list(values)


def test_shape_vocabulary_holds_the_slot_layout_fixed():
    """Positions, counts and per-band areas are identical at every sweep
    point, so a centre count cannot track the parameter through geometry."""
    _, values, _, _ = stimuli.SWEEPS["shape_vocabulary"]
    counts, inks = [], []
    for k in values:
        img = stimuli.shape_vocabulary(k)
        counts.append(len(_components(img)))
        inks.append(_ink(img))
    assert len(set(counts)) == 1
    assert np.ptp(inks) / np.mean(inks) < 0.03


def test_element_vocabulary_size_is_recoverable():
    _, values, _, _ = stimuli.SWEEPS["element_kinds"]
    got = [_distinct_outlines(stimuli.element_vocabulary(k)) for k in values]
    assert got == list(values)


def test_element_vocabulary_holds_the_lattice_and_ink_fixed():
    _, values, _, _ = stimuli.SWEEPS["element_kinds"]
    counts, inks = [], []
    for k in values:
        img = stimuli.element_vocabulary(k)
        counts.append(len(_components(img)))
        inks.append(_ink(img))
    assert len(set(counts)) == 1
    assert np.ptp(inks) / np.mean(inks) < 0.03


# --- not-separateness: bleed --------------------------------------------

def test_bleed_is_recoverable_as_a_transition_width():
    """The 10-90 tone rise across a motif's edge, over the motif radius."""
    _, values, _, _ = stimuli.SWEEPS["bleed"]
    r, step = stimuli._BLEED_RADIUS, stimuli._BLEED_STEP
    lo, hi = stimuli._BLEED_INK, stimuli._BLEED_GROUND
    got = []
    for v in values:
        g = _gray(stimuli.bleeding_motifs(v)).astype(float)
        row = g[step // 2, step // 2:step // 2 + step]
        # 10% and 90% crossings, interpolated, so the width is resolved to a
        # fraction of a pixel. Counting whole intermediate pixels instead ties
        # adjacent sweep points together at both ends, where the steps are
        # under two pixels.
        rise = row[:int(np.argmax(row >= hi - 1)) + 1]
        idx = np.arange(len(rise), dtype=float)
        x10 = np.interp(lo + 0.1 * (hi - lo), rise, idx)
        x90 = np.interp(lo + 0.9 * (hi - lo), rise, idx)
        got.append((x90 - x10) / 0.8 / r)
    # The floor is two pixels: at bleed = 0 the tone steps across one boundary
    # pixel, and the 10-90 span of a step is about two. The first two sweep
    # points therefore read alike — 0 and 0.055 differ by 1.9 px of ramp — which
    # costs a rank and caps rho at 0.998. They are still different images, which
    # is what the sweep needs; only this readout cannot separate them.
    assert _monotone(values, got) >= 0.99
    assert np.allclose(got, values, atol=2.5 / r)
    assert not np.array_equal(stimuli.bleeding_motifs(values[0]),
                              stimuli.bleeding_motifs(values[1]))


def test_bleed_holds_the_half_tone_contour_and_the_ink_fixed():
    """The ramp is centred on the motif radius, so the motif's half-tone
    contour — and the mean tone of the canvas — do not move with the bleed."""
    _, values, _, _ = stimuli.SWEEPS["bleed"]
    inks = [_ink(stimuli.bleeding_motifs(v)) for v in values]
    assert np.ptp(inks) / np.mean(inks) < 0.03


def test_bleed_ramps_stay_below_the_pipelines_edge_floor_ceiling():
    """The widest bleed in the sweep must still present a gradient the pipeline
    is willing to call a boundary — otherwise the sweep runs off the end of the
    instrument rather than measuring it."""
    from centres.field import _EDGE_FLOOR

    widest = max(stimuli.SWEEPS["bleed"][1]) * stimuli._BLEED_RADIUS
    contrast = stimuli._BLEED_GROUND - stimuli._BLEED_INK
    assert 8.0 * contrast / widest > _EDGE_FLOOR
