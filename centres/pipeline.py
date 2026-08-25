from .field import reconstruct_field, _field_and_distance
from .graph import build_graph, propagate_strength
from .regions import segment_regions
from .energy import total_energy
from .centers import Center
import cv2
from skimage.feature import blob_log
from scipy.spatial.distance import cdist
import numpy as np


#: Fraction of the field's own peak LoG response a local maximum must reach to
#: count as a detection. Relative rather than absolute (#27): the Laplacian is a
#: linear filter, so the LoG response scales in direct proportion to whatever
#: build_structural_field's amplitude turns out to be for a given image --
#: which, via field.max(), depends on a single outlier pixel (a vignette, a
#: crop, a mount). An absolute threshold compared against that response is
#: exactly what let field.max() silently change what counts as a detection
#: everywhere; a threshold relative to each image's own response range cannot
#: be moved by rescaling the field, whatever the field's amplitude is set by.
#: Calibrated against the #19 generator sweeps to land close to the counts the
#: old absolute threshold=0.08 gave wherever it was already well-behaved (e.g.
#: jitter: 481, 472, 478 for the first three sweep points, matching to the count).
_DETECTION_THRESHOLD_REL = 0.2

#: Directions rays are cast in when testing whether a detection is enclosed
#: (#8). Real bounded regions are not exotic shapes at this test's scale, so
#: few directions already separate them cleanly from a plateau -- see
#: _enclosure_fraction.
_ENCLOSURE_RAYS = 8
#: A ray counts as reaching a boundary once dist drops to this fraction of the
#: candidate's own dist value.
_ENCLOSURE_DECAY = 0.4
#: Fraction of the rays that must reach a boundary for a detection to count as
#: enclosed rather than sitting on a structureless plateau. 3-of-8: a majority
#: (0.5, 4-of-8) turned out to suppress real, weakly-bounded structure along
#: with true plateaus -- discovered when #22's not_separateness sweep, which
#: needs exactly that kind of soft-boundary population, regressed from rho
#: +0.78 to +0.29. 3-of-8 recovers it (+0.70) while still passing the single
#: circle and lattice acceptance cases; corpus plateau suppression is weaker
#: as a result (e.g. Pazyryk drops 219->207 rather than 219->151), a real
#: trade-off, not a wash. 2-of-8 recovers tracking further but lets plateau
#: detections back through the lattice test, so it isn't a free lunch either.
_ENCLOSURE_FRACTION = 0.375
_ENCLOSURE_ANGLES = 2 * np.pi * np.arange(_ENCLOSURE_RAYS) / _ENCLOSURE_RAYS


def _enclosure_fraction(y, x, dist):
    """Of the rays cast outward from (y, x), what fraction reach a boundary?

    ``dist`` is the *raw*, uncapped distance-to-edge transform -- the true
    distance to the single nearest edge in the closest direction, which is
    exactly what a distance transform gives for free. A point at the centre of
    a real bounded region is roughly equidistant from an edge in most
    directions, however far that distance is; a point on a structureless
    plateau (the far corner of a mostly-blank canvas, say) has an edge nearby
    in at most a few directions and open space in the rest, which is what made
    it "far from an edge" in the first place. Casting rays of length equal to
    the candidate's own dist value and checking whether each still finds an
    edge nearby tells the two apart without reference to any per-image scale
    estimate (#8's own root-cause note: the plateau is defined by an absence
    of structure, not by a distance relative to one), and without depending on
    the LoG's reported scale, which for a plateau detection is an artefact of
    the ladder's ceiling rather than a real one (#9).
    """
    h, w = dist.shape
    d0 = float(dist[y, x])
    if d0 <= 1e-6:
        return 1.0
    ys = np.clip(np.round(y + d0 * np.sin(_ENCLOSURE_ANGLES)).astype(int), 0, h - 1)
    xs = np.clip(np.round(x + d0 * np.cos(_ENCLOSURE_ANGLES)).astype(int), 0, w - 1)
    return float(np.count_nonzero(dist[ys, xs] <= _ENCLOSURE_DECAY * d0)) / _ENCLOSURE_RAYS


def detect_centers(field, dist=None):
    """Detect multi-scale centres using Laplacian-of-Gaussian blob detection.

    A single blob_log call with log-spaced sigma values avoids duplicate
    detections that occurred when running four separate overlapping scale ranges.
    blob_log applies its own NMS internally via the overlap parameter.

    ``dist``, if given, is the raw distance-to-edge transform ``field`` was
    built from (see ``centres.field._field_and_distance``). When present, a
    detection sitting on a structureless plateau -- no enclosing boundary
    nearby in most directions -- is dropped rather than reported as a centre
    (#8). Optional so callers detecting on a hand-built field (most of the
    unit tests) get the plain detector.
    """
    blobs = blob_log(
        field, min_sigma=2, max_sigma=48, num_sigma=10,
        threshold=None, threshold_rel=_DETECTION_THRESHOLD_REL, log_scale=True,
    )
    centers = []
    for y, x, sigma in blobs:
        yi, xi = int(y), int(x)
        if dist is not None and _enclosure_fraction(yi, xi, dist) < _ENCLOSURE_FRACTION:
            continue
        centers.append(
            Center(
                id=len(centers),
                x=float(x),
                y=float(y),
                scale=float(sigma * np.sqrt(2)),
                strength=float(field[yi, xi]),
            )
        )
    return centers


#: Containment radius for the hierarchy, in units of the parent's own extent.
#: 1.0 is strict containment and fires for only 4% of centres; 1.5 reaches 74%
#: while keeping the in-band separation strict containment achieves. See
#: assign_hierarchy.
CONTAINMENT = 1.5


def _extent(centre):
    """A centre's true radius: the equivalent radius of the region it occupies.

    ``Center.scale`` is the LoG blob scale, which systematically understates how
    much space a centre actually takes up — measured at a median region radius of
    1.28 times the blob scale. Containment tested against the blob scale is
    therefore too tight, which is why the strict condition almost never fired and
    why a fudge factor of 3 was needed to make the hierarchy populate at all.

    Falls back to the blob scale when a centre has no region.
    """
    region = getattr(centre, "region", None)
    if region is not None and region.area > 0:
        return float(np.sqrt(region.area / np.pi))
    return float(centre.scale)


def assign_hierarchy(centers):
    """Assign each centre to the smallest centre whose extent contains it.

    Two changes from the previous version, which took the *nearest* larger centre
    within three times its blob scale.

    **Containment is tested against real extent.** The old test used
    ``dist < 3 * scale_j``, and the multiplier of 3 was documented as necessary
    because the strict condition "almost never fires" — measured at 4% of centres.
    That is a symptom: the blob scale understates a centre's extent, so strict
    containment against it is the wrong test rather than too strict a one. Against
    the region's equivalent radius the strict condition reaches 29%, and 1.5 times
    it reaches 74% while keeping the separation strict containment achieves.

    **The parent is the smallest containing centre, not the nearest.** A hierarchy
    is meant to record successive levels, and taking the nearest larger centre
    lets a small centre beside a large one skip every level between them, so the
    ratio recorded is not a step in the scaling hierarchy at all. This matters
    directly for *levels of scale*, which reads those ratios.

    Measured against a stimulus sweeping the constructed parent:child ratio, the
    two changes together lift the separation between ratios inside the sourced
    band of 2–5 and outside it from +0.050 to +0.091.
    """
    if not centers:
        return centers
    pos = np.array([[c.x, c.y] for c in centers])
    extent = np.array([_extent(c) for c in centers])
    dist = cdist(pos, pos)
    for i, c in enumerate(centers):
        candidates = np.where(
            (dist[i] < CONTAINMENT * extent) & (extent > extent[i] * (1.0 + 1e-9))
        )[0]
        if len(candidates) == 0:
            continue
        centers[i].parent = int(candidates[np.argmin(extent[candidates])])
    return centers


def analyze(image):
    field, dist = _field_and_distance(image)
    centers = detect_centers(field, dist)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    centers = assign_polarity(centers, gray)
    centers = segment_regions(field, centers, gray)
    centers = assign_hierarchy(centers)
    G = build_graph(centers)
    G = propagate_strength(G)
    field2 = reconstruct_field(field.shape, centers)
    energy = total_energy(field2, centers, G)
    return field2, centers, G, energy


def random_centers(n, shape):
    h, w = shape
    centers = []
    for i in range(n):
        centers.append(
            Center(
                id=i,
                x=np.random.uniform(0, w),
                y=np.random.uniform(0, h),
                scale=np.random.uniform(5, 40),
                strength=np.random.uniform(0.3, 1),
            )
        )
    return centers


def evolve(
    shape=(300, 300),
    iterations=200,
    n_centers=40,
    T_start=0.1,
    T_end=0.001,
    progress=None,
    initial_centers=None,
):
    """Generative evolution via simulated annealing toward minimum structural energy.

    Each iteration perturbs all centre positions and scales, evaluates energy,
    and accepts or rejects via the Metropolis criterion. Temperature decays
    exponentially from T_start to T_end.

    The defaults were lowered ten-fold with the move to the degree of life (#28).
    T_start=1.0 was sized against an objective spanning roughly 0.95 to 1.74; the
    degree of life spans about 0.32 to 0.41 with per-move deltas near 0.005, so
    the old schedule ran effectively hot throughout and accepted almost every
    proposal. Measured over three seeds, the cooler schedule gains about +0.03 in
    final degree of life (0.410 -> 0.443).

    initial_centers: if provided, seed the search from these centres rather
                     than from a random configuration. Useful for refining the
                     centre structure detected in an existing image.
    progress: optional callable(iteration, total, energy, temperature, accepted)
              called after each iteration.
    """
    h, w = shape
    if initial_centers is not None:
        centers = [
            Center(id=c.id, x=c.x, y=c.y, scale=c.scale, strength=c.strength)
            for c in initial_centers
        ]
    else:
        centers = random_centers(n_centers, shape)
    centers = assign_hierarchy(centers)
    # propagate_strength maps intrinsic strengths to their stationary point, so
    # it must always be fed the intrinsic values. Only x, y and scale are
    # annealed below; strength is a fixed input, not part of the search state.
    # Re-seeding each time keeps every iteration's propagation a pure function
    # of the current geometry. (Without this, feeding a propagated vector back
    # in amplifies it by up to 1/(1 - alpha) per iteration and the strengths
    # drift geometrically across the anneal.)
    intrinsic = [c.strength for c in centers]

    def _propagate(centers):
        for c, s in zip(centers, intrinsic):
            c.strength = s
        return propagate_strength(build_graph(centers))

    G = _propagate(centers)
    field = reconstruct_field(shape, centers)
    current_energy = total_energy(field, centers, G)

    for t in range(iterations):
        T = T_start * (T_end / T_start) ** (t / iterations)
        saved = [(c.x, c.y, c.scale, c.strength, c.parent) for c in centers]

        for c in centers:
            c.x = float(np.clip(c.x + np.random.normal(0, 2), 0, w))
            c.y = float(np.clip(c.y + np.random.normal(0, 2), 0, h))
            c.scale = float(np.clip(c.scale * np.exp(np.random.normal(0, 0.02)), 2, 80))

        centers = assign_hierarchy(centers)
        G_new = _propagate(centers)
        field_new = reconstruct_field(shape, centers)
        new_energy = total_energy(field_new, centers, G_new)

        delta = new_energy - current_energy
        if delta < 0 or np.random.random() < np.exp(-delta / (T + 1e-10)):
            current_energy = new_energy
            G, field = G_new, field_new
            accepted = True
        else:
            for c, (x, y, sc, st, p) in zip(centers, saved):
                c.x, c.y, c.scale, c.strength, c.parent = x, y, sc, st, p
            accepted = False

        if progress is not None:
            progress(t + 1, iterations, current_energy, T, accepted)

    return field, centers


#: Outer radius of the surround annulus, as a multiple of the centre's own scale.
#: In units of the centre, not of the image, per the invariant in field.py.
POLARITY_SURROUND = 2.0


def assign_polarity(centers, gray):
    """Label each centre with the signed contrast between it and its surround.

    For each centre, the Michelson contrast between the mean intensity within its
    own radius and the mean over the annulus from that radius out to
    ``POLARITY_SURROUND`` times it:

        polarity = (surround - interior) / (local range)

    where the range is over the same patch. Positive means the interior is darker
    than what surrounds it.

    Normalising by the local range rather than by the local sum — Michelson
    contrast, the obvious first choice — is what makes the measure exactly
    antisymmetric under inversion. Michelson flips sign when an image is
    inverted but does *not* preserve magnitude, so the same design rendered
    light-on-dark would score differently from dark-on-light. Which polarity
    counts as figure is a convention; the magnitude of the distinction should not
    depend on that convention.

    This is what lets anything downstream tell a motif from the gap between
    motifs. The two are indistinguishable in the structural field itself, which
    is a distance transform and carries no tone at all — see ``Center.polarity``
    and #26.
    """
    if not centers:
        return centers
    h, w = gray.shape[:2]
    img = gray.astype(np.float64)
    for c in centers:
        r = max(c.scale, 1.0)
        outer = int(np.ceil(r * POLARITY_SURROUND))
        x0, x1 = int(max(c.x - outer, 0)), int(min(c.x + outer + 1, w))
        y0, y1 = int(max(c.y - outer, 0)), int(min(c.y + outer + 1, h))
        patch = img[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d2 = (xx - c.x) ** 2 + (yy - c.y) ** 2
        inside = d2 <= r * r
        ring = (d2 > r * r) & (d2 <= (r * POLARITY_SURROUND) ** 2)
        if not inside.any() or not ring.any():
            continue
        interior = patch[inside].mean()
        surround = patch[ring].mean()
        spread = float(patch.max() - patch.min())
        c.polarity = float((surround - interior) / spread) if spread > 1e-9 else 0.0
    return centers
