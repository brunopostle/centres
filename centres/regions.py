"""Regions: the spatial extent each centre occupies, with real geometry.

Everything in this module exists because of one finding. The audit compared each
of Alexander's fifteen properties, as stated in Salingaros (2025), against what
`properties.py` computes, and found that eleven measure a different quantity from
the one named on them — not through a slip in the formula, but because the
information was never in the representation.

A centre was `(x, y, scale, strength)`. That carries position and size and
nothing else. But the source defines:

- *good shape* as **compactness** — "compact shapes are cognitively graspable"
- *positive space* as **convexity** — "the experienced space is typically convex,
  while the enclosing solid boundary is mostly concave"
- *thick boundaries* by a **ratio** — "the boundary measures roughly 1/3 of what
  it bounds"
- *deep interlock* as a **complex interface** — "two regions interpenetrate at a
  semi-permeable interface … a complex (not brusque) interface"
- *echoes* as **shape similarity** — "similar visual patterns and shapes repeated"
- *contrast* as **tone** — "black-white and colour contrast"
- *local symmetries* as **bilateral symmetry about the vertical axis**

None of compactness, convexity, thickness, interface complexity, shape or tone is
recoverable from a position and a radius. This module supplies them, by giving
each centre the region it actually occupies.

**One limit is worth knowing before relying on this.** The regions are seeded at
detected centres, and a centre is a maximum of the distance transform — so a
region only exists where something is *far from an edge*. Thin structures are
therefore invisible to it: a band of half-width w produces a maximum of exactly
w, and on the band-thickness stimulus the dark bands peak at 3.6 to 14.2 px while
the light areas they separate peak at 60.2 px. Not one centre lands on a band, at
any thickness. Properties about thin things — thick boundaries, deep interlock —
cannot be measured here however the watershed is seeded, and need the band
structure read from the image instead.

The segmentation is a watershed of the structural field, seeded at the detected
centres. That is the natural choice here: the field is a distance transform, its
ridges are the medial axis of the edge map, and watershed on its negation cuts
exactly along those ridges — so each centre receives the basin it dominates, and
basin boundaries fall where the original image had edges.
"""

from dataclasses import dataclass, field as _field

import cv2
import numpy as np


#: Seed radius for the watershed, as a fraction of a centre's own scale.
SEED_FRACTION = 0.25


@dataclass
class Region:
    """The extent of one centre, and the shape descriptors the properties need.

    All descriptors are dimensionless, so none of them carries a pixel scale that
    would make it depend on the photograph rather than on the artwork — the
    invariant documented in ``centres/field.py``.
    """

    #: Pixel area, the one extensive quantity, kept for weighting.
    area: float = 0.0

    #: 4*pi*A / P^2. One for a disc, falling towards zero as the outline becomes
    #: ragged or elongated. This is *good shape* as the source defines it.
    compactness: float = 0.0

    #: Area over the area of the convex hull. One for a convex region. This is
    #: *positive space*: the source says experienced space is convex and its
    #: enclosing solid is concave, so the two populations should differ here.
    solidity: float = 0.0

    #: Mean intensity over the region, normalised to [0, 1]. Tone, which the
    #: structural field discards entirely.
    tone: float = 0.0

    #: Standard deviation of intensity within the region, normalised. Texture.
    tone_spread: float = 0.0

    #: Overlap between the region and its own mirror image about a vertical axis
    #: through its centroid, as a fraction of area. One for a bilaterally
    #: symmetric region. The source is specific that the axis is the vertical
    #: one — "bilateral symmetry about the vertical axis respects gravitational
    #: stability".
    vertical_symmetry: float = 0.0

    #: The same about a horizontal axis, so that the two can be compared. The
    #: source distinguishes them; nothing else in the pipeline is directional.
    horizontal_symmetry: float = 0.0

    #: Ratio of the region's two principal axes, in [0, 1]. One is isotropic.
    elongation: float = 0.0

    #: Orientation of the major axis in radians, in [0, pi).
    orientation: float = 0.0

    #: Rotationally-invariant shape signature, for comparing one motif with
    #: another. Hu moments are log-scaled because they span many decades.
    shape_signature: tuple = _field(default_factory=tuple)

    #: Characteristic width, 2A/P — the width of the band if the region were one.
    #: In pixels, so use it only as a ratio against another length.
    thickness: float = 0.0

    #: Mean length of this region's shared interfaces with its neighbours, each
    #: divided by the square root of the smaller region's area. A straight cut
    #: across a compact region scores about 1; an interdigitating one scores
    #: several times that. This is *deep interlock* as the source defines it —
    #: "two regions interpenetrate at a semi-permeable interface … a complex
    #: (not brusque) interface joins the two regions into a larger whole."
    interface_complexity: float = 0.0

    #: Thickness of this region divided by the equivalent diameter of the largest
    #: neighbour it borders. This is *thick boundaries*: "the boundary measures
    #: roughly 1/3 of what it bounds."
    boundary_ratio: float = 0.0


def _mirror_overlap(mask, axis):
    """Fraction of a mask that survives reflection about its own centroid axis."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return 0.0
    if axis == "vertical":
        centre = xs.mean()
        reflected = np.rint(2 * centre - xs).astype(int)
        keep = (reflected >= 0) & (reflected < mask.shape[1])
        hit = mask[ys[keep], reflected[keep]]
    else:
        centre = ys.mean()
        reflected = np.rint(2 * centre - ys).astype(int)
        keep = (reflected >= 0) & (reflected < mask.shape[0])
        hit = mask[reflected[keep], xs[keep]]
    return float(hit.sum() / ys.size)


def _describe(mask, gray):
    """Compute every descriptor for one region mask."""
    r = Region()
    area = float(mask.sum())
    if area < 4:
        return r
    r.area = area

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return r
    contour = max(contours, key=cv2.contourArea)
    perimeter = float(cv2.arcLength(contour, True))
    contour_area = float(cv2.contourArea(contour))
    if perimeter > 0:
        # Clipped at 1: a digitised disc can exceed it slightly, and a value
        # above 1 would mean "more compact than a circle", which is meaningless.
        r.compactness = min(4.0 * np.pi * contour_area / (perimeter**2), 1.0)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    if hull_area > 0:
        r.solidity = min(contour_area / hull_area, 1.0)

    if perimeter > 0:
        r.thickness = 2.0 * contour_area / perimeter

    values = gray[mask]
    r.tone = float(values.mean()) / 255.0
    r.tone_spread = float(values.std()) / 255.0

    r.vertical_symmetry = _mirror_overlap(mask, "vertical")
    r.horizontal_symmetry = _mirror_overlap(mask, "horizontal")

    ys, xs = np.nonzero(mask)
    cov = np.cov(np.vstack([xs, ys]).astype(float))
    if cov.ndim == 2:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        if eigenvalues[1] > 1e-9:
            r.elongation = float(np.sqrt(max(eigenvalues[0], 0.0) / eigenvalues[1]))
            major = eigenvectors[:, 1]
            r.orientation = float(np.arctan2(major[1], major[0]) % np.pi)

    moments = cv2.HuMoments(cv2.moments(mask.astype(np.uint8))).flatten()
    r.shape_signature = tuple(
        float(-np.sign(m) * np.log10(abs(m))) if abs(m) > 1e-30 else 0.0
        for m in moments
    )
    return r


def segment_regions(field, centers, gray):
    """Give every centre the region it dominates, described.

    Watershed on the negated structural field, seeded one marker per centre. The
    field's ridges are the medial axis of the edge map, so the basins meet where
    the original image had edges and each centre receives the extent it actually
    occupies.

    Centres whose markers collide — two detections inside one basin — share that
    basin's descriptors. That is the honest outcome rather than an error: it says
    the detector found two centres where the segmentation finds one region.
    """
    if not centers:
        return centers
    h, w = field.shape[:2]

    # Seed each centre with a small disc rather than a single pixel. A lone
    # pixel is a fragile marker: it can land on a ridge of the field, or be
    # swallowed by a neighbouring basin, and then that centre gets no region at
    # all -- measured at 40% of centres on a jittered lattice. The disc is a
    # fraction of the centre's own scale, per the units invariant. Larger
    # centres are stamped first so that a smaller centre nested inside one keeps
    # its own seed.
    markers = np.zeros((h, w), np.int32)
    for i, c in sorted(enumerate(centers), key=lambda p: -p[1].scale):
        y = int(np.clip(round(c.y), 0, h - 1))
        x = int(np.clip(round(c.x), 0, w - 1))
        cv2.circle(markers, (x, y), max(1, int(c.scale * SEED_FRACTION)), i + 1, -1)

    # watershed wants an 8-bit 3-channel image; the negated field puts basins
    # where the field peaks, which is where the centres are.
    relief = np.clip((1.0 - field) * 255.0, 0, 255).astype(np.uint8)
    cv2.watershed(cv2.cvtColor(relief, cv2.COLOR_GRAY2BGR), markers)

    for i, c in enumerate(centers):
        mask = markers == (i + 1)
        c.region = _describe(mask, gray) if mask.any() else Region()

    _describe_interfaces(markers, centers)
    return centers


def _describe_interfaces(markers, centers):
    """Measure what each region shares with its neighbours.

    Two properties are about the *relation* between adjacent regions rather than
    about either one alone, and neither can be computed from a region in
    isolation: *deep interlock* is the complexity of the shared interface, and
    *thick boundaries* is a ratio between a band and what it bounds.

    Interface length is counted along the watershed line. ``cv2.watershed`` marks
    it -1 and leaves it one pixel wide, so two basins never carry adjacent labels
    -- there is always a line between them, and counting label-to-label adjacency
    directly finds nothing at all. Each line pixel is instead attributed to the
    pair of labels it separates, which is both correct and a direct measure of
    interface length in pixels.
    """
    h, w = markers.shape
    ys, xs = np.nonzero(markers == -1)
    if ys.size == 0:
        return

    stack = []
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        stack.append(markers[np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)])
    around = np.stack(stack)

    sentinel = np.iinfo(np.int32).max
    highest = np.where(around > 0, around, 0).max(axis=0)
    lowest = np.where(around > 0, around, sentinel).min(axis=0)
    separating = (highest > 0) & (lowest < sentinel) & (highest != lowest)
    if not separating.any():
        return

    keys, counts = np.unique(
        np.stack([lowest[separating], highest[separating]]), axis=1, return_counts=True
    )
    pairs = {(int(p), int(q)): int(n) for (p, q), n in zip(keys.T, counts)}

    neighbours = {}
    for (p, q), length in pairs.items():
        neighbours.setdefault(p, []).append((q, length))
        neighbours.setdefault(q, []).append((p, length))

    for i, c in enumerate(centers):
        adjacent = neighbours.get(i + 1, [])
        if not adjacent or c.region is None or c.region.area <= 0:
            continue
        complexities, bounded = [], []
        for j, length in adjacent:
            other = centers[j - 1].region
            if other is None or other.area <= 0:
                continue
            # A straight cut across the smaller region is about sqrt(area) long.
            smaller = min(c.region.area, other.area)
            complexities.append(length / np.sqrt(smaller))
            bounded.append(other)
        if complexities:
            c.region.interface_complexity = float(np.mean(complexities))
        if bounded:
            largest = max(bounded, key=lambda r: r.area)
            diameter = 2.0 * np.sqrt(largest.area / np.pi)
            if diameter > 0:
                c.region.boundary_ratio = float(c.region.thickness / diameter)
