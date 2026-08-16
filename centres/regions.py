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
    return centers
