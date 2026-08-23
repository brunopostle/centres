import cv2
import numpy as np
import pytest
from centres.centers import Center
from centres.field import _field_and_distance
from centres.pipeline import detect_centers, assign_hierarchy, analyze


def gaussian_field(shape=(100, 100), yx=(50, 50), sigma=10):
    h, w = shape
    Y, X = np.mgrid[0:h, 0:w]
    field = np.exp(-((X - yx[1]) ** 2 + (Y - yx[0]) ** 2) / (2 * sigma**2))
    return field.astype(float)


def make_bgr_with_rect(h=120, w=120):
    img = np.full((h, w, 3), 180, dtype=np.uint8)
    img[30:90, 30:90] = 20
    return img


# --- detect_centers ---


def test_detect_centers_finds_blob():
    field = gaussian_field(yx=(50, 60), sigma=10)
    centers = detect_centers(field)
    assert len(centers) >= 1


def test_detected_center_near_blob():
    field = gaussian_field(yx=(50, 60), sigma=10)
    centers = detect_centers(field)
    xs = np.array([c.x for c in centers])
    ys = np.array([c.y for c in centers])
    dists = np.sqrt((xs - 60) ** 2 + (ys - 50) ** 2)
    assert dists.min() < 15


def test_detect_centers_returns_center_objects():
    centers = detect_centers(gaussian_field())
    for c in centers:
        assert isinstance(c, Center)
        assert c.scale > 0
        assert 0.0 <= c.strength <= 1.0


def test_detect_centers_ids_sequential():
    centers = detect_centers(gaussian_field())
    ids = [c.id for c in centers]
    assert ids == list(range(len(ids)))


def test_detect_centers_invariant_to_field_amplitude():
    """Rescaling the whole field by a positive constant must not change what's
    detected (#27). blob_log's threshold is a fraction of each field's own
    peak response (threshold_rel), not an absolute number a rescaled field
    could fall short of or blow past -- which is what let a single outlier
    pixel (via field.max() in build_structural_field) silently change what
    counted as a detection everywhere.
    """
    field = gaussian_field(yx=(50, 60), sigma=10)
    base = detect_centers(field)
    scaled = detect_centers(field * 0.05)
    assert len(base) == len(scaled) and len(base) >= 1
    for c1, c2 in zip(base, scaled):
        assert c1.x == pytest.approx(c2.x)
        assert c1.y == pytest.approx(c2.y)
        assert c1.scale == pytest.approx(c2.scale)


def _circle_canvas(size=400, r=40, bg=230, fg=40):
    img = np.full((size, size, 3), bg, np.uint8)
    cv2.circle(img, (size // 2, size // 2), r, (fg, fg, fg), -1)
    return img


def _lattice_canvas(size=400, n=3, r=25, bg=230, fg=40, margin=60):
    img = np.full((size, size, 3), bg, np.uint8)
    positions = np.linspace(margin, size - margin, n)
    centres = []
    for y in positions:
        for x in positions:
            cv2.circle(img, (int(x), int(y)), r, (fg, fg, fg), -1)
            centres.append((y, x))
    return img, centres


def test_single_circle_on_blank_canvas_yields_one_centre():
    """A single circle on an otherwise empty canvas must detect exactly the
    circle, not the canvas corners (#8).

    Before the enclosure filter, the distance transform is *largest* at
    whichever point is farthest from the circle's edge -- typically a canvas
    corner -- so the corners and edge-midpoints outscored the circle itself
    and the circle wasn't detected at all (its LoG response never cleared the
    detection threshold once four corners and four edge-midpoints did first).
    """
    field, dist = _field_and_distance(_circle_canvas())
    centers = detect_centers(field, dist)
    assert len(centers) == 1
    c = centers[0]
    assert np.hypot(c.x - 200, c.y - 200) < 15


def test_circle_lattice_has_no_corner_or_margin_detections():
    """Every circle in a 3x3 lattice is detected, and nothing is detected in
    the blank margin around them -- in particular not at the four canvas
    corners, the plateau this issue is about (#8).

    Total count is not asserted at exactly 9: real, closely-spaced circles
    also produce genuine local maxima in the gaps between them (interstitial
    "background" centres, a real and separately-handled population -- see
    Center.polarity and #26 -- not a plateau artifact). Those are legitimately
    enclosed by nearby structure on every side, which is exactly what this
    filter is checking for, so it correctly leaves them alone.
    """
    img, true_centres = _lattice_canvas()
    field, dist = _field_and_distance(img)
    centers = detect_centers(field, dist)

    for (ty, tx) in true_centres:
        nearest = min(np.hypot(c.x - tx, c.y - ty) for c in centers)
        assert nearest < 15, f"no detection near true circle at ({tx}, {ty})"

    h, w = img.shape[:2]
    margin = 30
    for c in centers:
        in_margin = (
            c.x < margin or c.x > w - margin or c.y < margin or c.y > h - margin
        )
        assert not in_margin, f"spurious margin/corner detection at ({c.x}, {c.y})"


def test_detect_centers_without_dist_skips_enclosure_filter():
    """dist is optional (most unit tests detect on a hand-built field with no
    distance transform to give); omitting it must not change behaviour."""
    field = gaussian_field(yx=(50, 60), sigma=10)
    assert detect_centers(field) == detect_centers(field, None)


# --- assign_hierarchy ---


def test_parent_assigned_when_contained():
    # small centre at dist=5 from large centre with scale=30 → contained
    large = Center(id=0, x=50.0, y=50.0, scale=30.0, strength=0.8)
    small = Center(id=1, x=55.0, y=50.0, scale=5.0, strength=0.5)
    result = assign_hierarchy([large, small])
    assert result[1].parent == 0


def test_no_parent_when_not_contained():
    # small centre at dist=50 from large with scale=10 → outside
    large = Center(id=0, x=0.0, y=0.0, scale=10.0, strength=0.8)
    small = Center(id=1, x=50.0, y=0.0, scale=5.0, strength=0.5)
    result = assign_hierarchy([large, small])
    assert result[1].parent is None


def test_largest_centre_has_no_parent():
    centers = [
        Center(id=0, x=0.0, y=0.0, scale=50.0, strength=1.0),
        Center(id=1, x=5.0, y=0.0, scale=10.0, strength=0.5),
    ]
    result = assign_hierarchy(centers)
    assert result[0].parent is None


def test_assign_hierarchy_empty():
    assert assign_hierarchy([]) == []


# --- analyze ---


def test_analyze_field_shape():
    img = make_bgr_with_rect()
    field, centers, G, energy = analyze(img)
    assert field.shape == img.shape[:2]


def test_analyze_field_normalized():
    field, centers, G, energy = analyze(make_bgr_with_rect())
    assert field.min() >= 0.0
    assert field.max() <= 1.0 + 1e-6


def test_analyze_energy_finite():
    _, _, _, energy = analyze(make_bgr_with_rect())
    assert np.isfinite(energy)


def test_analyze_centers_list():
    _, centers, _, _ = analyze(make_bgr_with_rect())
    assert isinstance(centers, list)
