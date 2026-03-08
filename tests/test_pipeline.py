import numpy as np
from centres.centers import Center
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
