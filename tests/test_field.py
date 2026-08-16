import numpy as np
import pytest
from centres.field import build_structural_field, reconstruct_field
from centres.centers import Center


def make_bgr(h, w, value=180):
    return np.full((h, w, 3), value, dtype=np.uint8)


def make_bgr_with_rect(h=100, w=100, fill=180, rect_fill=20):
    img = make_bgr(h, w, fill)
    img[20:80, 20:80] = rect_fill
    return img


# --- build_structural_field ---


def test_shape_matches_input():
    field = build_structural_field(make_bgr(80, 120))
    assert field.shape == (80, 120)


def test_range_zero_to_one():
    field = build_structural_field(make_bgr(80, 80))
    assert field.min() >= 0.0
    assert field.max() <= 1.0 + 1e-6


def test_field_is_bounded_in_units_of_the_cap():
    """The field is expressed as a fraction of the distance cap.

    This replaces a test asserting field.max() == 1.0, which was tautological
    while the field was divided by its own maximum. Dividing by the cap instead
    (#27) means 1.0 is a fixed structural quantity — CAP_SPACINGS times the
    typical edge spacing — so the maximum is only reached when some region is
    actually that large, and the value carries meaning.
    """
    field = build_structural_field(make_bgr_with_rect())
    assert field.min() >= 0.0
    # 0.1 * blur, itself in [0, 1], is added before dividing by the cap
    assert field.max() <= 1.0 + 0.1


def test_region_wider_than_the_cap_saturates():
    """A region far larger than the cap reaches the top of the scale."""
    from centres.field import CAP_SPACINGS, edge_spacing
    import cv2

    img = make_bgr(600, 600, value=20)
    img[2:598, 2:598] = 20
    img[0:2, :] = 200  # a thin frame is the only structure
    field = build_structural_field(img)
    assert field.max() == pytest.approx(1.0, abs=0.15)


def test_field_peaks_inside_bounded_region():
    # Large dark region with a narrow bright border: the interior is ~45px from the
    # nearest edge, exterior corners are at most ~7px — interior should score higher.
    img = make_bgr(100, 100, value=200)
    img[5:95, 5:95] = 20
    field = build_structural_field(img)
    assert field[50, 50] > field[0, 0]


# --- reconstruct_field ---


def test_reconstruct_shape():
    centers = [Center(id=0, x=25.0, y=25.0, scale=5.0, strength=1.0)]
    field = reconstruct_field((50, 60), centers)
    assert field.shape == (50, 60)


def test_reconstruct_range():
    centers = [Center(id=0, x=25.0, y=25.0, scale=5.0, strength=1.0)]
    field = reconstruct_field((50, 50), centers)
    assert field.min() >= 0.0
    assert field.max() == pytest.approx(1.0, abs=1e-6)


def test_reconstruct_peaks_near_center():
    centers = [Center(id=0, x=60.0, y=40.0, scale=5.0, strength=1.0)]
    field = reconstruct_field((100, 100), centers)
    peak = np.unravel_index(field.argmax(), field.shape)
    assert abs(peak[0] - 40) <= 2
    assert abs(peak[1] - 60) <= 2


def test_reconstruct_empty_centers():
    field = reconstruct_field((50, 50), [])
    assert field.shape == (50, 50)
    assert field.max() == 0.0
