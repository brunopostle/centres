"""Regression tests for issue #15: undefined is not zero, and not 10/10.

A featureless grey canvas detects no centres. Every deviation-based measure
then returned a raw 0, and 0 is those measures' *ideal* value, so
``normalize_all`` reported a blank image as 10/10 on seven of the fifteen
properties. The tool could not distinguish no structure from ideal structure.

These tests pin down two things:

1. **Where "undefined" begins, per measure.** The boundary is not one blanket
   rule. Most measures need one of the thing they average over; ``echoes``
   needs two, because the standard deviation of a single ratio is exactly 0 and
   0 is its ideal; ``not_separateness`` needs a connected component of two;
   ``alternating_repetition`` needs a node of degree two, which is stricter
   than "the graph has an edge". Each test below asserts *both* sides of that
   boundary, so a future blanket rule in either direction fails here.

2. **That every output path renders it as not-a-score**, rather than
   substituting a default.
"""

import json
import numpy as np
import pytest

from centres.centers import Center
from centres.field import reconstruct_field
from centres.graph import build_graph, propagate_strength
from centres.properties import (
    alternating_repetition,
    boundaries,
    compute_all,
    contrast,
    deep_interlock,
    echoes,
    good_shape,
    gradients,
    levels_of_scale,
    local_symmetries,
    normalize_all,
    not_separateness,
    positive_space,
    roughness,
    simplicity,
    strong_centres,
    the_void,
)

KEYS = [
    "levels_of_scale", "strong_centres", "boundaries", "alternating_repetition",
    "positive_space", "good_shape", "local_symmetries", "deep_interlock",
    "contrast", "gradients", "roughness", "echoes", "the_void",
    "simplicity", "not_separateness",
]


def c(id, x, y, scale, strength=1.0, parent=None):
    return Center(id=id, x=float(x), y=float(y), scale=float(scale),
                  strength=strength, parent=parent)


def _connected(centers):
    return propagate_strength(build_graph(centers))


def _empty_inputs():
    """What ``analyze`` hands back for a featureless canvas."""
    return np.zeros((64, 64)), [], build_graph([])


# --------------------------------------------------------------------------
# Where the boundary lies, measure by measure.
# Each test asserts the undefined side and the first defined side.
# --------------------------------------------------------------------------

# --- measures over parent-child pairs: undefined with none, defined with one ---


@pytest.mark.parametrize("fn", [levels_of_scale, positive_space, local_symmetries])
def test_pair_measures_undefined_without_hierarchy(fn):
    assert fn([]) is None
    assert fn([c(0, 0, 0, 30.0), c(1, 10, 0, 10.0)]) is None  # centres, no pairs


@pytest.mark.parametrize("fn", [levels_of_scale, positive_space, local_symmetries])
def test_pair_measures_defined_with_one_pair(fn):
    # The mean of a single deviation is a deviation. One pair is enough.
    assert fn([c(0, 0, 0, 30.0), c(1, 10, 0, 10.0, parent=0)]) is not None


# --- echoes: the one measure whose boundary is two, not one ---


def test_echoes_undefined_without_pairs():
    assert echoes([]) is None
    assert echoes([c(0, 0, 0, 30.0), c(1, 10, 0, 10.0)]) is None


def test_echoes_undefined_with_exactly_one_pair():
    """std of one ratio is 0, and 0 is this measure's ideal — so one pair
    would be reported as perfect self-similarity. An echo needs two
    occurrences to be an echo."""
    assert echoes([c(0, 0, 0, 30.0), c(1, 10, 0, 10.0, parent=0)]) is None


def test_echoes_defined_with_two_pairs():
    centers = [c(0, 0, 0, 90.0), c(1, 5, 0, 30.0, parent=0),
               c(2, -5, 0, 30.0, parent=0)]
    assert echoes(centers) == pytest.approx(0.0, abs=1e-10)


# --- measures over centres: undefined with none, defined with one ---


def test_strong_centres_boundary():
    assert strong_centres([]) is None
    assert strong_centres([c(0, 0, 0, 5.0, strength=0.7)]) == pytest.approx(0.7)


def test_simplicity_boundary():
    assert simplicity([]) is None
    # A one-element population has no inequality: Gini 0, defined.
    assert simplicity([c(0, 0, 0, 5.0, strength=0.7)]) == pytest.approx(0.0, abs=1e-6)


def test_good_shape_boundary():
    """Undefined with no centres — the denominator is the centre count. But
    centres with no hierarchy give a genuine 0: no centre has sub-structure."""
    assert good_shape([]) is None
    assert good_shape([c(0, 0, 0, 5.0), c(1, 50, 0, 5.0)]) == 0.0


def test_the_void_boundary():
    assert the_void(np.ones((50, 50)), []) is None
    centers = [c(0, 25, 25, 10.0, strength=2.0)]
    assert the_void(reconstruct_field((50, 50), centers), centers) is not None


def test_the_void_undefined_when_strongest_centre_is_off_frame():
    """No pixel of the field falls inside the disc, so there is nothing to
    average a gradient over."""
    assert the_void(np.ones((50, 50)), [c(0, 500, 500, 3.0)]) is None


def test_gradients_boundary():
    """The field read here is the *reconstructed* field. With no centres it is
    identically zero, so its mean squared gradient is 0 by construction — which
    normalises to a perfect 10."""
    field = np.zeros((50, 50))
    field[:, 25:] = 1.0
    assert gradients(field, []) is None
    assert gradients(field, [c(0, 25, 25, 5.0)]) > 0


# --- roughness: needs two centres to have a spacing at all ---


def test_roughness_boundary():
    assert roughness([]) is None
    assert roughness([c(0, 0, 0, 5.0)]) is None
    # Two centres give a (degenerate) spacing distribution: CV = 0, defined.
    assert roughness([c(0, 0, 0, 5.0), c(1, 40, 0, 5.0)]) == pytest.approx(0.0, abs=1e-6)


# --- measures over graph edges: undefined with none, defined with one ---


def test_boundaries_boundary():
    far = [c(0, 0, 0, 3.0), c(1, 500, 0, 3.0)]
    assert boundaries(reconstruct_field((100, 100), far), far, build_graph(far)) is None
    near = [c(0, 30, 50, 6.0), c(1, 55, 50, 6.0)]
    G = _connected(near)
    assert G.edges
    assert boundaries(reconstruct_field((100, 100), near), near, G) is not None


def test_boundaries_undefined_when_field_is_empty_at_every_peak():
    """Edges exist, but the field is flat zero, so no ratio can be formed."""
    centers = [c(0, 30, 50, 6.0), c(1, 55, 50, 6.0)]
    G = _connected(centers)
    assert G.edges
    assert boundaries(np.zeros((100, 100)), centers, G) is None


def test_contrast_boundary():
    assert contrast(build_graph([c(0, 0, 0, 3.0), c(1, 500, 0, 3.0)])) is None
    # Edges with equal strengths are a genuine zero, not an undefined.
    G = _connected([c(0, 0, 0, 10.0, strength=1.0), c(1, 15, 0, 10.0, strength=1.0)])
    assert G.edges
    assert contrast(G) == pytest.approx(0.0, abs=1e-6)


def test_deep_interlock_boundary():
    assert deep_interlock([], build_graph([])) is None
    centers = [c(0, 0, 0, 10.0), c(1, 15, 0, 10.0)]
    G = _connected(centers)
    assert G.edges
    assert deep_interlock(centers, G) == pytest.approx(1.0, abs=1e-6)


# --- alternating_repetition: needs a node of degree 2, not merely an edge ---


def test_alternating_repetition_undefined_without_edges():
    assert alternating_repetition(build_graph([])) is None
    assert alternating_repetition(build_graph([c(0, 0, 0, 3.0), c(1, 500, 0, 3.0)])) is None


def test_alternating_repetition_undefined_for_isolated_pairs():
    """A graph of disjoint pairs has edges, but every node has one neighbour.
    A single neighbour cannot alternate with anything, so the quantity being
    averaged does not exist. This is why "the graph has edges" is the wrong
    test for this measure."""
    centers = [c(0, 0, 0, 10.0), c(1, 8, 0, 10.0),
               c(2, 900, 0, 10.0), c(3, 908, 0, 10.0)]
    G = _connected(centers)
    assert G.edges
    assert max(d for _, d in G.degree()) == 1
    assert alternating_repetition(G) is None


def test_alternating_repetition_defined_at_degree_two():
    centers = [c(0, 0, 0, 10.0, strength=1.0), c(1, 8, 0, 10.0, strength=0.1),
               c(2, -8, 0, 10.0, strength=2.0)]
    G = _connected(centers)
    assert max(d for _, d in G.degree()) >= 2
    assert alternating_repetition(G) is not None


# --- not_separateness: needs a connected component of at least two ---


def test_not_separateness_undefined_below_a_component_of_two():
    assert not_separateness(build_graph([])) is None
    assert not_separateness(build_graph([c(0, 0, 0, 5.0)])) is None
    # Many nodes, but the largest connected component still has one node:
    # algebraic connectivity is unmeasurable, not zero.
    G = build_graph([c(i, i * 5000, 0, 5.0) for i in range(4)])
    assert not G.edges
    assert not_separateness(G) is None


def test_not_separateness_defined_for_a_component_of_two():
    G = _connected([c(0, 0, 0, 10.0), c(1, 15, 0, 10.0)])
    assert G.edges
    assert not_separateness(G) is not None


# --------------------------------------------------------------------------
# compute_all / normalize_all contract
# --------------------------------------------------------------------------


def test_compute_all_is_all_none_for_empty_inputs():
    raw = compute_all(*_empty_inputs())
    assert set(raw) == set(KEYS)
    assert all(v is None for v in raw.values()), {
        k: v for k, v in raw.items() if v is not None
    }


def test_normalize_all_propagates_none_and_never_invents_ten():
    raw = compute_all(*_empty_inputs())
    norm = normalize_all(raw)
    assert set(norm) == set(KEYS)
    assert all(v is None for v in norm.values()), {
        k: v for k, v in norm.items() if v is not None
    }


def test_normalize_all_still_maps_a_genuine_zero_to_ten():
    """The undefined case must not be implemented by suppressing raw zeros.
    A raw 0 that was actually measured still normalises to 10."""
    raw = {k: 0.0 for k in KEYS}
    norm = normalize_all(raw)
    assert norm["levels_of_scale"] == pytest.approx(10.0)
    assert norm["boundaries"] == pytest.approx(10.0)
    assert norm["echoes"] == pytest.approx(10.0)
    assert norm["contrast"] == pytest.approx(0.0)


def test_normalize_all_mixed_none_and_numbers():
    raw = {k: 0.0 for k in KEYS}
    raw["echoes"] = None
    raw["not_separateness"] = None
    norm = normalize_all(raw)
    assert norm["echoes"] is None
    assert norm["not_separateness"] is None
    assert norm["levels_of_scale"] == pytest.approx(10.0)


# --------------------------------------------------------------------------
# End to end: a featureless canvas
# --------------------------------------------------------------------------


def test_flat_grey_scores_nothing():
    """Issue #15's headline case. Previously 7 of 15 at 10.0/10."""
    from centres.pipeline import analyze

    field, centers, G, _ = analyze(np.full((256, 256, 3), 128, np.uint8))
    assert len(centers) == 0
    norm = normalize_all(compute_all(field, centers, G))
    scored = {k: v for k, v in norm.items() if v is not None}
    assert not scored, f"a featureless canvas still scores: {scored}"


# --------------------------------------------------------------------------
# Output paths: CLI table, --json, GUI table
# --------------------------------------------------------------------------


def test_cli_bar_renders_none_distinctly():
    from centres.cli import _bar, _BAR_WIDTH

    assert len(_bar(None)) == _BAR_WIDTH
    # must not be mistakable for an empty bar (score 0) or a full one
    assert _bar(None) != _bar(0.0)
    assert _bar(None) != _bar(10.0)
    assert "█" not in _bar(None)


def test_cli_table_prints_undefined_without_crashing(capsys):
    from centres.cli import print_properties

    print_properties({k: None for k in KEYS})
    out = capsys.readouterr().out
    assert "undefined" in out
    assert "10.0" not in out
    for label in ("levels of scale", "boundaries", "echoes", "the void"):
        assert label in out


def test_cli_table_mixed_defined_and_undefined(capsys):
    from centres.cli import print_properties

    raw = {k: None for k in KEYS}
    raw["roughness"] = 0.5
    print_properties(raw)
    out = capsys.readouterr().out
    assert "10.0" in out  # roughness peaks at 0.5
    assert "undefined" in out


def test_json_emits_null_not_zero():
    from centres.cli import properties_json

    payload = properties_json(0, 0.0, {k: None for k in KEYS})
    text = json.dumps(payload)
    assert "null" in text
    reloaded = json.loads(text)
    for k in KEYS:
        assert reloaded["properties"][k] == {"score": None, "raw": None}


def test_json_reports_degree_of_life_not_structural_energy():
    """The reported quantity is L = -E (#28).

    ``analyze`` still returns the energy, because that is what ``evolve()``
    minimises, but nothing user-facing reports it: an energy whose minimum is
    the absence of structure is the wrong quantity to put in front of a reader.
    """
    from centres.cli import properties_json

    payload = properties_json(3, -0.3859, {k: None for k in KEYS})
    assert payload["degree_of_life"] == 0.3859
    assert "structural_energy" not in payload


def test_json_still_emits_numbers_when_defined():
    from centres.cli import properties_json

    raw = {k: 0.0 for k in KEYS}
    raw["echoes"] = None
    payload = json.loads(json.dumps(properties_json(3, 1.25, raw)))
    assert payload["properties"]["echoes"] == {"score": None, "raw": None}
    assert payload["properties"]["levels_of_scale"]["score"] == 10.0


def test_gui_table_renders_undefined_without_crashing():
    import os

    # Must be set before Qt is imported, or QtWidgets pulls in a display.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    # QtWidgets rather than PyQt6: the package can import while its native
    # libraries (libEGL and friends) are missing, and that must skip, not error.
    qtwidgets = pytest.importorskip(
        "PyQt6.QtWidgets", reason="GUI is an optional extra")
    QApplication = qtwidgets.QApplication

    from centres.gui import CentresMainWindow, _PROPERTY_LABELS

    app = QApplication.instance() or QApplication([])
    win = CentresMainWindow()
    try:
        win._update_table({k: None for k in KEYS})
        for row, (key, _) in enumerate(_PROPERTY_LABELS):
            assert win._table.item(row, 1).text() == "—"
            assert win._table.item(row, 2).text() == "—"
        # and a mixed case still shows the numbers it has
        raw = {k: None for k in KEYS}
        raw["roughness"] = 0.5
        win._update_table(raw)
        row = [k for k, _ in _PROPERTY_LABELS].index("roughness")
        assert win._table.item(row, 1).text() == "10.0"
    finally:
        win.close()
        app.processEvents()
