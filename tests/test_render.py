"""The stimulus renderer (#32) produces valid contact sheets.

A viewer, not a measurement, so this only checks that it runs and writes readable
PNGs of the expected shape — one tile per frame, captioned. It renders a single
short sweep plus the null controls to a temp dir to stay fast; ``python -m
audit.render`` renders them all.
"""

import cv2

from audit import render, stimuli


def test_null_controls_sheet_written(tmp_path):
    paths = render.render_null_controls(str(tmp_path))
    assert len(paths) == 1
    img = cv2.imread(paths[0])
    assert img is not None and img.ndim == 3
    # Wide enough for all five controls in one row of COLS columns.
    assert img.shape[1] >= render.THUMB * len(stimuli.NULL_CONTROLS) // render.COLS


def test_one_sweep_sheet_has_a_tile_per_frame(tmp_path, monkeypatch):
    # Render just one cheap sweep so the test stays fast.
    name = "dominance"
    monkeypatch.setattr(stimuli, "SWEEPS", {name: stimuli.SWEEPS[name]})
    paths = render.render_sweeps(str(tmp_path))
    assert len(paths) == 1 and paths[0].endswith(f"sweep_{name}.png")
    img = cv2.imread(paths[0])
    assert img is not None
    # The grid holds one captioned tile per sweep value; height covers every row.
    n = len(stimuli.SWEEPS[name][1])
    rows = (n + render.COLS - 1) // render.COLS
    cell = render.THUMB + render.CAPTION
    assert img.shape[0] >= render.TITLE + rows * cell


def test_render_all_writes_one_sheet_per_sweep_plus_controls(tmp_path, monkeypatch):
    # Two sweeps only, to keep it quick; checks the file set and naming.
    subset = {k: stimuli.SWEEPS[k] for k in ("dominance", "void_size")}
    monkeypatch.setattr(stimuli, "SWEEPS", subset)
    paths = render.render_all(str(tmp_path))
    names = sorted(p.split("/")[-1] for p in paths)
    assert names == ["null_controls.png", "sweep_dominance.png", "sweep_void_size.png"]
