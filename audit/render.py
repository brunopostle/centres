"""Render the synthetic stimuli as contact sheets, for visual inspection (#32).

``audit/stimuli.py`` builds every generator by construction, and
``tests/test_stimuli.py`` recovers each one's parameter from the rendered pixels —
but neither lets a person *look* at what the audit is measuring. This does: it
writes one PNG per sweep, tiling all its frames with the parameter under each, plus
one sheet of the null controls. Run it with

    python -m audit.render [output_dir]        # default: docs/stimuli/

and open the PNGs. It is a viewer, not part of the audit; nothing depends on its
output. The committed sheets in ``docs/stimuli/`` are exactly what this produces,
so a change to a generator shows up as a changed image in the diff.
"""

import os
import sys

import cv2
import numpy as np

from . import stimuli

#: Long side of each thumbnail in a sheet, in pixels. Small enough that a whole
#: sweep fits on screen and the noise frames do not bloat the PNG, large enough
#: that the motif stays legible.
THUMB = 200
#: Frames per row before wrapping.
COLS = 6
#: Maximum PNG compression: the noise stimuli are near-incompressible, so this
#: matters — it roughly halves the committed sheets.
_PNG = [cv2.IMWRITE_PNG_COMPRESSION, 9]
#: Height of the caption strip drawn under each thumbnail.
CAPTION = 22
#: Height of the title strip at the top of each sheet.
TITLE = 30
PAD = 6
_BG = 255  # sheet background: white


def _thumb(img):
    """Downscale a stimulus to a THUMB-tall thumbnail (they are square)."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.resize(img, (THUMB, THUMB), interpolation=cv2.INTER_AREA)


def _caption(text, width):
    """A white strip of the given width carrying centred black text."""
    strip = np.full((CAPTION, width, 3), _BG, np.uint8)
    scale = 0.45
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = max((width - tw) // 2, 1)
    y = (CAPTION + th) // 2
    cv2.putText(strip, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1,
                cv2.LINE_AA)
    return strip


def _tile(img, caption):
    """A captioned thumbnail: the thumbnail with its caption strip below."""
    thumb = _thumb(img)
    return np.vstack([thumb, _caption(caption, thumb.shape[1])])


def _sheet(title, tiles):
    """Tile captioned thumbnails into a grid under a title strip."""
    cell_h, cell_w = tiles[0].shape[:2]
    rows = (len(tiles) + COLS - 1) // COLS
    grid = np.full((rows * cell_h + (rows + 1) * PAD,
                    COLS * cell_w + (COLS + 1) * PAD, 3), _BG, np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, COLS)
        y = PAD + r * (cell_h + PAD)
        x = PAD + c * (cell_w + PAD)
        grid[y:y + cell_h, x:x + cell_w] = tile
    title_strip = np.full((TITLE, grid.shape[1], 3), _BG, np.uint8)
    cv2.putText(title_strip, title, (PAD, TITLE - 9), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([title_strip, grid])


def render_sweeps(outdir):
    """One contact sheet per sweep: every frame, captioned with its parameter."""
    written = []
    for name, (fn, values, target, _) in stimuli.SWEEPS.items():
        tiles = [_tile(fn(v), f"{name}={v:g}") for v in values]
        sheet = _sheet(f"{name}  ->  {target}", tiles)
        path = os.path.join(outdir, f"sweep_{name}.png")
        cv2.imwrite(path, sheet, _PNG)
        written.append(path)
    return written


def render_null_controls(outdir):
    """One sheet of the null controls (structureless or precisely-structured)."""
    tiles = [_tile(fn(), name) for name, fn in stimuli.NULL_CONTROLS.items()]
    sheet = _sheet("null controls  (high wholeness here is false)", tiles)
    path = os.path.join(outdir, "null_controls.png")
    cv2.imwrite(path, sheet, _PNG)
    return [path]


def render_all(outdir):
    os.makedirs(outdir, exist_ok=True)
    return render_null_controls(outdir) + render_sweeps(outdir)


def main():
    default = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "docs", "stimuli")
    outdir = sys.argv[1] if len(sys.argv) > 1 else default
    written = render_all(outdir)
    print(f"wrote {len(written)} contact sheets to {outdir}")
    for p in written:
        print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
