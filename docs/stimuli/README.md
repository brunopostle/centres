# Synthetic stimuli — contact sheets

These PNGs are the parametric generators in [`audit/stimuli.py`](../../audit/stimuli.py),
rendered for visual inspection (issue #32). Each is what the ground-truth sweeps in
`python -m audit` actually measure.

- `null_controls.png` — the structureless and precisely-structured controls
  (`flat_grey`, `white_noise`, `smooth_noise`, `random_blobs`, `regular_grid`). A
  measure that reports high wholeness for these is not measuring wholeness.
- `sweep_<name>.png` — one sheet per generator, tiling every frame with its
  parameter underneath and titled `<generator> -> <property it isolates>`. Reading
  left to right shows the single structural quantity the sweep varies while holding
  the rest fixed.

Regenerate them (they are exactly this command's output, so a change to a generator
shows up as a changed image in the diff):

```
python -m audit.render            # writes here, docs/stimuli/
python -m audit.render <dir>      # or elsewhere
```
