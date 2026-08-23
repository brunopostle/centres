# Synthetic stimuli — contact sheets

These PNGs are the parametric generators in [`audit/stimuli.py`](../../audit/stimuli.py),
rendered for visual inspection (issue #32). Each is what the ground-truth sweeps in
`python -m audit` actually measure; see [AUDIT.md §12](../../AUDIT.md#12-ground-truth-sweeps--how-well-each-measure-tracks-its-own-property)
for how well each measure tracks it.

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

## Null controls

![Null controls: flat grey, white noise, smooth noise, random blobs, regular grid](null_controls.png)

## Sweeps, in AUDIT.md §12 order

### tonal_delta → contrast

![tonal_delta sweep, mapping to contrast](sweep_tonal_delta.png)

### interlock_depth → deep interlock

![interlock_depth sweep, mapping to deep interlock](sweep_interlock_depth.png)

### dominance → strong centres

![dominance sweep, mapping to strong centres](sweep_dominance.png)

### zone_width → gradients

![zone_width sweep, mapping to gradients](sweep_zone_width.png)

### ground_solidity → positive space

![ground_solidity sweep, mapping to positive space](sweep_ground_solidity.png)

### shape_vocabulary → echoes (↓)

![shape_vocabulary sweep, mapping to echoes](sweep_shape_vocabulary.png)

### bleed → not-separateness

![bleed sweep, mapping to not-separateness](sweep_bleed.png)

### element_kinds → simplicity

![element_kinds sweep, mapping to simplicity](sweep_element_kinds.png)

### border_band → boundaries (interior optimum at 0.3)

![border_band sweep, mapping to boundaries](sweep_border_band.png)

### scale_ratio → levels of scale (interior optimum at 3)

![scale_ratio sweep, mapping to levels of scale](sweep_scale_ratio.png)

### jitter → roughness

![jitter sweep, mapping to roughness](sweep_jitter.png)

### bilateral_asymmetry → local symmetries (↓)

![bilateral_asymmetry sweep, mapping to local symmetries](sweep_bilateral_asymmetry.png)

### motif_circularity → good shape

![motif_circularity sweep, mapping to good shape](sweep_motif_circularity.png)

### alternation → alternating repetition

![alternation sweep, mapping to alternating repetition](sweep_alternation.png)

### void_size → the void

![void_size sweep, mapping to the void](sweep_void_size.png)
