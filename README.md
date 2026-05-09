# centres

A computational implementation of Christopher Alexander's theory of *wholeness* from *The Nature of Order*. Given an image, it detects a multi-scale hierarchy of structural centres, builds a reinforcement graph between them, and computes a structural energy score measuring coherence.

The theory formalises Alexander's claim that living structure — in carpets, paintings, buildings, cities, and nature — arises from a recursive system of mutually reinforcing centres at many scales. See [THEORY.md](THEORY.md) for the full mathematical formulation.

## Installation

Command-line tool only:

```
pip install -e .
```

Desktop GUI (requires PyQt6):

```
pip install -e ".[gui]"
```

Pre-built Windows executables are available on the [releases page](../../releases).

## Usage

### Desktop GUI

Launch the graphical application:

```
centres-gui
```

Use **File → Open Image** (or the toolbar button) to load an image, then click **Analyse**. The structural field and centres are shown on the left; the 15 Alexander property scores appear in a colour-coded table on the right. Use **File → Save Visualisation** or **File → Export JSON** to save results.

### Analyse an image (command line)

```
centres analyse path/to/image.jpg
centres analyse path/to/image.jpg --no-display     # print result only
centres analyse path/to/image.jpg --save out.png   # save visualisation to file
centres analyse path/to/image.jpg --max-size 800   # override rescale limit
centres analyse path/to/image.jpg --json           # machine-readable JSON output
```

Output is a structural energy score and a table of all 15 of Alexander's structural properties, each normalised to a 0–10 wholeness score (10 = most present, consistently — no mixed directions to interpret). A block bar gives an immediate visual read of each score. The raw computed value is shown alongside.

The visualisation shows detected centres (cyan circles, radius proportional to scale), the parent hierarchy (white lines), and the reinforcement graph (lime lines, weight proportional to edge weight).

### Generate or refine a composition

Start from a blank canvas with random centres:
```
centres evolve
centres evolve --size 512 --n-centres 60 --iterations 500
centres evolve --save generated.png --no-display
```

Or seed the search from the centres detected in an existing image, letting annealing improve its structural energy while preserving approximate spatial character:
```
centres evolve path/to/image.jpg --iterations 200
centres evolve path/to/image.jpg --iterations 500 --save refined.png --no-display
```

Note: image-seeded runs are slower per iteration because they inherit all detected centres (potentially hundreds) rather than a small fixed number.

## Input images

The structural field is built using a distance transform from detected edges, so it works best on images where structure is expressed through clear boundaries: carpets, textiles, ornamental patterns, paintings, and natural scenes. Architectural plans must be in figure-ground form (solid filled regions) rather than as line drawings.

Large images are automatically downscaled to `--max-size` before processing. The blob detection sigma range (2–48 px) is calibrated for images around 1024 px; processing very large images at full resolution would both be slow and detect only very small-scale features.

## Theory

See [THEORY.md](THEORY.md) for a full account of the structural field, the centre hierarchy, the reinforcement graph, and the five-component energy functional. The document explains how Alexander's 15 structural properties emerge as consequences of energy minimisation rather than as separate design rules.

## License

Copyright (C) 2026 Bruno Postle

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

See [LICENSE](LICENSE) for the full licence text.
