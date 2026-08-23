"""The audit's headline thresholds, as real tests (#17).

The rest of the suite mostly asserts that each formula computes itself, so it would
keep passing under the failures AUDIT.md documents. These tests assert the
properties the audit actually checks — the ones a regression in the *pipeline* or
the *degree of life* would break — so they fail here rather than waiting for a
65-minute `python -m audit` run:

  - the reported degree of life ranks composed structure above dense noise (#29),
    and a mechanical grid below real art (the interior-optimum, §3/§17);
  - a blank canvas is not alive, and noise is crushed to ~0 by the wholeness gate;
  - the reported score is invariant under isometries, which cannot change structure.

They run at the audit's own regime — synthetic controls at their native 1024 px
canvas, the corpus image capped at 1024 — because that is the resolution the #29
separation is validated at (at 256/512 the front end resolves noise differently; see
#9). That makes them slower than the formula tests, so they carry ``@pytest.mark.slow``
and can be skipped with ``pytest -m 'not slow'``; CI runs the full suite.

Companion guards live elsewhere and are not duplicated here: the centre-count
confound (B1/#14) in ``test_energy``, the undefined-not-10/10 null contract
(B2/#15) in ``test_undefined``, the propagation fixed point (A5/#12) in
``test_graph``, the #29 candidate OR-rule in ``test_discrimination``, and edge/field
equivariance in ``test_equivariance``.
"""

import os

import cv2
import numpy as np
import pytest

from centres.pipeline import analyze
from centres.properties import compute_all, normalize_all
from centres.transforms import identity, mirror, rot90
from audit import stimuli

IMAGES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")


def _degree_of_life(img):
    _, _, _, energy = analyze(img)
    return -energy


def _norm(img):
    field, centers, G, energy = analyze(img)
    return normalize_all(compute_all(field, centers, G, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))


@pytest.fixture(scope="module")
def ardabil():
    img = cv2.imread(os.path.join(IMAGES, "ardabil.jpg"))
    assert img is not None, "corpus image images/ardabil.jpg is required for these tests"
    h, w = img.shape[:2]
    s = min(1024 / max(h, w), 1.0)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1.0 else img


# --- #29: the degree of life ranks structure above noise ---------------------

@pytest.mark.slow
def test_degree_of_life_ranks_artwork_above_noise(ardabil):
    """The central result (#29): the reported score puts a real artwork well above
    dense noise. The wholeness gate crushes noise — which has local structure but no
    single whole — toward zero, so the margin is wide (measured ~0.20 vs <=0.04)."""
    art = _degree_of_life(ardabil)
    noise = [_degree_of_life(g(seed=0)) for g in
             (stimuli.white_noise, stimuli.smooth_noise, stimuli.random_blobs)]
    assert art > 0.15
    assert max(noise) < 0.10
    assert art > 2 * max(noise)


@pytest.mark.slow
def test_mechanical_grid_scores_below_real_art(ardabil):
    """The interior-optimum guard (§3/§17): a regular grid is reinforced but not a
    single whole, so the gate must not rank it as alive as composed artwork."""
    assert _degree_of_life(stimuli.regular_grid()) < _degree_of_life(ardabil)


@pytest.mark.slow
def test_noise_is_crushed_toward_zero():
    """Dense noise scores near zero on the gated degree of life (not above carpets,
    as it did before #29)."""
    for gen in (stimuli.white_noise, stimuli.smooth_noise, stimuli.random_blobs):
        assert _degree_of_life(gen(seed=0)) < 0.10


def test_blank_canvas_is_not_alive():
    """A featureless field has no structure: exactly zero, and every property
    undefined rather than a false perfect (the #28/#15 contract at score level).
    Fast — a flat field detects nothing — so it runs in the default suite."""
    img = stimuli.flat_grey(n=256)
    field, centers, G, energy = analyze(img)
    assert -energy == 0.0
    assert all(v is None for v in normalize_all(
        compute_all(field, centers, G, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))).values())


# --- isometry: a mirror or a quarter-turn cannot change structure ------------

@pytest.mark.slow
def test_degree_of_life_invariant_under_isometry(ardabil):
    """Mirror and rot90 are exact isometries, so the reported degree of life must
    barely move; the pipeline is symmetrised over the dihedral group for this
    (measured spread ~0.01)."""
    lives = [_degree_of_life(fn(ardabil)) for fn in (identity, mirror, rot90)]
    assert max(lives) - min(lives) < 0.05


@pytest.mark.slow
def test_most_properties_invariant_under_isometry(ardabil):
    """The 0–10 property scores are near-invariant under isometry too. levels_of_scale
    is a known looser case (it reads the LoG scale lattice, ~1 pt), so the guard is on
    the median spread across the fifteen, which catches a broad regression."""
    per = [_norm(fn(ardabil)) for fn in (identity, mirror, rot90)]
    spreads = []
    for key in per[0]:
        vals = [p[key] for p in per if p[key] is not None]
        if len(vals) == 3:
            spreads.append(max(vals) - min(vals))
    assert float(np.median(spreads)) < 0.5
