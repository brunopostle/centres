"""The sweep-verdict logic must not mislabel correctly-tracking measures (#33).

Two classes of bug are guarded here, both of which flagged a measure that tracks
its ground truth as failing:

  * a measure that tracks *negatively* by design (echoes, local symmetries) was
    reported "does not track" because the flag assumed every measure rises;
  * an optimum measure that *peaks* at its ideal (boundaries, levels of scale
    after the #22 redefinition) was checked for a *minimum* at the ideal, so its
    extremum landed at a sweep end and it was flagged "not where it claims".

These tests drive the verdict functions on synthetic sweeps with a known shape.
"""

import numpy as np

from audit.run import _monotone_verdict, _optimum_verdict


def _pairs(xs, ys):
    return list(zip(xs, ys))


XS = list(np.linspace(0.0, 1.0, 12))


def test_monotone_rising_tracks(capsys):
    rho = _monotone_verdict("g", "m", _pairs(XS, XS), 0, expect=+1)
    assert rho > 0.9
    out = capsys.readouterr().out
    assert "does not track" not in out and "BACKWARDS" not in out
    assert "↑" in out


def test_monotone_falling_by_design_tracks(capsys):
    # echoes-like: the measure should fall with the parameter, and does.
    rho = _monotone_verdict("g", "echoes", _pairs(XS, [-x for x in XS]), 0, expect=-1)
    assert rho < -0.9
    out = capsys.readouterr().out
    assert "does not track" not in out and "BACKWARDS" not in out
    assert "↓" in out


def test_monotone_wrong_direction_is_backwards(capsys):
    # should rise (+1) but falls: a real failure, reported as backwards.
    _monotone_verdict("g", "m", _pairs(XS, [-x for x in XS]), 0, expect=+1)
    assert "runs BACKWARDS" in capsys.readouterr().out


def test_monotone_flat_does_not_track(capsys):
    rng = np.random.default_rng(0)
    ys = list(rng.normal(0, 1, len(XS)))  # no relationship to the parameter
    rho = _monotone_verdict("g", "m", _pairs(XS, ys), 0, expect=+1)
    assert abs(rho) < 0.5
    assert "does not track its own ground truth" in capsys.readouterr().out


def test_optimum_peak_at_claimed_is_clean(capsys):
    # boundaries-like: exp(-|log(ratio/target)|) peaks at the claimed value.
    ys = [np.exp(-abs(np.log((x + 1e-3) / 0.3))) for x in XS]
    at = _optimum_verdict("g", "boundaries", _pairs(XS, ys), 0.3, peak=True, dropped=0)
    assert abs(at - 0.3) < 0.1
    out = capsys.readouterr().out
    assert "maximum at" in out
    assert "is not where the measure claims" not in out
    assert "wants + then -" in out


def test_optimum_peak_far_from_claimed_is_flagged(capsys):
    ys = [np.exp(-abs(np.log((x + 1e-3) / 0.8))) for x in XS]  # peaks near 0.8, not 0.3
    _optimum_verdict("g", "m", _pairs(XS, ys), 0.3, peak=True, dropped=0)
    assert "maximum is not where the measure claims it is" in capsys.readouterr().out


def test_optimum_valley_uses_minimum(capsys):
    # a raw squared-deviation measure: minimal at the ideal.
    ys = [(x - 0.3) ** 2 for x in XS]
    at = _optimum_verdict("g", "m", _pairs(XS, ys), 0.3, peak=False, dropped=0)
    assert abs(at - 0.3) < 0.1
    out = capsys.readouterr().out
    assert "minimum at" in out and "wants - then +" in out
