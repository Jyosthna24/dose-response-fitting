import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from doseresp import bootstrap_ec50, compare_ec50, fit_groups, fit_hill, hill4  # noqa: E402
from make_synthetic_data import CONCS, TRUTH, make  # noqa: E402


def test_hill_midpoint_is_ec50():
    assert hill4(np.log10(2.0), 0, 100, np.log10(2.0), 1.3) == pytest.approx(50)


def test_noise_free_fit_recovers_parameters():
    df = make(noise_sd=0.0)
    wt = df[df.condition == "WT"]
    f = fit_hill(wt.concentration, wt.response)
    for k, v in TRUTH["WT"].items():
        assert f.params[k] == pytest.approx(v, abs=1e-3)
    assert f.r2 == pytest.approx(1.0)


def test_ec50_ci_coverage_is_close_to_nominal():
    # across repeated simulated experiments the 95 % CI should contain the truth ~95 % of the time
    hits = []
    for seed in range(40):
        df = make(seed=seed)
        for cond, s in df.groupby("condition"):
            lo, hi = fit_hill(s.concentration, s.response).ec50_ci95
            hits.append(lo < 10 ** TRUTH[cond]["logEC50"] < hi)
    assert np.mean(hits) > 0.85


def test_inhibitor_gives_negative_hill():
    x = np.array(CONCS[1:] * 3)
    y = hill4(np.log10(x), 0, 100, np.log10(1.0), -1.0)
    assert fit_hill(x, y).params["hill"] == pytest.approx(-1.0, abs=1e-3)


def test_fixed_bottom_is_respected():
    df = make(seed=1)
    wt = df[df.condition == "WT"]
    vehicle = wt.loc[wt.concentration == 0, "response"].mean()
    f = fit_hill(wt.concentration, wt.response, fix_bottom=vehicle)
    assert f.params["bottom"] == vehicle and f.dof == f.n - 3


def test_too_few_concentrations_raises():
    with pytest.raises(ValueError):
        fit_hill([0.1, 1, 10], [1, 50, 99])


def test_f_test_detects_real_shift_but_not_a_fake_one():
    real = compare_ec50(make(seed=2), "concentration", "response", "condition")
    assert real["p_value"] < 0.001

    # under the null (identical curves) the false-positive rate should be near 5 %
    same = {"A": TRUTH["WT"], "B": dict(TRUTH["WT"])}
    p = [compare_ec50(make(seed=s, truth=same), "concentration", "response", "condition")["p_value"]
         for s in range(30)]
    assert np.mean(np.array(p) < 0.05) <= 0.2


def test_bootstrap_interval_is_sensible():
    df = make(seed=5)
    wt = df[df.condition == "WT"]
    b = bootstrap_ec50(wt.concentration, wt.response, n_boot=200, seed=0)
    assert b["n_boot_ok"] > 150
    assert b["EC50_boot_low"] < 0.3 < b["EC50_boot_high"]


def test_fit_groups_summary_table():
    s = fit_groups(make(seed=6), "concentration", "response", "condition")
    assert list(s.condition) == ["WT", "F508del"]
    assert s.set_index("condition").loc["WT", "EC50"] < s.set_index("condition").loc["F508del", "EC50"]
