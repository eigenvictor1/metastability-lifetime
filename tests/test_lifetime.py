"""Regression tests against values stored in 2024-11_Scan_v1.2.nb (Reslist)."""

import math

import numpy as np
import pytest

from metastability_lifetime import compute_point, compute_ynu_scan, ynu_values
from metastability_lifetime.running import instability_scale, run_smn, step

# Reslist entry: {Ms, w, kappa, YR, Ynu, mb2, muin, log10tau, SElist, exp}
LONG_LIVED = dict(Ms=1e5, w=316227.7660168379, kappa=0.1, YR=0.30051128320709547, Ynu=3.8,
                  mb2=5.425722373405675e8, mu_in=47515.772590134584, log10_tau=1026.1748457027222,
                  SE=2792.804416598328)
SHORT_LIVED = dict(Ms=206913.80811147895, w=654318.9129712967, kappa=0.16237767391887217,
                   YR=0.023582922569987983, Ynu=1.2000000000000002, mb2=2.8764445471271586e8,
                   mu_in=166245.53854871631, SE=417.71729140004555)


def test_step_function():
    assert step(-1.0) == 0.0 and step(1.0) == 1.0
    assert step(0.0) == pytest.approx(0.5)
    x = np.linspace(-0.049, 0.049, 50)
    assert np.all(np.diff(step(x)) > 0)


@pytest.mark.parametrize("p", [LONG_LIVED, SHORT_LIVED])
def test_running_matches_notebook(p):
    MN = p["w"] * p["YR"] / (2 * math.sqrt(2))
    run = run_smn(p["Ms"], MN, p["Ynu"])
    xin = instability_scale(run)
    mu_in = math.exp(xin)
    mb2 = abs(run.beta_lam(xin)) * math.exp(-1.5) * mu_in**2
    assert mu_in == pytest.approx(p["mu_in"], rel=1e-5)
    assert mb2 == pytest.approx(p["mb2"], rel=1e-5)


def test_long_lived_point():
    p = LONG_LIVED
    r = compute_point(p["Ms"], p["w"], p["kappa"], p["YR"], p["Ynu"])
    assert r.status == "ok"
    assert r.converged
    assert r.action == pytest.approx(p["SE"], rel=2e-3)
    assert r.log10_tau == pytest.approx(p["log10_tau"], abs=1.0)


def test_short_lived_point_is_nope():
    p = SHORT_LIVED
    r = compute_point(p["Ms"], p["w"], p["kappa"], p["YR"], p["Ynu"])
    assert r.status == "nope"
    # exact steepest-descent path: S_E = 398.4; the notebook's Euler path gave 417.7
    assert r.action == pytest.approx(398.36, rel=2e-3)


def test_nonperturbative_point():
    # stored in the notebook's nonpertlist-type region: large kappa and Ynu
    r = compute_point(206913.80811147895, 654318.9129712967, 0.8858667904100826, 0.07019106558906348, 3.4000000000000004)
    assert r.status == "non-perturbative"
    assert r.lam_nonpert_scale < 1e5


def test_ynu_iterator_and_skip():
    assert ynu_values(1.2, 1.5, 0.1) == pytest.approx([1.2, 1.3, 1.4, 1.5])
    p = SHORT_LIVED
    res = compute_ynu_scan(p["Ms"], p["w"], p["kappa"], p["YR"], [1.2, 1.3])
    assert res[0].status == "nope"
    assert res[1].status == "skipped"
