"""Renormalisation-group running of the SM and of the SM + heavy neutrino.

Reproduces, from 2024-11_Scan_v1.2.nb:
  * ``RGESM``: SM running from m_t (``beta_sm``),
  * the threshold at M_N = w*YR/(2*sqrt 2) where the neutrino Yukawa t2 = Ynu^2 enters,
  * ``RGEn``: SM + N running from M_N to M_S (``beta_smn``), stopped when t2 or lambda
    reach 4*pi,
  * ``l1n``: one-loop correction to the effective quartic, lambda_eff = lambda + lambda_1,
  * ``step`` and ``lamsmooth``: the smoothed effective quartic used in the potential.
All scales are in GeV and x = ln(mu/GeV).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from ._betas import beta_sm, beta_smn
from .constants import M_TOP_POLE, PERTURBATIVITY_LIMIT, SM_INPUTS_AT_MT, SM_ORDER

LOOP = 1.0 / (16.0 * math.pi**2)
_RTOL = 1e-10
_ATOL = 1e-12
# Width of the region beyond [xM, xS] on which couplings are needed by the smooth steps
_EXTENSION = 0.25


def step(x):
    """Smooth step of the notebook: 0 for x < -0.05, 1 for x > 0.05, C-infinity in between.

    Mathematica: step[x_] = 1/4 (2 + (-2+4(10x))/Sqrt[(1-2(10x))^2] + ...), which
    simplifies to 1/(1 + exp(8z/(4z^2-1))) with z = 10x for |z| < 1/2.
    """
    z = 10.0 * np.asarray(x, dtype=float)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        inner = 1.0 / (1.0 + np.exp(8.0 * z / (4.0 * z**2 - 1.0)))
    out = np.where(z >= 0.5, 1.0, np.where(z <= -0.5, 0.0, inner))
    return out if out.ndim else float(out)


def lambda_one_loop(g1, g2, yt, t2):
    """``l1n`` of the notebook: one-loop correction to the effective quartic.

    In the notebook Hn = g1 (h), Gn = g2 (g), Yn = y_t, T2n = Ynu^2.
    """
    pref = 4.0 / (2048.0 * math.pi**2)
    gz2 = g2**2 + g1**2
    return pref * (
        -5.0 * g1**4
        + 6.0 * gz2**2 * np.log(gz2 / 4.0)
        - 10.0 * g2**2 * g1**2
        - 15.0 * g2**4
        + 12.0 * g2**4 * np.log(g2**2 / 4.0)
        + 144.0 * yt**4
        - 96.0 * yt**4 * np.log(yt**2 / 2.0)
        + 144.0 / 3.0 * t2**2
        - 32.0 * t2**2 * np.log((t2 + 1e-9) / 2.0)
    )


def run_sm(x_target: float, x_start: float | None = None, y_start=None):
    """SM couplings at x_target, running from m_t (or from a given state)."""
    if x_start is None:
        x_start = math.log(M_TOP_POLE)
        y_start = [SM_INPUTS_AT_MT[k] for k in SM_ORDER]
    if abs(x_target - x_start) < 1e-14:
        return np.array(y_start, dtype=float)
    sol = solve_ivp(beta_sm, (x_start, x_target), y_start, method="DOP853", rtol=_RTOL, atol=_ATOL)
    if not sol.success:
        raise RuntimeError(f"SM running failed: {sol.message}")
    return sol.y[:, -1]


@dataclass
class Running:
    """Result of the running between M_N and M_S."""

    xM: float
    xS: float
    xmax: float  # = xS, or the scale where perturbativity was lost
    perturbative: bool
    _pieces: list = field(repr=False)  # list of (xlo, xhi, OdeSolution)
    lam_eff_spline: CubicSpline | None = field(default=None, repr=False)
    lamsmooth_spline: CubicSpline | None = field(default=None, repr=False)

    def state(self, x):
        """Couplings (L, y, b, t, g, h, s, t2) at x (array over x if x is an array)."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.empty((8, x.size))
        for i, xi in enumerate(x):
            for lo, hi, sol in self._pieces:
                if lo - 1e-12 <= xi <= hi + 1e-12:
                    out[:, i] = sol(xi)
                    break
            else:
                raise ValueError(f"x={xi} outside the solved range")
        return out

    def lam(self, x):
        return self.state(x)[0]

    def lam_eff(self, x):
        """lambda + lambda_1 (``ln[x] + l1n[x]`` in the notebook)."""
        st = self.state(x)
        L, y, b, t, g, h, s, t2 = st
        return L + lambda_one_loop(h, g, y, t2)

    def beta_lam(self, x):
        """d lambda / d ln mu (``ln'[x]`` in the notebook)."""
        return beta_smn(x, self.state(x)[:, 0])[0]


def run_smn(Ms: float, MN: float, ynu: float):
    """Run SM -> M_N, apply the neutrino threshold and run SM+N -> M_S.

    Returns a :class:`Running`. ``perturbative`` is False if t2 or lambda reached 4*pi
    below M_S (then ``xmax`` is that scale, as stored in the notebook's ``nonpertlist``).
    """
    xS = math.log(Ms)
    xM = math.log(MN)
    if xM >= xS:
        raise ValueError(f"M_N = {MN:.4g} GeV must lie below M_S = {Ms:.4g} GeV")
    t2in = ynu**2
    L, y, b, t, g, h, s = run_sm(xM)
    y0 = [
        L * (1.0 + LOOP * t2in) + LOOP * t2in**2,
        y * (1.0 + LOOP * t2in),
        b,
        t,
        g,
        h,
        s,
        t2in,
    ]

    def ev_t2(x, Y):
        return Y[7] - PERTURBATIVITY_LIMIT

    def ev_L(x, Y):
        return Y[0] - PERTURBATIVITY_LIMIT

    for ev in (ev_t2, ev_L):
        ev.terminal = True
        ev.direction = 1

    # As Mathematica's WhenEvent, the 4 pi cut fires on an upward crossing only: a point
    # that starts above it at M_N (e.g. Ynu > sqrt(4 pi)) is not flagged, as in the notebook.
    main = solve_ivp(
        beta_smn, (xM, xS), y0, method="DOP853", rtol=_RTOL, atol=_ATOL,
        dense_output=True, events=(ev_t2, ev_L),
    )
    xmax = float(main.t[-1])
    # status -1: the couplings blew up (Landau pole) before the 4 pi events fired
    perturbative = main.status == 0
    pieces = [(xM, xmax, main.sol)]
    if perturbative:
        # Mathematica extrapolates the interpolating functions slightly outside
        # [xM, xS]; we integrate the same equations a little further instead.
        up = solve_ivp(beta_smn, (xS, xS + _EXTENSION), main.y[:, -1], method="DOP853",
                       rtol=_RTOL, atol=_ATOL, dense_output=True)
        down = solve_ivp(beta_smn, (xM, xM - _EXTENSION), y0, method="DOP853",
                         rtol=_RTOL, atol=_ATOL, dense_output=True)
        if up.success:
            pieces.append((xS, float(up.t[-1]), up.sol))
        if down.success:
            pieces.append((float(down.t[-1]), xM, down.sol))
    return Running(xM=xM, xS=xS, xmax=xmax, perturbative=perturbative, _pieces=pieces)


def build_lamsmooth(run: Running, n: int = 6000):
    """Spline of ``lamsmooth[x]`` = smoothed lambda_eff, frozen below M_N and above M_S."""
    lo = run.xM - _EXTENSION
    hi = run.xS + _EXTENSION
    lo_avail = min(p[0] for p in run._pieces)
    hi_avail = max(p[1] for p in run._pieces)
    lo, hi = max(lo, lo_avail), min(hi, hi_avail)
    xs = np.linspace(lo, hi, n)
    le = run.lam_eff(xs)
    leM = float(run.lam_eff(run.xM)[0])
    leS = float(run.lam_eff(run.xS)[0])
    ls = leM * step(run.xM - xs) + le * step(xs - run.xM) * step(run.xS - xs) + leS * step(xs - run.xS)
    run.lam_eff_spline = CubicSpline(xs, le)
    run.lamsmooth_spline = CubicSpline(xs, ls)
    run._lamsmooth_bounds = (lo, hi, leM, leS)
    return run.lamsmooth_spline


def lamsmooth(run: Running, x):
    """Evaluate the smooth effective quartic and its x-derivative (vectorised)."""
    lo, hi, leM, leS = run._lamsmooth_bounds
    x = np.asarray(x, dtype=float)
    val = np.where(x <= lo, leM, np.where(x >= hi, leS, run.lamsmooth_spline(np.clip(x, lo, hi))))
    der = np.where((x <= lo) | (x >= hi), 0.0, run.lamsmooth_spline(np.clip(x, lo, hi), 1))
    return val, der


def instability_scale(run: Running, n_grid: int = 2000):
    """x_in where lambda_eff = 0 (``FindRoot[ln[x]+l1n[x]==0, {x, xmax-1}]``).

    Returns the root in [xM, xmax] closest to xmax - 1, or None if there is none.
    """
    xs = np.linspace(run.xM, run.xmax, n_grid)
    vals = run.lam_eff(xs)
    idx = np.nonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) <= 0)[0]
    if idx.size == 0:
        return None
    target = run.xmax - 1.0
    roots = []
    for i in idx:
        a, b_ = xs[i], xs[i + 1]
        if vals[i] == 0.0:
            roots.append(a)
            continue
        roots.append(brentq(lambda z: float(run.lam_eff(z)[0]), a, b_, xtol=1e-13))
    return min(roots, key=lambda r: abs(r - target))
