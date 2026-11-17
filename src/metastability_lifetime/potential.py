"""Tree-level potential, tunnelling path and effective potential along the path.

All fields are in units of the singlet vev w (notebook variables Hw = H/w, Sw = S/w,
hw = h/w) and the potential is in units of w^4. From the notebook:

  Vfullw[Hw,Sw] = -kappa Hw^2 - (Ms2/w^2) Sw^2 + kappa Hw^2 Sw^2 + lamH Hw^4
                  + Ms2/(2 w^2) Sw^4

The path is the steepest-descent curve from the false vacuum (Hw, Sw) = (0, 1) to the
true vacuum (v/w, 0) (the notebook's ``Ridgelist``/``VFGw``); its arc length (times
sqrt 2, i.e. canonical normalisation) is the tunnelling coordinate hw. Along it

  Veffw[hw] = Vraw[hw] + 1/4 (lamsmooth[ln(hw w)] - lambda_eff(M_S)) hw^4,

i.e. the tree potential of the full theory plus the running of the SM quartic below M_S.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar

from .running import Running, lamsmooth

SQRT2 = math.sqrt(2.0)


class NoDescentPath(Exception):
    """The false vacuum is a minimum of the tree potential along the singlet valley
    (lamH - kappa^2 w^2 / (2 Ms2) >= 0), so there is no steepest-descent path."""


@dataclass
class TreePotential:
    kappa: float
    Ms2: float
    w: float
    lamH: float

    @property
    def m(self):  # Ms2 / w^2
        return self.Ms2 / self.w**2

    def V(self, H, S):
        k, m, l = self.kappa, self.m, self.lamH
        return -k * H**2 - m * S**2 + k * H**2 * S**2 + l * H**4 + 0.5 * m * S**4

    def grad(self, H, S):
        k, m, l = self.kappa, self.m, self.lamH
        dH = -2 * k * H + 2 * k * H * S**2 + 4 * l * H**3
        dS = -2 * m * S + 2 * k * H**2 * S + 2 * m * S**3
        return dH, dS


def steepest_descent_path(tree: TreePotential, vw: float, n_samples: int = 20000):
    """Exact steepest-descent path from the false vacuum (0, 1) to the true vacuum (vw, 0).

    This is the dh -> 0 limit of the notebook's ``Ridgelist`` construction (normalised
    gradient flow with explicit Euler steps dh = 1e-3, then a degree-80 polynomial fit).
    With finite dh that flow zig-zags across the steep singlet direction, which adds
    spurious potential energy of order m_S^2 dh^2 along the path; near the flat false
    vacuum this is comparable to the barrier and shifts S_E by up to ~10%.

    Returns arrays (hw, H, D): arc-length coordinate (times sqrt 2, i.e. the canonical
    field in units of w) and the path points, with D = 1 - S^2.
    """
    from scipy.integrate import solve_ivp

    m, k, lam = tree.m, tree.kappa, tree.lamH
    lam_valley = lam - k * k / (2.0 * m)  # quartic along the valley floor near (0, 1)
    if lam_valley >= 0.0:
        raise NoDescentPath(f"tree-level valley quartic {lam_valley:.4g} >= 0")
    # Along the steepest-descent path H grows monotonically up to vw, so we use
    # z = ln H as the independent variable with state (D = 1 - S^2, arc length):
    #   dH/dtau = -dV/dH = 2 k H D - 4 lam H^3        (> 0 for H < vw)
    #   dD/dtau = 2 S dV/dS = 4 (1 - D)(k H^2 - m D)
    # D (instead of S) avoids the cancellation in 1 - S^2 ~ 1e-13 near the flat
    # false vacuum, and z avoids the huge time scales of the gradient flow.
    H0 = 1e-7 * max(vw, 1.0)
    D0 = k * H0**2 / m  # start on the valley floor dV/dS = 0
    z_end = math.log(vw * (1.0 - 1e-9))

    def rhs(z, Y):
        D = Y[0]
        H = math.exp(z)
        S = math.sqrt(max(1.0 - D, 0.0))
        gH = -2.0 * k * H * D + 4.0 * lam * H**3
        gS = 2.0 * S * (k * H * H - m * D)
        Hdot = -gH
        if Hdot <= 0.0:
            Hdot = 1e-300
        dz = Hdot / H  # dz/dtau
        return [4.0 * (1.0 - D) * (k * H * H - m * D) / dz, SQRT2 * math.hypot(gH, gS) / dz]

    with np.errstate(over="ignore", invalid="ignore"):
        sol = solve_ivp(rhs, (math.log(H0), z_end), [D0, 0.0], method="Radau", rtol=1e-11,
                        atol=[1e-20, 1e-14], dense_output=True)
    if sol.status != 0:
        raise RuntimeError(f"steepest-descent path failed: {sol.message}")
    # sample uniformly in arc length (plus the solver nodes)
    s_steps = sol.y[1]
    s_targets = np.linspace(0.0, s_steps[-1], n_samples)
    fine = np.unique(np.concatenate([sol.t, np.interp(s_targets, s_steps, sol.t)]))
    D, sarc = sol.sol(fine)
    H = np.exp(fine)
    S_last = math.sqrt(max(1.0 - D[-1], 0.0))
    sarc = sarc + SQRT2 * math.hypot(H0, D0 / 2.0)  # distance from the exact origin (0, 1)
    # close the path at the true vacuum (vw, 0), i.e. D = 1
    H = np.append(H, vw)
    sarc = np.append(sarc, sarc[-1] + SQRT2 * math.hypot(vw - H[-2], S_last))
    D = np.append(D, 1.0)
    keep = np.concatenate([[True], np.diff(sarc) > 1e-9 * max(sarc[-1], 1.0)])
    hw = np.concatenate([[0.0], sarc[keep]])
    return hw, np.concatenate([[0.0], H[keep]]), np.concatenate([[0.0], D[keep]])


@dataclass
class PathPotential:
    """Effective potential along the tunnelling path, in units of w^4."""

    tree: TreePotential
    run: Running
    w: float
    lam_eff_S: float  # lambda_eff(M_S) = lammatch + l1n[xS]
    hw_path: np.ndarray = field(repr=False)
    H_path: np.ndarray = field(repr=False)
    D_path: np.ndarray = field(repr=False)  # D = 1 - S^2

    def __post_init__(self):
        hw, H, D = self.hw_path, self.H_path, self.D_path
        L = hw[-1]
        # Same symmetric extension as the notebook (UnitVListLong): H odd and S (hence D)
        # even around hw = 0, H odd and S even around the end of the path.
        x = np.concatenate([-hw[:0:-1], hw, 2 * L - hw[-2::-1]])
        Hx = np.concatenate([-H[:0:-1], H, 2 * H[-1] - H[-2::-1]])
        Dx = np.concatenate([D[:0:-1], D, D[-2::-1]])
        self.hw_end = L
        self._H = CubicSpline(x, Hx)
        self._D = CubicSpline(x, Dx)

    def fields(self, hw):
        """(H/w, S/w) along the path."""
        return self._H(hw), np.sqrt(np.clip(1.0 - self._D(hw), 0.0, None))

    def Vraw(self, hw):
        """Vfullw(H, S) - Vfullw(0, 1) along the path, written without cancellations:
        -kappa H^2 D + lamH H^4 + (m/2) D^2 with D = 1 - S^2."""
        t = self.tree
        H, D = self._H(hw), self._D(hw)
        return -t.kappa * H * H * D + t.lamH * H**4 + 0.5 * t.m * D * D

    def dVraw(self, hw):
        t = self.tree
        H, D = self._H(hw), self._D(hw)
        dH = -2.0 * t.kappa * H * D + 4.0 * t.lamH * H**3
        dD = -t.kappa * H * H + t.m * D
        return dH * self._H(hw, 1) + dD * self._D(hw, 1)

    def _dlam(self, hw):
        hw = np.asarray(hw, dtype=float)
        safe = np.where(hw > 0, hw, 1.0)
        x = np.log(safe * self.w)
        ls, dls = lamsmooth(self.run, x)
        return ls - self.lam_eff_S, dls

    def V(self, hw):
        """``Veffw[hw]``."""
        hw = np.asarray(hw, dtype=float)
        dl, _ = self._dlam(hw)
        out = self.Vraw(hw) + 0.25 * dl * hw**4
        return np.where(hw > 0, out, self.Vraw(hw)) if out.ndim else float(out)

    def dV(self, hw):
        hw = np.asarray(hw, dtype=float)
        dl, ddl = self._dlam(hw)
        out = self.dVraw(hw) + dl * hw**3 + 0.25 * ddl * hw**3
        return out if out.ndim else float(out)

    def true_vacuum(self):
        """``hwmax``: minimum of Vraw near the end of the path."""
        L = self.hw_end
        grid = np.linspace(0.5 * L, 1.5 * L, 4001)
        vals = self.Vraw(grid)
        i = int(np.argmin(vals))
        lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, grid.size - 1)]
        res = minimize_scalar(lambda z: float(self.Vraw(z)), bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-12 * max(L, 1.0)})
        x0 = float(res.x)
        # polish: the minimum is a zero of V' (needed to ~1e-15 for the shooting)
        a, b = x0 - 1e-6 * L, x0 + 1e-6 * L
        fa, fb = float(self.dV(a)), float(self.dV(b))
        if fa < 0 < fb:
            return float(brentq(lambda z: float(self.dV(z)), a, b, xtol=1e-16 * L, rtol=1e-15))
        return x0

    def barrier_top(self, muinw: float, hwmax: float):
        """``hwmin``: the barrier, i.e. the local maximum of Veffw reached from
        hw = e^(-1/4) muinw, with hw > muinw/500 (``FindArgMax`` of the notebook).

        If the climb runs into hw = muinw/500 that value is returned, which the caller
        classifies as "no barrier".
        """
        lo = muinw / 500.0
        hi = hwmax
        grid = np.geomspace(lo, hi, 6001)
        vals = self.V(grid)
        i = int(np.clip(np.searchsorted(grid, math.exp(-0.25) * muinw), 0, grid.size - 1))
        while True:  # hill-climb on the grid
            if i > 0 and vals[i - 1] > vals[i]:
                i -= 1
            elif i < grid.size - 1 and vals[i + 1] > vals[i]:
                i += 1
            else:
                break
        if i == 0:
            return float(lo)
        a, b = grid[i - 1], grid[min(i + 1, grid.size - 1)]
        res = minimize_scalar(lambda z: -float(self.V(z)), bounds=(a, b), method="bounded",
                              options={"xatol": 1e-14 * hi})
        return float(res.x)
