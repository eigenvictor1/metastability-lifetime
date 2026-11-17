"""O(4)-symmetric bounce by overshoot/undershoot shooting.

Solves  u''(r) + 3/r u'(r) = V'(u)  (fields in units of w, radius in units of 1/w)
for a release point u0 between the barrier top (hwmin) and the true vacuum (hwmax),
with the false vacuum at u = 0, exactly as in the notebook's "(*shooting*)" block:

  * overshoot:  u(r) drops to 1/inf  (u0 too close to the true vacuum),
  * undershoot: u'(r) turns positive (u0 too far from it),

and the Euclidean action S_E = 2 pi^2 int r^3 (u'^2/2 + V(u)) dr up to the
zero crossing / turning point.

Differences from the notebook (numerics only):
  * the decimal digit-by-digit scan of u0 is replaced by a bisection in
    ln(hwmax - u0), which needs far fewer integrations for thin-wall bounces;
  * the integration starts from the regular series u = u0 + V'(u0) r^2/8;
  * the action is integrated together with the ODE instead of with NIntegrate.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from .constants import Settings

TWO_PI2 = 2.0 * math.pi**2


class TabulatedPotential:
    """Fast scalar evaluation of V and V' on [-hi, hi] by piecewise cubic Hermite
    interpolation of (V, V') on a dense uniform grid. V is even in the field (H is odd
    and S even along the path), so negative u is mapped to |u|. Beyond hi the exact
    (slow) functions are used."""

    def __init__(self, V, dV, hi: float, n: int = 400_001):
        x = np.linspace(0.0, hi, n)
        f = np.asarray(V(x), dtype=float)
        d = np.asarray(dV(x), dtype=float)
        hstep = x[1] - x[0]
        # cubic on each cell in t = (u - x_i)/h:  f + h d t + c2 t^2 + c3 t^3
        df = np.diff(f)
        c2 = 3 * df - hstep * (2 * d[:-1] + d[1:])
        c3 = -2 * df + hstep * (d[:-1] + d[1:])
        self._f, self._d = f[:-1].tolist(), d[:-1].tolist()
        self._c2, self._c3 = c2.tolist(), c3.tolist()
        self._h = hstep
        self._n = n - 1
        self._hi = hi
        self._V, self._dV = V, dV

    def _cell(self, u):
        z = u / self._h
        i = int(z)
        if i >= self._n:
            i = self._n - 1
        return i, z - i

    def V(self, u):
        if u < 0.0:
            u = -u
        if u > self._hi:
            return float(self._V(u))
        i, t = self._cell(u)
        return self._f[i] + t * (self._h * self._d[i] + t * (self._c2[i] + t * self._c3[i]))

    def dV(self, u):
        sign = 1.0
        if u < 0.0:
            u, sign = -u, -1.0
        if u > self._hi:
            return sign * float(self._dV(u))
        i, t = self._cell(u)
        return sign * (self._d[i] + t * (2.0 * self._c2[i] + 3.0 * t * self._c3[i]) / self._h)


@dataclass
class Shot:
    u0: float
    outcome: str  # "overshoot", "undershoot" or "unresolved"
    r_end: float
    action_direct: float  # 2 pi^2 int r^3 (u'^2/2 + V)
    action_virial: float  # pi^2/2 int r^3 u'^2  (equals the action for an exact bounce)

    @property
    def action(self):
        """The direct action, unless it disagrees with the virial one by more than 1%.

        That happens when V_eff is slightly negative at very small field (frozen quartic
        below M_N plus the threshold shift): the long massless tail of a near-critical
        shot then picks up int r^3 V with V < 0 out to huge r, while the kinetic
        (virial) form is insensitive to it.
        """
        d, v = self.action_direct, self.action_virial
        if math.isfinite(d) and math.isfinite(v) and abs(d - v) > 0.01 * abs(v):
            return v
        return d


@dataclass
class Bounce:
    action: float
    action_direct: float
    action_virial: float
    u0: float  # release point of the last undershoot (``u0min`` of the notebook)
    u0_overshoot: float
    converged: bool
    shots: list = field(default_factory=list, repr=False)

    @property
    def action_history(self):
        """Actions of successive undershoots (the notebook's ``SElist``)."""
        return [s.action for s in self.shots if s.outcome == "undershoot"]


def shoot(dV, V, u0: float, settings: Settings, u_true: float | None = None) -> Shot:
    r0 = 1.0 / settings.inf
    d0 = float(dV(u0))
    if d0 >= 0.0:
        # at or beyond the true vacuum: the field never leaves it / runs away
        return Shot(u0, "overshoot", r0, float("nan"), float("nan"))
    u_start = u0 + d0 * r0**2 / 8.0
    v_start = d0 * r0 / 4.0

    def rhs(r, Y):
        u, v, _, _ = Y
        dVu = float(dV(u))
        Vu = float(V(u))
        return [v, dVu - 3.0 * v / r, r**3 * (0.5 * v * v + Vu), r**3 * v * v]

    def over(r, Y):
        return Y[0] - r0

    over.terminal = True
    over.direction = -1

    def under(r, Y):
        return Y[1]

    under.terminal = True
    under.direction = 1

    u_cap = 2.0 * (u_true if u_true is not None else u0)

    def runaway(r, Y):
        return Y[0] - u_cap

    runaway.terminal = True
    runaway.direction = 1

    sol = solve_ivp(
        rhs, (r0, settings.inf), [u_start, v_start, 0.0, 0.0], method="DOP853",
        rtol=settings.rtol, atol=settings.atol, events=(over, under, runaway),
    )
    if sol.status == -1:
        raise RuntimeError(f"bounce integration failed: {sol.message}")
    if sol.t_events[2].size:
        return Shot(u0, "overshoot", float(sol.t_events[2][0]), float("nan"), float("nan"))
    if sol.t_events[0].size:
        outcome = "overshoot"
        r_end, Y = sol.t_events[0][0], sol.y_events[0][0]
    elif sol.t_events[1].size and sol.t_events[1][0] < 0.99 * settings.inf:
        outcome = "undershoot"
        r_end, Y = sol.t_events[1][0], sol.y_events[1][0]
    else:
        r_end, Y = sol.t[-1], sol.y[:, -1]
        if float(V(u0)) > 0.0:
            # Released above the false-vacuum energy: by energy conservation (with
            # friction) the field can never reach u = 0, so this is an undershoot that
            # has not turned around before r = inf.
            outcome = "undershoot"
        else:
            # Still sitting near u0 at r = inf: it would overshoot later.
            outcome = "unresolved"
    return Shot(u0, outcome, float(r_end), TWO_PI2 * Y[2], 0.5 * math.pi**2 * Y[3])


def find_bounce(pot, hwmin: float, hwmax: float, settings: Settings) -> Bounce:
    """Bisection on the release point until it is resolved to 10**-ormax (units of w)
    and the action of successive undershoots changes by less than se_acc."""
    tab = TabulatedPotential(pot.V, pot.dV, hi=2.0 * hwmax, n=800_001)
    V, dV = tab.V, tab.dV
    width = hwmax - hwmin
    tol = 10.0 ** (-settings.ormax)  # u0 resolution (units of w), as the notebook
    # bracket in t = ln(hwmax - u0)
    t_under = math.log(width * (1.0 - 1e-6))  # next to the barrier top: undershoot
    t_over = math.log(max(1e-15 * hwmax, 1e-300))  # next to the true vacuum: overshoot
    shots = []
    last_under = None
    last_over = None
    unresolved = 0

    s = shoot(dV, V, hwmax - math.exp(t_under), settings, hwmax)
    shots.append(s)
    if s.outcome != "undershoot":
        raise RuntimeError("starting next to the barrier top does not undershoot")
    last_under = s

    converged = False
    prev_action = None
    for _ in range(settings.max_shots):
        t_mid = 0.5 * (t_under + t_over)
        u0 = hwmax - math.exp(t_mid)
        s = shoot(dV, V, u0, settings, hwmax)
        shots.append(s)
        if s.outcome == "undershoot":
            t_under = t_mid
            last_under = s
            width_now = math.exp(t_under) - math.exp(t_over)
            if prev_action is not None:
                se_ok = abs(s.action - prev_action) <= settings.se_acc * abs(s.action)
                # notebook: the S_E test starts once u0 is resolved to 10**-(ormax-5)
                if se_ok and width_now < 10.0 ** (-(settings.ormax - 5)):
                    converged = True
                    break
            prev_action = s.action
            if width_now < tol:
                break
        else:
            if s.outcome == "unresolved":
                unresolved += 1
            t_over = t_mid
            last_over = s
            if math.exp(t_under) - math.exp(t_over) < tol:
                break
        if math.exp(t_under) - math.exp(t_over) < 1e-15 * hwmax:
            break  # float64 resolution of u0 reached
    if unresolved:
        warnings.warn(
            f"{unresolved} shots did not finish before r = inf = {settings.inf:g}; "
            "they were treated as overshoots. Increase Settings.inf if results look wrong.",
            RuntimeWarning, stacklevel=2,
        )
    return Bounce(
        action=last_under.action,
        action_direct=last_under.action_direct,
        action_virial=last_under.action_virial,
        u0=last_under.u0,
        u0_overshoot=last_over.u0 if last_over else float("nan"),
        converged=converged,
        shots=shots,
    )
