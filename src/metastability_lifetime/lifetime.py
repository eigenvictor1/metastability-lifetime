"""Vacuum lifetime of one parameter point (the lifetime part of ``Everything``).

Parameters (same names as the notebook):
    Ms     heavy scale M_S = Sqrt[Ms2] in GeV (matching scale; the notebook passes Ms2)
    w      singlet vev in GeV
    kappa  Higgs-singlet portal coupling
    YR     singlet-neutrino Yukawa; the neutrino mass is M_N = w YR / (2 sqrt 2)
    Ynu    neutrino Yukawa to the Higgs (t2 = Ynu^2)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from .bounce import Bounce, find_bounce
from .constants import Settings
from .potential import NoDescentPath, PathPotential, TreePotential, steepest_descent_path
from .running import build_lamsmooth, instability_scale, run_smn

LOG10E = math.log10(math.e)

# status values (the notebook's lists / "prob" labels)
OK = "ok"  # Reslist with a lifetime
SHORT = "nope"  # Reslist, lifetime below the cut (Lograte >= LogratecritGeV)
NONPERT = "non-perturbative"  # nonpertlist
LAMSM_POS = "lamSM>0"  # nonEFTlist
LAMH_NEG = "lamH<0"  # nonEFTlist
MSV_HE = "MSV@HE"  # nonEFTlist
NO_BARRIER = "no barrier"  # nonEFTlist
NO_INSTABILITY = "no instability scale"  # (FindRoot would fail in the notebook)
NO_DESCENT = "no descent path"  # tree-level valley quartic >= 0: false vacuum is a tree minimum
SKIPPED = "skipped"  # Reslist "nope" without computation, after a previous nope/no barrier


@dataclass
class Result:
    Ms: float
    w: float
    kappa: float
    YR: float
    Ynu: float
    status: str
    MN: float = float("nan")  # heavy-neutrino mass, GeV
    lam_nonpert_scale: float = float("nan")  # Lambda_non-pert (GeV), for non-perturbative points
    lamH: float = float("nan")  # tree-level matched quartic at M_S
    mu_in: float = float("nan")  # instability scale, GeV
    mb2: float = float("nan")  # upper bound on m_h^2 (GeV^2)
    hw_barrier: float = float("nan")  # hwmin (units of w)
    hw_true_vacuum: float = float("nan")  # hwmax (units of w)
    u0: float = float("nan")  # bounce release point (units of w)
    action: float = float("nan")  # Euclidean action S_E
    action_direct: float = float("nan")  # 2 pi^2 int r^3 (u'^2/2 + V)
    action_virial: float = float("nan")  # pi^2/2 int r^3 u'^2 (check: equal for an exact bounce)
    lograte: float = float("nan")  # ln(Gamma/V / GeV^4)
    log10_tau: float = float("nan")  # log10(lifetime / yr)
    converged: bool = False
    action_history: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


def compute_point(Ms, w, kappa, YR, Ynu, settings: Settings | None = None, details: bool = False):
    """Run the full lifetime calculation for one point. Returns a :class:`Result`
    (and a dict with intermediate objects if ``details=True``)."""
    st = settings or Settings()
    Ms2 = Ms * Ms
    MN = w * YR / (2.0 * math.sqrt(2.0))
    res = Result(Ms, w, kappa, YR, Ynu, status=OK, MN=MN)
    extra = {}

    def done():
        return (res, extra) if details else res

    run = run_smn(Ms, MN, Ynu)
    extra["running"] = run
    if not run.perturbative:
        res.status = NONPERT
        res.lam_nonpert_scale = math.exp(run.xmax)
        return done()
    build_lamsmooth(run)

    xS = run.xS
    lnS = float(run.lam(xS)[0])
    le_S = float(run.lam_eff(xS)[0])
    w2, k2 = w * w, kappa * kappa
    lamH = lnS + w2 * k2 / (2 * Ms2) + 6 * k2 * lnS * w2 / (math.pi**2 * Ms2) - 3 * k2 * k2 * w2 * w2 / (math.pi**2 * Ms2**2)
    res.lamH = lamH
    if le_S > 0 or lamH <= 0:
        res.status = LAMSM_POS if le_S > 0 else LAMH_NEG
        return done()

    xin = instability_scale(run)
    if xin is None:
        res.status = NO_INSTABILITY
        return done()
    muin = math.exp(xin)
    res.mu_in = muin
    res.mb2 = float(abs(run.beta_lam(xin)) * math.exp(-1.5) * muin**2)

    tree = TreePotential(kappa=kappa, Ms2=Ms2, w=w, lamH=lamH)
    v = math.sqrt(kappa * w2 / (2 * lamH))
    vw = v / w
    muinw = muin / w
    try:
        hw, H, D = steepest_descent_path(tree, vw)
    except NoDescentPath:
        res.status = NO_DESCENT
        return done()
    pot = PathPotential(tree=tree, run=run, w=w, lam_eff_S=le_S, hw_path=hw, H_path=H, D_path=D)
    extra["potential"] = pot

    hwmax = pot.true_vacuum()
    res.hw_true_vacuum = hwmax
    if pot.V(hwmax - 10.0 ** (-st.ormax + 1)) >= 0:
        res.status = MSV_HE
        return done()
    hwmin = pot.barrier_top(muinw, hwmax)
    res.hw_barrier = hwmin
    if hwmin < muinw / 100.0 or pot.V(hwmin) <= 0.0:
        res.status = NO_BARRIER
        return done()

    b: Bounce = find_bounce(pot, hwmin, hwmax, st)
    extra["bounce"] = b
    res.u0 = b.u0
    res.action = float(b.action)
    res.action_direct = float(b.action_direct)
    res.action_virial = float(b.action_virial)
    res.converged = b.converged
    res.action_history = [float(a) for a in b.action_history]
    SE = b.action
    lograte = 4 * math.log(b.u0) + 2 * math.log(SE / (2 * math.pi)) - SE + 4 * math.log(w)
    res.lograte = float(lograte)
    res.log10_tau = float(st.log10_tau_offset - LOG10E * lograte)
    if lograte >= st.lograte_crit:
        res.status = SHORT
    return done()


def ynu_values(ynu_min, ynu_max, dynu):
    """Mathematica's Do[..., {Ynu, min, max, d}] iterator."""
    n = int(math.floor((ynu_max - ynu_min) / dynu + 1e-9))
    return [ynu_min + i * dynu for i in range(n + 1)]


def compute_ynu_scan(Ms, w, kappa, YR, ynu_list, settings: Settings | None = None, skip_after_failure=True):
    """``Everything[{Ms2, w, kappa, YR, Ynumin, Ynumax, dYnu}]``: loop over Ynu.

    As in the notebook, once a point is too short-lived ("nope") or has no barrier,
    the remaining (larger) Ynu values are not computed and are marked "skipped".
    """
    out = []
    stop = False
    for ynu in ynu_list:
        if stop:
            out.append(Result(Ms, w, kappa, YR, ynu, status=SKIPPED, MN=w * YR / (2 * math.sqrt(2))))
            continue
        try:
            r = compute_point(Ms, w, kappa, YR, ynu, settings)
        except Exception as exc:  # keep long scans alive
            r = Result(Ms, w, kappa, YR, ynu, status=f"error: {exc}")
        out.append(r)
        if skip_after_failure and r.status in (SHORT, NO_BARRIER):
            stop = True
    return out
