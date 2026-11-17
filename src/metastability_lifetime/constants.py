"""Physical inputs and numerical settings, taken from 2024-11_Scan_v1.2.nb."""

from __future__ import annotations

import math
from dataclasses import dataclass

# --- SM boundary conditions at mu = m_t (notebook section "Basic stuff") -------------
M_TOP_POLE = 173.1  # GeV, scale where the SM couplings are given (xt = ln mt)
V_EW = 246.0  # GeV, used to convert running masses to Yukawas

MT_RUNNING = 161.98  # GeV
MB_RUNNING = 2.702  # GeV
MTAU_RUNNING = 1.78412  # GeV

SM_INPUTS_AT_MT = {
    "L": 0.12607,  # Higgs quartic
    "y": MT_RUNNING / V_EW * math.sqrt(2.0),  # top Yukawa
    "b": MB_RUNNING / V_EW * math.sqrt(2.0),  # bottom Yukawa
    "t": MTAU_RUNNING / V_EW * math.sqrt(2.0),  # tau Yukawa
    "g": 0.64765,  # g2
    "h": 0.358545,  # g1 (SM normalisation)
    "s": 1.1618,  # g3
}
SM_ORDER = ("L", "y", "b", "t", "g", "h", "s")
SMN_ORDER = SM_ORDER + ("t2",)

M_PLANCK = 1.22e19  # GeV (only used as default upper end of the SM running)

# Perturbativity cut used in the notebook (WhenEvent[t2 >= 4 Pi] and [L >= 4 Pi])
PERTURBATIVITY_LIMIT = 4.0 * math.pi

# --- Decay rate -> lifetime conversion -----------------------------------------------
# log10(tau / yr) = LOG10_TAU_OFFSET - log10(e) * Lograte, with Lograte = ln(Gamma/V / GeV^4)
LOG10_TAU_OFFSET = -158.06
# Points with Lograte >= LOGRATE_CRIT_GEV are flagged "nope" (too short-lived) in the notebook
LOGRATE_CRIT_GEV = -376.104


@dataclass
class Settings:
    """Numerical knobs. Defaults reproduce the notebook ("Main function" section).

    ormax:   number of digits of the release point u0 resolved by the shooting
             (the notebook's decimal search stops at 10**-ormax, in units of w).
    inf:     largest bubble radius integrated (and 1/inf is the starting radius),
             in units of 1/w. The notebook's error "point at the edge of the search
             region" corresponds to this being too small.
    se_acc:  relative accuracy requested for the Euclidean action.
    rtol, atol: tolerances of the bounce ODE integration.
    max_shots: safety cap on the number of shooting integrations.
    """

    ormax: int = 9
    inf: float = 1.0e5
    se_acc: float = 1.0e-3
    rtol: float = 1.0e-11
    atol: float = 1.0e-15
    max_shots: int = 300
    log10_tau_offset: float = LOG10_TAU_OFFSET
    lograte_crit: float = LOGRATE_CRIT_GEV
