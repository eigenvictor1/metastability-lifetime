# metastability-lifetime

Python port of the vacuum-lifetime calculation in the Mathematica notebook
`2024-11_Scan_v1.2.nb` (function `Everything`, without the collider part).

The model is the SM plus a complex singlet (mass scale M_S, vev w) and a heavy neutrino (mass M_N = w·Y_R/(2√2), Yukawa Y_ν to the Higgs). For one parameter point the package:

1. **Runs the couplings.** It solves the SM RGEs (3-loop λ) from m_t up to M_N, applies the neutrino threshold, then solves the SM + N RGEs (with t2 = Y_ν²) from M_N to M_S. The run stops if t2 or λ crosses 4π (non-perturbative point).
2. **Finds the instability scale.** It adds the one-loop correction λ₁ and finds μ_in where λ_eff = λ + λ₁ = 0, and the bound m_h² < |β_λ(μ_in)| e^{-3/2} μ_in².
3. **Matches at M_S.** It computes the tree-level quartic λ_H of the full theory.
4. **Builds the tunnelling path.** This is the steepest-descent path of the tree potential V(H, S) from the false vacuum (0, w) to the true vacuum (v, 0). Along it, V_eff = V_tree + ¼(λ_smooth(h) − λ_eff(M_S)) h⁴, so below M_S the SM quartic runs.
5. **Checks the vacuum structure.** The checks are λ_SM(M_S) > 0, λ_H < 0, a high-energy vacuum above the electroweak one (`MSV@HE`), and no barrier.
6. **Solves for the bounce.** It solves the O(4) bounce by overshoot/undershoot shooting and integrates the Euclidean action S_E.
7. **Converts to a lifetime.** ln(Γ/V) = 4 ln(u₀ w) + 2 ln(S_E/2π) − S_E, and log₁₀(τ/yr) = −158.06 − log₁₀(e)·ln(Γ/V).

## Install (Windows)

```powershell
cd C:\Users\victo\dev\metastability-lifetime
py -m venv .venv
.venv\Scripts\activate
pip install -e ".[test]"
```

## Use

Single point (masses in GeV; `--Ms` is √Ms2 of the notebook):

```powershell
metastability-lifetime point --Ms 1e5 --w 316227.766 --kappa 0.1 --YR 0.3005113 --Ynu 3.8
```

Grid scan, run in parallel and written to `results\<grid>-<date>.csv`:

```powershell
metastability-lifetime scan examples\grid.json --workers 8
```

A grid is either a Cartesian product (`examples/grid.json`) or a notebook-style list of
`[Ms, w, kappa, YR, Ynumin, Ynumax, dYnu]` (`examples/grid_notebook_style.json`).
As in the notebook, the Y_ν values for a given (M_S, w, κ, Y_R) are scanned in increasing order. After the first too-short-lived or no-barrier point, the remaining values are marked `skipped`; `--no-skip` turns this off.

From Python:

```python
from metastability_lifetime import compute_point, Settings

r = compute_point(Ms=1e5, w=316227.766, kappa=0.1, YR=0.3005113, Ynu=3.8)
print(r.status, r.mu_in, r.action, r.log10_tau)

# intermediate objects (running, potential along the path, bounce shots)
r, extra = compute_point(1e5, 316227.766, 0.1, 0.3005113, 3.8, details=True)
pot = extra["potential"]          # pot.V(hw), pot.dV(hw), pot.fields(hw)
```

`Settings` holds the notebook's numerical knobs:
- `ormax = 9`: digits of the release point
- `inf = 1e5`: largest bubble radius, in units of 1/w
- `se_acc = 1e-3`: relative accuracy of S_E

It also holds the lifetime constants (`log10_tau_offset = -158.06`, `lograte_crit = -376.104`).

## Output

| field | meaning (notebook name) |
|---|---|
| `status` | `ok`, `nope` (too short-lived), `non-perturbative`, `lamSM>0`, `lamH<0`, `MSV@HE`, `no barrier`, `no instability scale`, `no descent path`, `skipped` |
| `MN` | heavy-neutrino mass in GeV |
| `lam_nonpert_scale` | scale where t2 or λ reaches 4π (Λ_non-pert) |
| `lamH` | tree-level matched quartic at M_S |
| `mu_in` | instability scale in GeV (`muin`) |
| `mb2` | upper bound on m_h² in GeV² |
| `hw_barrier`, `hw_true_vacuum` | barrier top and true vacuum along the path, in units of w (`hwmin`, `hwmax`) |
| `u0` | bounce release point, in units of w (`u0min`) |
| `action` | Euclidean action S_E (`SEf`) |
| `action_direct`, `action_virial` | the two evaluations of S_E (see difference 4) |
| `lograte` | ln(Γ/V / GeV⁴) (`Lograte`) |
| `log10_tau` | log₁₀(τ/yr) (`log10tau`) |
| `action_history` | S_E of successive undershoots (`SElistl`) |

`log10_tau` is filled in for `nope` points too. The `nope` cut (`Lograte ≥ −376.104`) corresponds to log₁₀(τ/yr) ≈ 5.3, although the notebook's text says the lifetime should exceed 10¹⁰ yr. Adjust `Settings.lograte_crit` if 5.3 is not intended.

## Validation against the notebook

`validation/notebook_reslist.csv` is the notebook's saved `Reslist`. The script below recomputes every stored point that was actually computed (283 points) and writes `validation/comparison.csv`:

```powershell
python validation\compare_notebook.py --workers 8
```

Result on all 283 computed points (median 2 s per point on one core):

- **Running (μ_in, m_h² bound).** 235 points agree to better than 5×10⁻⁷ (m_h² bound to 3×10⁻⁶). The other 39 all have the notebook root below M_N, an extrapolation artifact (difference 2 below).
- **Long-lived points.** Both codes return a lifetime for 86 points. S_E differs by 0.7% median, 6% at the 90th percentile and 11% at most. Relative to log₁₀τ itself, the difference is 1% median. For the 55 points with log₁₀τ > 100 the S_E difference is 0.6% median.
- **Short-lived points.**

  | outcome (Python vs notebook) | points |
  |---|---|
  | "nope" in both | 127 (S_E 3% median difference) |
  | "no barrier" here, "nope" in the notebook (S_E ≈ 70–160) | 56 |
  | "non-perturbative" here, "nope" in the notebook | 7 |
  | "non-perturbative" here, a lifetime in the notebook | 2 |

  For the last group the notebook kept computing past the 4π cut (difference 3 below).
- **Ok/nope flips.** 5 points are "nope" here but have a lifetime in the notebook. All sit within a few units of the cut at log₁₀τ ≈ 5.3: Python gives 0.5 to 4.9, the notebook 5.9 to 16.
- **Where the S_E differences come from.** They track the tunnelling-path construction (difference 1 below), not the running or the shooting. The internal check, S_E against its virial form, agrees to ~10⁻⁵ for the reference point.

## Differences from the notebook

These are deliberate. All of them are numerical or edge-case behaviour; the physics inputs and formulas are the notebook's.

1. **Tunnelling path.** The notebook follows the normalised gradient flow with explicit Euler steps dh = 10⁻³, then fits the path with a degree-80 polynomial. In the steep singlet direction those steps zigzag, which adds spurious energy ~ m_S² dh² along the path. Near the flat false vacuum (massless Higgs at tree level) this is comparable to the barrier. The result also does not converge as dh is lowered: S_E moves from 400 to 366 when dh goes from 10⁻³ to 3×10⁻⁴, and breaks down at 10⁻⁴.

   The package computes the exact steepest-descent curve instead. It integrates the flow with H as the independent variable and D = 1 − S² to avoid round-off, and evaluates V_tree − V_false = −κH²D + λ_H H⁴ + (m/2)D² without cancellations. The result is independent of the sampling: S_E agrees to 10⁻⁶ between 5k and 60k samples. Along this path the heavy singlet cancels exactly at tree level and the low-field potential reduces to the SM quartic, as it should.

   For long-lived points (barrier at large field) the two agree to ~10⁻³. For points whose barrier sits at small field, S_E differs by a few percent, up to ~10%. Some large-κ points the notebook stored with S_E ≈ 70–160 have no barrier at all along the exact path. All of these are short-lived either way.
2. **Instability scale.** The notebook runs `FindRoot` from x_max − 1, which can lie below M_N, where it uses Mathematica's extrapolation of the interpolating function. That gives roots below M_N, e.g. 92 TeV for a point with M_N = 100 TeV, whose real zero of λ_eff is at 201 TeV. The package takes the zero of λ_eff in [M_N, M_S].
3. **Non-perturbative points.** The notebook records them in `nonpertlist` but goes on computing a lifetime with the extrapolated couplings. The package stops and returns `non-perturbative`. As in the notebook, the 4π cut only fires on an upward crossing, so a point that already has Y_ν² > 4π at M_N, e.g. Y_ν = 3.8, is not flagged.
4. **Shooting numerics.**
   - The digit-by-digit scan of u₀ is replaced by bisection in ln(h_max − u₀).
   - The integration starts from the regular series u₀ + V′(u₀) r²/8.
   - S_E is integrated together with the ODE, and the virial value π²/2 ∫ r³ u′² is kept as a check (`action_direct`, `action_virial`).
   - If the two differ by more than 1%, the virial value is used. This happens when V_eff is slightly negative at very small field, because the quartic is frozen below M_N and the threshold shift pushes it under zero. The long massless tail of a near-critical shot then integrates r³V < 0 out to huge r.
   - The stopping rule is the notebook's: u₀ resolved to 10^-(ormax−5) and successive undershoot actions within `se_acc`, or u₀ resolved to 10^-ormax.
5. **Barrier.** As `FindArgMax`, `hwmin` is the local maximum of V_eff reached from e^(−1/4) μ_in/w with hw > μ_in/(500 w). A barrier whose top is not above the false vacuum (V_eff(hwmin) ≤ 0) is classified `no barrier` instead of failing in the shooting.
6. **No descent path.** If λ_eff(M_S) < 0 only because of the one-loop term, while the tree-level quartic along the singlet valley λ_H − κ²w²/(2M_S²) ≥ 0, the false vacuum is a tree-level minimum. There is then no steepest-descent path, and the notebook's Euler flow just wanders for 10⁶ steps. These points get status `no descent path`; this happens e.g. at κ = 0.1, Y_R = 0.30, Y_ν = 3.9.
7. **Not ported.** The collider/SMEFT part (`FCCCs`, `paramDictionary`, the `exp` code) is not ported; the lifetime does not depend on it.

## Layout

```
src/metastability_lifetime/
    _betas.py      beta functions, generated from the notebook (tools/gen_betas.py)
    constants.py   SM inputs at m_t, lifetime constants, Settings
    running.py     RG running, thresholds, lambda_1, lamsmooth, instability scale
    potential.py   tree potential, steepest-descent path, V_eff along the path
    bounce.py      overshoot/undershoot shooting and Euclidean action
    lifetime.py    compute_point (one point), compute_ynu_scan (Everything)
    cli.py         command line
tests/             pytest regression tests against stored notebook values
validation/        notebook Reslist and the comparison script
tools/             notebook -> readable code -> _betas.py
```
