"""Compare the package against results stored in 2024-11_Scan_v1.2.nb.

notebook_reslist.csv is the notebook's saved ``Reslist``. Rows with an SE_list were
actually computed (the others were skipped after a short-lived point). For those we
compare the instability scale, the m_h^2 bound, the Euclidean action (the notebook
uses the second-to-last entry of its SElist) and log10(tau/yr).

Usage (from the repository root):
    python validation/compare_notebook.py                 # all computed points
    python validation/compare_notebook.py --n 30 --workers 4
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from metastability_lifetime import compute_point  # noqa: E402


def _run(row):
    t = time.time()
    try:
        r = compute_point(float(row["Ms"]), float(row["w"]), float(row["kappa"]), float(row["YR"]), float(row["Ynu"]))
    except Exception as exc:  # report and continue
        return row, None, repr(exc), time.time() - t
    return row, r, "", time.time() - t


def rel(a, b):
    return abs(a - b) / abs(b) if b else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="random subset size (0 = all)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(os.path.join(HERE, "notebook_reslist.csv")) as f:
        rows = [r for r in csv.DictReader(f) if r["SE_list"]]
    if args.n:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.n]

    out = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for row, r, err, dt in ex.map(_run, rows):
            se_nb = [float(x) for x in row["SE_list"].split(";")]
            se_ref = se_nb[-2] if len(se_nb) > 1 else se_nb[-1]
            rec = dict(Ms=row["Ms"], w=row["w"], kappa=row["kappa"], YR=row["YR"], Ynu=row["Ynu"],
                       nb_mu_in=row["mu_in"], nb_mb2=row["mb2"], nb_SE=se_ref, nb_log10_tau=row["log10_tau"],
                       status="", mu_in="", mb2="", SE="", log10_tau="", d_mu_in="", d_mb2="", d_SE="",
                       d_log10_tau="", seconds=f"{dt:.1f}", error=err)
            if r is not None:
                rec.update(status=r.status, mu_in=r.mu_in, mb2=r.mb2, SE=r.action, log10_tau=r.log10_tau,
                           d_mu_in=rel(r.mu_in, float(row["mu_in"])), d_mb2=rel(r.mb2, float(row["mb2"])),
                           d_SE=rel(r.action, se_ref))
                if row["log10_tau"] != "nope" and r.status == "ok":
                    rec["d_log10_tau"] = r.log10_tau - float(row["log10_tau"])
            out.append(rec)
            print(f"{row['Ms'][:8]:>8} {row['w'][:9]:>9} k={float(row['kappa']):.3f} YR={float(row['YR']):.3f} "
                  f"Ynu={float(row['Ynu']):.2f}  {rec['status']:<10} SE {rec['SE'] if rec['SE']=='' else round(rec['SE'],2)}"
                  f" vs {se_ref:.2f}  ({dt:.0f}s) {err}", flush=True)

    if args.out is None:
        args.out = os.path.join(HERE, "comparison.csv")
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0]))
        wr.writeheader()
        wr.writerows(out)

    def stats(key):
        v = [x[key] for x in out if isinstance(x[key], float) and not math.isnan(x[key])]
        if not v:
            return "n/a"
        a = sorted(abs(t) for t in v)
        return f"n={len(v)}  median |d|={a[len(a)//2]:.2e}  max |d|={a[-1]:.2e}"

    print("\nrelative differences:")
    for k in ("d_mu_in", "d_mb2", "d_SE"):
        print(f"  {k:<12} {stats(k)}")
    print(f"  d_log10_tau (absolute) {stats('d_log10_tau')}")
    agree = sum(1 for x in out if x["status"] in ("ok", "nope") and
                ((x["status"] == "nope") == (x["nb_log10_tau"] == "nope")))
    print(f"  same ok/nope classification: {agree}/{len(out)}")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
