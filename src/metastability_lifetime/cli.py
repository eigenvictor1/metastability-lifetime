"""Command line interface.

    metastability-lifetime point --Ms 1e5 --w 316227.77 --kappa 0.1 --YR 0.3005 --Ynu 3.8
    metastability-lifetime scan examples/grid.json --workers 4
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields

from .constants import Settings
from .lifetime import Result, compute_point, compute_ynu_scan, ynu_values

FIELDS = [f.name for f in fields(Result)]


def _settings_from(d: dict | None, args=None) -> Settings:
    st = Settings(**(d or {}))
    if args is not None:
        for name in ("ormax", "inf", "se_acc"):
            val = getattr(args, name, None)
            if val is not None:
                setattr(st, name, val)
    return st


def _add_setting_args(p):
    p.add_argument("--ormax", type=int, help="digits of the bounce release point (default 9)")
    p.add_argument("--inf", type=float, help="maximum bubble radius in units of 1/w (default 1e5)")
    p.add_argument("--se-acc", dest="se_acc", type=float, help="relative accuracy of S_E (default 1e-3)")


def _row(r: Result) -> dict:
    d = r.as_dict()
    d["action_history"] = ";".join(f"{x:.10g}" for x in d["action_history"])
    return d


def cmd_point(args):
    st = _settings_from(None, args)
    t = time.time()
    r = compute_point(args.Ms, args.w, args.kappa, args.YR, args.Ynu, st)
    d = r.as_dict()
    d["seconds"] = round(time.time() - t, 2)
    if args.json:
        print(json.dumps(d, indent=2))
    else:
        width = max(len(k) for k in d)
        for k, v in d.items():
            print(f"{k:<{width}}  {v}")


def _expand_grid(grid: dict):
    """Yield (Ms, w, kappa, YR, [Ynu...]) tuples."""
    if "points" in grid:  # notebook PSGrid style: [Ms, w, kappa, YR, Ynumin, Ynumax, dYnu]
        for p in grid["points"]:
            Ms, w, kappa, YR, a, b, c = p
            yield Ms, w, kappa, YR, ynu_values(a, b, c)
        return
    ynu = grid["Ynu"]
    ylist = ynu_values(ynu["min"], ynu["max"], ynu["step"]) if isinstance(ynu, dict) else list(ynu)
    for Ms, w, kappa, YR in itertools.product(grid["Ms"], grid["w"], grid["kappa"], grid["YR"]):
        yield Ms, w, kappa, YR, ylist


def _work(job):
    Ms, w, kappa, YR, ylist, st_dict, skip = job
    return compute_ynu_scan(Ms, w, kappa, YR, ylist, Settings(**st_dict), skip_after_failure=skip)


def cmd_scan(args):
    with open(args.grid) as f:
        grid = json.load(f)
    st = _settings_from(grid.get("settings"), args)
    jobs = [(*g, vars(st), not args.no_skip) for g in _expand_grid(grid)]
    n_points = sum(len(j[4]) for j in jobs)
    out = args.out
    if out is None:
        stem = os.path.splitext(os.path.basename(args.grid))[0]
        os.makedirs("results", exist_ok=True)
        out = os.path.join("results", f"{stem}-{time.strftime('%Y-%m-%d_%H%M%S')}.csv")
    print(f"{len(jobs)} (Ms, w, kappa, YR) combinations, {n_points} points -> {out}", flush=True)

    done = 0
    t0 = time.time()
    with open(out, "w", newline="") as f, ProcessPoolExecutor(max_workers=args.workers) as ex:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        futures = [ex.submit(_work, j) for j in jobs]
        for fut in as_completed(futures):
            for r in fut.result():
                wr.writerow(_row(r))
                done += 1
            f.flush()
            print(f"  {done}/{n_points} points  ({time.time() - t0:.0f}s)", flush=True)
    print(f"written {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="metastability-lifetime", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("point", help="lifetime of a single parameter point")
    p.add_argument("--Ms", type=float, required=True, help="M_S in GeV (= sqrt(Ms2))")
    p.add_argument("--w", type=float, required=True, help="singlet vev w in GeV")
    p.add_argument("--kappa", type=float, required=True)
    p.add_argument("--YR", type=float, required=True)
    p.add_argument("--Ynu", type=float, required=True)
    p.add_argument("--json", action="store_true", help="print JSON")
    _add_setting_args(p)
    p.set_defaults(func=cmd_point)

    s = sub.add_parser("scan", help="scan a parameter grid given in a JSON file")
    s.add_argument("grid", help="JSON grid file (see examples/grid.json)")
    s.add_argument("--out", help="output CSV (default results/<grid>-<date>.csv)")
    s.add_argument("--workers", type=int, default=os.cpu_count())
    s.add_argument("--no-skip", action="store_true",
                   help="compute every Ynu even after a short-lived / no-barrier point")
    _add_setting_args(s)
    s.set_defaults(func=cmd_scan)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
