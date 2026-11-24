"""Solve one static instance with the ALNS routing engine.

Examples
    python run_static.py C75_R5
    python run_static.py R100_R4 --policy cost --runs 10 --iters 300
    python run_static.py C75_R5 --pads 1 --ugv 2      (extended model)
"""
import argparse
import os
import re
import statistics
import sys

from drp.instance import load
from drp.params import Params
from drp.policies import regret_policy, cost_policy, nearest_policy, fcs_only
from drp.alns import ALNS

DATA = os.path.join(os.path.dirname(__file__), "..", "figshare_instances",
                    "extracted", "DATA", "DATA")


def find(name):
    fam = re.match(r"[A-Z]+", name).group(0)
    path = os.path.join(DATA, fam, name + ".txt")
    if not os.path.exists(path):
        sys.exit(f"instance not found: {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("--policy", choices=["regret", "cost", "nearest"], default="regret")
    ap.add_argument("--fcs-only", action="store_true", help="never use the UGV")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--scale", type=float, default=None, help="km per coordinate unit")
    ap.add_argument("--pads", type=int, default=None, help="pads per station (endogenous queue)")
    ap.add_argument("--ugv", type=int, default=1, help="number of UGVs")
    ap.add_argument("--customers", type=int, default=None, help="use only the first n customers")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    inst = load(find(a.instance))
    if a.customers:
        inst = inst.subset(a.customers)
    P = Params(pads=a.pads, n_ugv=0 if a.fcs_only else a.ugv)
    if a.scale:
        P.dist_scale = a.scale
    pol = {"regret": regret_policy(P.xi1, P.xi2), "cost": cost_policy(P.xi1, P.xi2),
           "nearest": nearest_policy()}[a.policy]
    if a.fcs_only:
        pol = fcs_only(pol)

    print(inst, f"| policy={a.policy} scale={P.dist_scale} pads={P.pads} ugv={P.n_ugv}")
    costs, secs = [], []
    for run in range(a.runs):
        alns = ALNS(inst, P, pol, seed=run, iters=a.iters, verbose=a.verbose)
        best, res = alns.run()
        costs.append(res.cost)
        secs.append(alns.seconds)
        print(f"run {run}: cost {res.cost:9.2f}  routes {res.n_routes}  km {res.travel_km:7.2f}"
              f"  sat-loss {res.sat_loss:5.2f}  station visits {res.station_visits}"
              f"  swaps {res.swaps}  wait {res.station_wait:6.2f}"
              f"  feasible {res.feasible}  {alns.seconds:5.1f}s")
    if a.runs > 1:
        print(f"mean {statistics.mean(costs):.2f}  best {min(costs):.2f}"
              f"  sd {statistics.stdev(costs):.2f}  mean time {statistics.mean(secs):.1f}s")


if __name__ == "__main__":
    main()
