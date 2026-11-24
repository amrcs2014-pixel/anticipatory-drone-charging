"""Which distance scale reproduces the paper's small-instance optima (Table 6)?

The instance files do not state units.  This script assumes each small
instance is the first n customers of the matching 75- or 100-customer file
(an assumption: the paper does not say how they were built) and reports the
best cost found for several values of km-per-coordinate-unit.

Usage:  python calibrate_units.py
"""
import os
import sys

from drp.instance import load
from drp.params import Params
from drp.policies import regret_policy
from drp.alns import ALNS

DATA = os.path.join(os.path.dirname(__file__), "..", "figshare_instances",
                    "extracted", "DATA", "DATA")

# name: (family, customers, stations, Gurobi value in Table 6, proven optimal?)
TARGETS = {
    "C7_R5":   ("C", 7, 5, 212.67, True),
    "RC6_R3":  ("RC", 6, 3, 220.33, True),
    "R11_R3":  ("R", 11, 3, 329.76, True),
    "C11_R3":  ("C", 11, 3, 324.64, False),
    "R12_R4":  ("R", 12, 4, 335.05, False),
    "RC10_R6": ("RC", 10, 6, 333.01, False),
    "RC12_R4": ("RC", 12, 4, 439.82, False),
    "C15_R4":  ("C", 15, 4, 428.78, False),
}
SCALES = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]


def best_cost(inst, scale, seeds=(0, 1, 2), iters=120):
    P = Params(dist_scale=scale)
    best = None
    for s in seeds:
        _, res = ALNS(inst, P, regret_policy(), seed=s, iters=iters).run()
        if best is None or res.cost < best.cost:
            best = res
    return best


def main():
    print("km per unit: " + "  ".join(f"{s:>8}" for s in SCALES))
    for name, (fam, n, st, target, opt) in TARGETS.items():
        for size in (75, 100):
            path = os.path.join(DATA, fam, f"{fam}{size}_R{st}.txt")
            inst = load(path).subset(n)
            row = []
            for s in SCALES:
                r = best_cost(inst, s)
                row.append(f"{r.cost:8.2f}" + ("" if r.feasible else "*"))
            print(f"{name:8s} from {fam}{size}_R{st}: " + " ".join(row)
                  + f"   | paper {target:.2f}{' (optimal)' if opt else ''}")
    print("* = best solution found still violates a constraint")


if __name__ == "__main__":
    sys.exit(main())
