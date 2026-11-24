"""Go / no-go test: how much can a better charging decision gain?

For each simulated day the routes and departure times are fixed once (and
cached on disk), then flown under:
    regret        the base paper's rule
    nearest       shortest detour
    rollout-air   every option simulated to the end of the day, drones in the air only
    rollout-all   the same, with later departures also simulated (uses future knowledge)
Rollout uses the regret rule for all later decisions.

Examples
    python run_rollout.py C75_R5 --seeds 30 --ugv 1
    python run_rollout.py C75_R5 --seeds 30 --ugv 2 --workers 6
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import re
import statistics as st
import sys
import time

from drp.instance import load
from drp.params import Params
from drp.policies import regret_policy, nearest_policy
from drp.dynamic import release_times, plan_day, execute

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "figshare_instances", "extracted", "DATA", "DATA")


def t_crit(df):
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
             8: 2.306, 9: 2.262, 10: 2.228, 14: 2.145, 19: 2.093, 24: 2.064,
             29: 2.045, 39: 2.023, 49: 2.010, 59: 2.001, 99: 1.984}
    return table.get(df) or table[max(k for k in table if k <= df)]


def ci95(xs):
    n = len(xs)
    if n < 2:
        return (xs[0] if xs else 0.0), float("nan")
    return st.mean(xs), t_crit(n - 1) * st.stdev(xs) / math.sqrt(n)


def load_instance(name, customers):
    fam = re.match(r"[A-Z]+", name).group(0)
    path = os.path.join(DATA, fam, name + ".txt")
    case = os.path.join(HERE, "case", name + ".txt")      # case-study instances built from OSM
    if os.path.exists(case):
        path = case
    if not os.path.exists(path):
        sys.exit(f"instance not found: {path}")
    inst = load(path)
    return inst.subset(customers) if customers else inst


def plan_one(job):
    name, customers, pkw, dod, epoch, iters, seed = job
    inst = load_instance(name, customers)
    P = Params(**pkw)
    rel = release_times(inst, P, dod, seed=seed, epoch=epoch)
    sched, _ = plan_day(inst, P, regret_policy(P.xi1, P.xi2), rel, epoch, iters, seed)
    return seed, sched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--battery", type=float, default=100.0)
    ap.add_argument("--pads", type=int, default=1)
    ap.add_argument("--ugv", type=int, default=1)
    ap.add_argument("--dod", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--epoch", type=float, default=10.0)
    ap.add_argument("--customers", type=int, default=None)
    ap.add_argument("--late-penalty", type=float, default=200.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    pkw = dict(dist_scale=a.scale, E=a.battery, pads=a.pads, n_ugv=a.ugv,
               infeasible_penalty=a.late_penalty)
    P = Params(**pkw)
    inst = load_instance(a.instance, a.customers)
    tag = (f"{inst.name.replace('[:', '_').replace(']', '')}_s{a.scale}_E{a.battery:g}_p{a.pads}"
           f"_u{a.ugv}_d{a.dod}_i{a.iters}_e{a.epoch:g}")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    cache = os.path.join(HERE, "results", f"schedules_{tag}.json")
    scheds = {}
    if os.path.exists(cache):
        scheds = {int(k): v for k, v in json.load(open(cache)).items()}
    todo = [s for s in range(a.seeds) if s not in scheds]
    print(inst, f"| scale {a.scale} battery {a.battery} pads {a.pads} ugv {a.ugv} dod {a.dod}"
                f" | {a.seeds} days, {len(todo)} to plan with {a.workers} workers", flush=True)
    if todo:
        t0 = time.time()
        jobs = [(a.instance, a.customers, pkw, a.dod, a.epoch, a.iters, s) for s in todo]
        with mp.Pool(a.workers) as pool:
            for seed, sched in pool.imap_unordered(plan_one, jobs):
                scheds[seed] = sched
        json.dump({str(k): v for k, v in sorted(scheds.items())}, open(cache, "w"))
        print(f"planned {len(todo)} days in {time.time() - t0:.0f}s -> {cache}", flush=True)

    runs = {
        "regret": lambda s: execute(inst, P, regret_policy(P.xi1, P.xi2), s),
        "nearest": lambda s: execute(inst, P, nearest_policy(), s),
        "rollout-air": lambda s: execute(inst, P, regret_policy(P.xi1, P.xi2), s, rollout="air"),
        "rollout-all": lambda s: execute(inst, P, regret_policy(P.xi1, P.xi2), s, rollout="all"),
    }
    out = {n: [] for n in runs}
    t0 = time.time()
    for seed in range(a.seeds):
        sched = [(t, r) for t, r in scheds[seed]]
        for n, f in runs.items():
            out[n].append(f(sched))
    print(f"executed in {time.time() - t0:.0f}s\n")

    print(f"{'rule':12} {'cost':>9} {'+-95%':>7} {'late':>5} {'sat-loss':>8} {'stations':>8} "
          f"{'swaps':>6} {'sw-wait':>8} {'decisions':>9} {'>1 option':>9} {'overrides':>9}")
    for n, rows in out.items():
        m, h = ci95([r.cost for r in rows])
        f = lambda g: st.mean(g(r) for r in rows)
        print(f"{n:12} {m:9.2f} {h:7.2f} {f(lambda r: r.late):5.2f} {f(lambda r: r.sat_loss):8.2f} "
              f"{f(lambda r: r.station_visits):8.1f} {f(lambda r: r.swaps):6.1f} "
              f"{f(lambda r: r.swap_wait):8.1f} {f(lambda r: r.n_decisions):9.1f} "
              f"{f(lambda r: r.n_real_choices):9.1f} {f(lambda r: r.n_overrides):9.1f}")

    base = out["regret"]
    base_mean = st.mean(r.cost for r in base)
    print("\nPaired difference against 'regret' (same schedule, same day):")
    for n in list(out)[1:]:
        diffs = [x.cost - y.cost for x, y in zip(out[n], base)]
        m, h = ci95(diffs)
        better = sum(1 for d in diffs if d < -1e-9)
        worse = sum(1 for d in diffs if d > 1e-9)
        verdict = "significant at 5%" if abs(m) > h else "not significant"
        print(f"  {n:12} mean {m:+9.2f} ({100 * m / base_mean:+.1f}%)  95% CI [{m - h:+.2f}, {m + h:+.2f}]"
              f"  better on {better}, worse on {worse}, equal on {len(diffs) - better - worse}"
              f"  -> {verdict}")


if __name__ == "__main__":
    main()
