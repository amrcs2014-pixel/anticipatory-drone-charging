"""Compare charging rules on identical dynamic days.

For every seed the routes and departure times are fixed first (with the
planning rule), then each charging rule flies exactly that schedule.  The
perfect-information benchmark is the same procedure with every order known at
time 0 and a much larger planning budget.

Examples
    python run_dynamic.py C75_R5
    python run_dynamic.py C75_R5 --dod 0.5 --seeds 10 --ugv 2
    python run_dynamic.py R75_R3 --battery 45 --scale 0.1
"""
import argparse
import math
import os
import re
import statistics as st
import sys
import time

from drp.instance import load
from drp.params import Params
from drp.policies import regret_policy, cost_policy, nearest_policy
from drp.dynamic import release_times, plan_day, execute

DATA = os.path.join(os.path.dirname(__file__), "..", "figshare_instances",
                    "extracted", "DATA", "DATA")

# two-sided 95% Student t critical values by degrees of freedom
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228, 14: 2.145, 19: 2.093, 29: 2.045}


def ci95(xs):
    """Mean and half-width of the 95% confidence interval."""
    n = len(xs)
    if n < 2:
        return (xs[0] if xs else 0.0), float("nan")
    df = n - 1
    tcrit = T95.get(df) or T95[max(k for k in T95 if k <= df)]
    return st.mean(xs), tcrit * st.stdev(xs) / math.sqrt(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("--scale", type=float, default=0.25, help="km per coordinate unit")
    ap.add_argument("--battery", type=float, default=100.0, help="Wh")
    ap.add_argument("--pads", type=int, default=1, help="pads per station")
    ap.add_argument("--ugv", type=int, default=1)
    ap.add_argument("--dod", type=float, default=0.5, help="share of orders revealed during the day")
    ap.add_argument("--policies", nargs="+", default=["regret", "cost", "nearest"])
    ap.add_argument("--plan-policy", default="regret")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--iters", type=int, default=20, help="ALNS iterations per dynamic re-plan")
    ap.add_argument("--bench-iters", type=int, default=200,
                    help="ALNS iterations for the perfect-information plan (0 = skip)")
    ap.add_argument("--epoch", type=float, default=10.0, help="minutes")
    ap.add_argument("--customers", type=int, default=None)
    ap.add_argument("--late-penalty", type=float, default=200.0,
                    help="cost of serving an order outside its tolerable window")
    ap.add_argument("--no-filter", action="store_true",
                    help="let rules pick options that make the next customer late")
    a = ap.parse_args()

    fam = re.match(r"[A-Z]+", a.instance).group(0)
    path = os.path.join(DATA, fam, a.instance + ".txt")
    if not os.path.exists(path):
        sys.exit(f"instance not found: {path}")
    inst = load(path)
    if a.customers:
        inst = inst.subset(a.customers)
    P = Params(dist_scale=a.scale, E=a.battery, pads=a.pads, n_ugv=a.ugv,
               infeasible_penalty=a.late_penalty, filter_late=not a.no_filter)
    makers = {"regret": lambda: regret_policy(P.xi1, P.xi2),
              "cost": lambda: cost_policy(P.xi1, P.xi2),
              "nearest": nearest_policy}

    print(inst, f"| scale {P.dist_scale} battery {P.E} Wh pads {P.pads} ugv {P.n_ugv} "
                f"dod {a.dod} | plan rule {a.plan_policy}, {a.iters} iterations per re-plan, "
                f"benchmark {a.bench_iters} iterations, {a.seeds} seeds")
    dyn = {n: [] for n in a.policies}      # per policy: list of SimResult, one per seed
    pi = {n: [] for n in a.policies}
    t0 = time.time()
    swapped = 0
    for s in range(a.seeds):
        rel = release_times(inst, P, a.dod, seed=s, epoch=a.epoch)
        sched, _ = plan_day(inst, P, makers[a.plan_policy](), rel, a.epoch, a.iters, s)
        for n in a.policies:
            dyn[n].append(execute(inst, P, makers[n](), sched))
        if a.bench_iters:
            rel0 = {c: 0.0 for c in inst.customers}
            sched0, _ = plan_day(inst, P, makers[a.plan_policy](), rel0, a.epoch,
                                 a.iters, s, first_iters=a.bench_iters)
            for n in a.policies:
                r0, d = execute(inst, P, makers[n](), sched0), dyn[n][-1]
                # with every order known in advance the dynamic schedule is also
                # available, so the benchmark takes the better of the two
                swapped += r0.cost > d.cost
                pi[n].append(r0 if r0.cost <= d.cost else d)
        print(f"  seed {s} done ({time.time() - t0:.0f}s)", flush=True)

    print(f"\n{'rule':8} {'dynamic cost':>13} {'+-95%':>7} {'late':>5} {'sat-loss':>8} "
          f"{'stations':>8} {'swaps':>6} {'sw-wait':>8} | {'perfect-info':>12} {'+-95%':>7} {'late':>5}")
    for n in a.policies:
        m, h = ci95([r.cost for r in dyn[n]])
        f = lambda g, rows=dyn[n]: st.mean(g(r) for r in rows)
        line = (f"{n:8} {m:13.2f} {h:7.2f} {f(lambda r: r.late):5.1f} "
                f"{f(lambda r: r.sat_loss):8.2f} {f(lambda r: r.station_visits):8.1f} "
                f"{f(lambda r: r.swaps):6.1f} {f(lambda r: r.swap_wait):8.1f} | ")
        if pi[n]:
            pm, ph = ci95([r.cost for r in pi[n]])
            line += f"{pm:12.2f} {ph:7.2f} {st.mean(r.late for r in pi[n]):5.1f}"
        print(line)

    base = a.policies[0]
    print(f"\nPaired difference in dynamic cost against '{base}' (same schedule, same seed):")
    for n in a.policies[1:]:
        diffs = [x.cost - y.cost for x, y in zip(dyn[n], dyn[base])]
        m, h = ci95(diffs)
        wins = sum(1 for d in diffs if d < -1e-9)
        ties = sum(1 for d in diffs if abs(d) <= 1e-9)
        verdict = "significant at 5%" if abs(m) > h else "not significant"
        print(f"  {n:8} mean {m:+9.2f}  95% CI [{m - h:+.2f}, {m + h:+.2f}]  "
              f"cheaper on {wins}/{len(diffs)} days, tied on {ties}  -> {verdict}")

    if a.bench_iters:
        print("\nValue of information (dynamic minus perfect-information, paired by seed):")
        for n in a.policies:
            diffs = [x.cost - y.cost for x, y in zip(dyn[n], pi[n])]
            m, h = ci95(diffs)
            base_cost = st.mean(y.cost for y in pi[n])
            print(f"  {n:8} mean {m:+9.2f}  95% CI [{m - h:+.2f}, {m + h:+.2f}]  "
                  f"= {100 * m / base_cost:.1f}% of the benchmark cost")
        print(f"  (the dynamic schedule beat the perfect-information plan on {swapped} of "
              f"{a.seeds * len(a.policies)} rule-days and was used as the benchmark there)")


if __name__ == "__main__":
    main()
