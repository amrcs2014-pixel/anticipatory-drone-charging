"""Apply already trained models to cached days of any setting (no retraining).

    python eval_policy.py results/models_C75_R5_ugv1_nearest.pkl RC75_R4 --ugv 1 --days 30
"""
import argparse
import json
import os
import pickle
import statistics as st
import sys

from drp.params import Params
from drp.policies import regret_policy, nearest_policy, fastest_policy
from drp.dynamic import execute, release_times
from drp.learn import ci95
from run_rollout import load_instance, HERE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("instance")
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--battery", type=float, default=100.0)
    ap.add_argument("--pads", type=int, default=1)
    ap.add_argument("--ugv", type=int, default=1)
    ap.add_argument("--dod", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--epoch", type=float, default=10.0)
    ap.add_argument("--late-penalty", type=float, default=200.0)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--first-day", type=int, default=0, help="first cached day (seed) to use")
    a = ap.parse_args()
    models = pickle.load(open(a.model, "rb"))
    models = models if isinstance(models, list) else [models]
    P = Params(dist_scale=a.scale, E=a.battery, pads=a.pads, n_ugv=a.ugv,
               infeasible_penalty=a.late_penalty)
    inst = load_instance(a.instance, None)
    tag = (f"{inst.name}_s{a.scale}_E{a.battery:g}_p{a.pads}_u{a.ugv}_d{a.dod}"
           f"_i{a.iters}_e{a.epoch:g}")
    cache = os.path.join(HERE, "results", f"schedules_{tag}.json")
    if not os.path.exists(cache):
        sys.exit(f"no cached schedules: {cache}")
    scheds = {int(k): [(t, r) for t, r in v] for k, v in json.load(open(cache)).items()}
    days = [s for s in range(a.first_day, a.first_day + a.days) if s in scheds]
    makers = {"regret": lambda: regret_policy(P.xi1, P.xi2), "nearest": nearest_policy,
              "fastest": fastest_policy}
    bname = getattr(models[0], "base", "regret")
    base = makers[bname]
    out = {
        "regret": [execute(inst, P, makers["regret"](), scheds[s]) for s in days],
        "nearest": [execute(inst, P, makers["nearest"](), scheds[s]) for s in days],
        "fastest": [execute(inst, P, makers["fastest"](), scheds[s]) for s in days],
        "rollout-all": [execute(inst, P, base(), scheds[s], rollout="all") for s in days],
    }
    fc = getattr(models[0], "forecast", False)
    rels = {s: release_times(inst, P, a.dod, seed=s, epoch=a.epoch) if fc else None for s in days}
    fits = [[execute(inst, P, base(), scheds[s], chooser=m.chooser(), rel=rels[s]) for s in days]
            for m in models]
    c = {n: [r.cost for r in rows] for n, rows in out.items()}
    c["learned"] = [st.mean(f[d].cost for f in fits) for d in range(len(days))]

    def pair(x, y):
        d = [u - v for u, v in zip(c[x], c[y])]
        m, h = ci95(d)
        return (f"{m:+8.1f} ({100 * m / st.mean(c[y]):+5.1f}%) [{m - h:+7.1f}, {m + h:+7.1f}]"
                f"{'*' if abs(m) > h else ' '}")
    gap = st.mean(c[bname]) - st.mean(c["rollout-all"])
    got = st.mean(c[bname]) - st.mean(c["learned"])
    print(f"{inst.name:9} ugv {a.ugv} dod {a.dod} scale {a.scale} E {a.battery:g} | "
          f"late: regret {st.mean(r.late for r in out['regret']):5.2f}, nearest "
          f"{st.mean(r.late for r in out['nearest']):5.2f}, learned "
          f"{st.mean(r.late for f in fits for r in f):5.2f}\n"
          f"   learned - fastest: {pair('learned', 'fastest')}   "
          f"   learned - nearest: {pair('learned', 'nearest')}\n"
          f"   learned - regret : {pair('learned', 'regret')}\n"
          f"   ceiling - {bname:7}: {pair('rollout-all', bname)}   recovered {100 * got / gap:4.0f}%")


if __name__ == "__main__":
    main()
