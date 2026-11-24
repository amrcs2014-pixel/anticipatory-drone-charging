"""Simple rules against the learned policy on the same held-out days.

    python compare_baselines.py results/models_C75_R5_ugv1_nearest.pkl C75_R5 --ugv 1 --days 60
"""
import argparse, json, os, pickle, statistics as st, sys
from drp.params import Params
from drp.policies import regret_policy, nearest_policy, fastest_policy
from drp.dynamic import execute
from drp.learn import ci95
from run_rollout import load_instance, HERE

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("instance")
ap.add_argument("--scale", type=float, default=0.25); ap.add_argument("--battery", type=float, default=100.0)
ap.add_argument("--pads", type=int, default=1); ap.add_argument("--ugv", type=int, default=1)
ap.add_argument("--dod", type=float, default=0.5); ap.add_argument("--days", type=int, default=30)
a = ap.parse_args()
models = pickle.load(open(a.model, "rb"))
P = Params(dist_scale=a.scale, E=a.battery, pads=a.pads, n_ugv=a.ugv, infeasible_penalty=200.0)
inst = load_instance(a.instance, None)
tag = f"{inst.name}_s{a.scale}_E{a.battery:g}_p{a.pads}_u{a.ugv}_d{a.dod}_i20_e10"
scheds = {int(k): [(t, r) for t, r in v] for k, v in
          json.load(open(os.path.join(HERE, "results", f"schedules_{tag}.json"))).items()}
days = [s for s in range(a.days) if s in scheds]
run = lambda pol, **kw: [execute(inst, P, pol(), scheds[s], **kw) for s in days]
res = {"regret": run(lambda: regret_policy(P.xi1, P.xi2)), "nearest": run(nearest_policy),
       "fastest": run(fastest_policy),
       "ceiling(fastest)": run(fastest_policy, rollout="all")}
c = {n: [r.cost for r in rows] for n, rows in res.items()}
fits = [run(nearest_policy, chooser=m.chooser()) for m in models]
c["learned"] = [st.mean(f[d].cost for f in fits) for d in range(len(days))]
late = {n: st.mean(r.late for r in rows) for n, rows in res.items()}
late["learned"] = st.mean(r.late for f in fits for r in f)
def pair(x, y):
    d = [u - v for u, v in zip(c[x], c[y])]; m, h = ci95(d)
    return f"{m:+8.1f} ({100*m/st.mean(c[y]):+5.1f}%) [{m-h:+7.1f}, {m+h:+7.1f}]{'*' if abs(m) > h else ' '}"
print(f"{inst.name:9} ugv {a.ugv} dod {a.dod} scale {a.scale} E {a.battery:g}, {len(days)} days | cost (late): "
      + ", ".join(f"{n} {st.mean(c[n]):.0f} ({late[n]:.2f})" for n in ("regret", "nearest", "fastest", "learned", "ceiling(fastest)")))
print(f"   fastest - nearest : {pair('fastest', 'nearest')}")
print(f"   learned - fastest : {pair('learned', 'fastest')}")
print(f"   ceiling - fastest : {pair('ceiling(fastest)', 'fastest')}")
