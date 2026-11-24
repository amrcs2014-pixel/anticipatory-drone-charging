"""Train the future-cost estimate and test it on held-out days.

The learned policy starts from a base rule (--base regret or nearest) and
learns when to deviate from it.  Uses the schedules cached by run_rollout.py
(same settings).  The first --test days are never used for training or for
choosing the margin.

    python run_rollout.py C75_R5 --seeds 300 --ugv 1      (plans the days)
    python train_policy.py C75_R5 --ugv 1 --base nearest --days 300 --test 60 --model-seeds 5
"""
import argparse
import json
import os
import pickle
import statistics as st
import sys
import time

from drp.params import Params
from drp.policies import regret_policy, nearest_policy, fastest_policy
from drp.dynamic import execute, release_times
from drp.learn import Dataset, FutureCostModel, ci95
from run_rollout import load_instance, HERE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--battery", type=float, default=100.0)
    ap.add_argument("--pads", type=int, default=1)
    ap.add_argument("--ugv", type=int, default=1)
    ap.add_argument("--dod", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--epoch", type=float, default=10.0)
    ap.add_argument("--late-penalty", type=float, default=200.0)
    ap.add_argument("--base", choices=["regret", "nearest", "fastest"], default="fastest",
                    help="rule the learned policy deviates from")
    ap.add_argument("--days", type=int, default=300, help="total cached days to use")
    ap.add_argument("--test", type=int, default=60, help="first days, held out")
    ap.add_argument("--val", type=int, default=30, help="next days, used only to choose the margin")
    ap.add_argument("--dagger", type=int, default=1)
    ap.add_argument("--model-seeds", type=int, default=5, help="independent model fits")
    ap.add_argument("--forecast", action="store_true",
                    help="give the model the orders already known but not yet dispatched")
    ap.add_argument("--save", default=None, help="write per-day test costs to this json file")
    ap.add_argument("--model-out", default=None, help="pickle the trained models here")
    a = ap.parse_args()

    P = Params(dist_scale=a.scale, E=a.battery, pads=a.pads, n_ugv=a.ugv,
               infeasible_penalty=a.late_penalty)
    inst = load_instance(a.instance, None)
    tag = (f"{inst.name}_s{a.scale}_E{a.battery:g}_p{a.pads}_u{a.ugv}_d{a.dod}"
           f"_i{a.iters}_e{a.epoch:g}")
    cache = os.path.join(HERE, "results", f"schedules_{tag}.json")
    if not os.path.exists(cache):
        sys.exit(f"no cached schedules: {cache}\nrun run_rollout.py with the same settings first")
    scheds = {int(k): [(t, r) for t, r in v] for k, v in json.load(open(cache)).items()}
    days = [s for s in range(a.days) if s in scheds]
    test, val, train = days[:a.test], days[a.test:a.test + a.val], days[a.test + a.val:]
    if len(train) < 20:
        sys.exit(f"only {len(train)} training days cached; plan more with run_rollout.py --seeds N")
    makers = {"regret": lambda: regret_policy(P.xi1, P.xi2), "nearest": nearest_policy,
              "fastest": fastest_policy}
    base = makers[a.base]
    rels = {}

    def ex(pol, s, **kw):
        if a.forecast and s not in rels:
            rels[s] = release_times(inst, P, a.dod, seed=s, epoch=a.epoch)
        return execute(inst, P, pol, scheds[s], rel=rels.get(s), **kw)
    print(inst, f"| ugv {a.ugv} | base rule {a.base} | train {len(train)} days, "
                f"margin chosen on {len(val)}, test {len(test)} held-out days, "
                f"{a.model_seeds} model fits")

    # 1. labels along the base rule's days and along the rollout's days ------------
    t0 = time.time()
    data = Dataset()
    for s in train:
        data.start_day(s)
        ex(base(), s, rollout="all", collector=data.collector, follow=False)
        ex(base(), s, rollout="all", collector=data.collector, follow=True)
    y = data.y
    print(f"collected {len(data)} labelled alternatives in {time.time() - t0:.0f}s: "
          f"{100 * sum(v < -1e-9 for v in y) / len(y):.0f}% better than the base choice, "
          f"{100 * sum(v > 1e-9 for v in y) / len(y):.0f}% worse")
    first = FutureCostModel().fit(data.X, data.y, forecast=a.forecast)
    for rnd in range(a.dagger):       # labels along the learned policy's own days
        before = len(data)
        for s in train:
            data.start_day(s)
            ex(base(), s, rollout="all", collector=data.collector,
                    chooser=first.labelled_chooser())
        first = FutureCostModel().fit(data.X, data.y, forecast=a.forecast)
        print(f"  round {rnd + 1}: +{len(data) - before} rows from the learned policy's days")

    # 2. several independent fits, margin chosen on the validation days ---------------
    base_val = [ex(base(), s).cost for s in val]
    models = []
    for k in range(a.model_seeds):
        m = FutureCostModel(random_state=k).fit(data.X, data.y, forecast=a.forecast)
        best = (None, 0.0)
        for margin in (0.0, 5.0, 10.0, 20.0, 40.0, 80.0):
            m.margin = margin
            cost = [ex(base(), s, chooser=m.chooser()).cost for s in val]
            saving = st.mean(b - c for b, c in zip(base_val, cost))
            if best[0] is None or saving > best[1]:
                best = (margin, saving)
        m.margin, m.base, m.forecast = best[0], a.base, a.forecast
        models.append(m)
        print(f"  fit {k}: margin {m.margin:g}, saving on validation days {best[1]:+.1f}")

    # 3. held-out test -------------------------------------------------------------------
    out = {
        "regret": [ex(makers["regret"](), s) for s in test],
        "nearest": [ex(makers["nearest"](), s) for s in test],
        "fastest": [ex(makers["fastest"](), s) for s in test],
        "rollout-all": [ex(base(), s, rollout="all") for s in test],
    }
    per_fit = [[ex(base(), s, chooser=m.chooser()) for s in test]
               for m in models]
    cost = {n: [r.cost for r in rows] for n, rows in out.items()}
    cost["learned"] = [st.mean(f[d].cost for f in per_fit) for d in range(len(test))]
    late = {n: st.mean(r.late for r in rows) for n, rows in out.items()}
    late["learned"] = st.mean(r.late for f in per_fit for r in f)
    print(f"\nHeld-out test, {len(test)} days (learned = mean over {len(models)} fits)")
    print(f"{'rule':12} {'cost':>9} {'+-95%':>7} {'late':>5}")
    for n in ("regret", "nearest", "fastest", "learned", "rollout-all"):
        m, h = ci95(cost[n])
        print(f"{n:12} {m:9.2f} {h:7.2f} {late[n]:5.2f}")

    def pair(x, y):
        d = [u - v for u, v in zip(cost[x], cost[y])]
        m, h = ci95(d)
        ref = st.mean(cost[y])
        return (f"mean {m:+8.2f} ({100 * m / ref:+5.1f}%)  95% CI [{m - h:+8.2f}, {m + h:+8.2f}]  "
                f"better {sum(v < -1e-9 for v in d)}, worse {sum(v > 1e-9 for v in d)}"
                f"  -> {'significant at 5%' if abs(m) > h else 'not significant'}")
    print("\nPaired differences:")
    print(f"  learned - fastest      {pair('learned', 'fastest')}")
    print(f"  learned - nearest      {pair('learned', 'nearest')}")
    print(f"  learned - regret       {pair('learned', 'regret')}")
    print(f"  nearest - regret       {pair('nearest', 'regret')}")
    print(f"  rollout-all - {a.base:8} {pair('rollout-all', a.base)}")
    print("\nEach fit separately, learned minus its base rule:")
    for k, f in enumerate(per_fit):
        d = [r.cost - c for r, c in zip(f, cost[a.base])]
        m, h = ci95(d)
        print(f"  fit {k}: {m:+8.2f}  [{m - h:+8.2f}, {m + h:+8.2f}]  "
              f"changed {st.mean(r.n_overrides for r in f):.1f} decisions a day")
    gap = st.mean(cost[a.base]) - st.mean(cost["rollout-all"])
    got = st.mean(cost[a.base]) - st.mean(cost["learned"])
    print(f"\nShare of the rollout-all gain over the {a.base} rule recovered: "
          f"{100 * got / gap:.0f}% ({got:.1f} of {gap:.1f})")
    print("\nMost important inputs (fit 0, share of total gain):")
    for g, n in models[0].importance()[:8]:
        print(f"  {n:24} {100 * g:5.1f}%")
    if a.model_out:
        pickle.dump(models, open(a.model_out, "wb"))
    if a.save:
        json.dump(cost, open(a.save, "w"))


if __name__ == "__main__":
    main()
