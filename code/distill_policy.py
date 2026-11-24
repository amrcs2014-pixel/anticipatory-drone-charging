"""Distil the learned charging policy into a small decision tree.

The tree answers, for each alternative to the base rule's choice: "switch to
this option?".  It is fitted to the learned policy's own decisions on the
training days and then flown on the held-out days.

    python distill_policy.py results/models_C75_R5_ugv1_nearest.pkl C75_R5 --ugv 1
"""
import argparse
import json
import os
import pickle
import statistics as st
import sys

from drp.params import Params
from drp.policies import regret_policy, nearest_policy, fastest_policy
from drp.dynamic import execute
from drp.learn import features, FEATURES, ci95
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
    ap.add_argument("--days", type=int, default=300)
    ap.add_argument("--test", type=int, default=60)
    ap.add_argument("--val", type=int, default=30)
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--tree-out", default=None)
    a = ap.parse_args()
    from sklearn.tree import DecisionTreeClassifier, export_text

    models = pickle.load(open(a.model, "rb"))
    model = models[0] if isinstance(models, list) else models
    P = Params(dist_scale=a.scale, E=a.battery, pads=a.pads, n_ugv=a.ugv,
               infeasible_penalty=a.late_penalty)
    inst = load_instance(a.instance, None)
    tag = (f"{inst.name}_s{a.scale}_E{a.battery:g}_p{a.pads}_u{a.ugv}_d{a.dod}"
           f"_i{a.iters}_e{a.epoch:g}")
    cache = os.path.join(HERE, "results", f"schedules_{tag}.json")
    if not os.path.exists(cache):
        sys.exit(f"no cached schedules: {cache}")
    scheds = {int(k): [(t, r) for t, r in v] for k, v in json.load(open(cache)).items()}
    days = [s for s in range(a.days) if s in scheds]
    test, train = days[:a.test], days[a.test + a.val:]
    makers = {"regret": lambda: regret_policy(P.xi1, P.xi2), "nearest": nearest_policy,
              "fastest": fastest_policy}
    bname = getattr(model, "base", "regret")
    base = makers[bname]

    # the learned policy's decisions on the training days --------------------------
    X, y = [], []
    inner = model.chooser()

    def recording(sim, k, i, j, t_done, E, options, b):
        choice = inner(sim, k, i, j, t_done, E, options, b)
        rows = features(sim, k, i, j, t_done, E, options, b)
        for n, r in enumerate(rows):
            if n != b:
                X.append(r)
                y.append(int(n == choice))
        return choice
    for s in train:
        execute(inst, P, base(), scheds[s], chooser=recording)
    print(inst, f"| base rule {bname} | {len(y)} alternatives seen on {len(train)} training days, "
                f"the learned policy switched to {sum(y)} of them ({100 * sum(y) / len(y):.1f}%)")

    def tree_chooser(tree):
        def choose(sim, k, i, j, t_done, E, options, b):
            rows = features(sim, k, i, j, t_done, E, options, b)
            alt = [n for n in range(len(options)) if n != b]
            p = tree.predict_proba([rows[n] for n in alt])[:, 1]
            best = max(range(len(alt)), key=lambda x: p[x])
            return alt[best] if p[best] > 0.5 else b
        return choose

    near = [execute(inst, P, base(), scheds[s]).cost for s in test]
    learned = [execute(inst, P, base(), scheds[s], chooser=model.chooser()).cost for s in test]
    print(f"\nHeld-out test, {len(test)} days.  base rule {st.mean(near):.2f}, "
          f"learned (fit 0) {st.mean(learned):.2f}")
    best = None
    for depth in a.depths:
        tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=20,
                                      class_weight="balanced", random_state=0).fit(X, y)
        rows = [execute(inst, P, base(), scheds[s], chooser=tree_chooser(tree)) for s in test]
        cost = [r.cost for r in rows]
        d = [c - n for c, n in zip(cost, near)]
        m, h = ci95(d)
        keep = (st.mean(near) - st.mean(cost)) / (st.mean(near) - st.mean(learned))
        print(f"  depth {depth}: cost {st.mean(cost):8.2f}  against the base rule {m:+8.2f} "
              f"[{m - h:+8.2f}, {m + h:+8.2f}]{'*' if abs(m) > h else ' '}  "
              f"keeps {100 * keep:3.0f}% of the learned policy's gain, "
              f"changes {st.mean(r.n_overrides for r in rows):.1f} decisions a day, "
              f"{tree.get_n_leaves()} leaves")
        if depth == a.depths[0] or (best and st.mean(cost) < best[0]):
            best = (st.mean(cost), depth, tree)
        elif best is None:
            best = (st.mean(cost), depth, tree)
    for depth in a.depths[:2]:
        tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=20,
                                      class_weight="balanced", random_state=0).fit(X, y)
        print(f"\nTree of depth {depth} (class 1 = switch to this alternative):")
        print(export_text(tree, feature_names=FEATURES, decimals=1))
    if a.tree_out:
        pickle.dump(best[2], open(a.tree_out, "wb"))


if __name__ == "__main__":
    main()
