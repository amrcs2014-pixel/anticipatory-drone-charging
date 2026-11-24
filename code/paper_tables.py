"""Tables for the manuscript: all rules on the same days in every setting.

The learned policy is the five-fit model trained on C75_R5 with one UGV
(deviating from the fastest rule); in every other setting it is applied
without retraining.

    python paper_tables.py results/models_C75_R5_ugv1_fastest.pkl > results/paper_tables.md
"""
import json
import os
import pickle
import statistics as st
import sys

from drp.params import Params
from drp.policies import regret_policy, nearest_policy, fastest_policy
from drp.dynamic import execute
from drp.learn import ci95
from run_rollout import load_instance, HERE

SETTINGS = [  # instance, ugv, dod, scale, battery, days
    ("C75_R5", 1, 0.5, 0.25, 100.0, 60),
    ("C75_R5", 1, 0.25, 0.25, 100.0, 30),
    ("C75_R5", 1, 0.75, 0.25, 100.0, 30),
    ("C75_R5", 2, 0.5, 0.25, 100.0, 30),
    ("C75_R3", 1, 0.5, 0.25, 100.0, 30),
    ("RC75_R4", 1, 0.5, 0.25, 100.0, 30),
    ("RC75_R4", 2, 0.5, 0.25, 100.0, 30),
    ("RC100_R5", 1, 0.5, 0.25, 100.0, 30),
    ("R75_R3", 1, 0.5, 0.25, 100.0, 30),
    ("R75_R3", 2, 0.5, 0.25, 100.0, 30),
    ("R100_R4", 1, 0.5, 0.25, 100.0, 30),
    ("C75_R5", 1, 0.5, 0.1, 45.0, 30),
]


def main():
    models = pickle.load(open(sys.argv[1], "rb"))
    rows = []
    settings = SETTINGS
    if len(sys.argv) > 2:      # e.g. "AMMAN75_R5,1,0.5,0.25,100,100;AMMAN150_R5,1,0.5,0.25,100,60"
        settings = [(a, int(b), float(c), float(d), float(e), int(f)) for a, b, c, d, e, f in
                    (x.split(",") for x in sys.argv[2].split(";"))]
    for name, ugv, dod, scale, bat, ndays in settings:
        P = Params(dist_scale=scale, E=bat, pads=1, n_ugv=ugv, infeasible_penalty=200.0)
        inst = load_instance(name, None)
        tag = f"{inst.name}_s{scale}_E{bat:g}_p1_u{ugv}_d{dod}_i20_e10"
        sch = {int(k): [(t, r) for t, r in v] for k, v in
               json.load(open(os.path.join(HERE, "results", f"schedules_{tag}.json"))).items()}
        days = [s for s in range(ndays) if s in sch]
        run = lambda pol, **kw: [execute(inst, P, pol(), sch[s], **kw) for s in days]
        res = {"regret": run(lambda: regret_policy(P.xi1, P.xi2)), "nearest": run(nearest_policy),
               "fastest": run(fastest_policy), "ceiling": run(fastest_policy, rollout="all"),
               "air": run(fastest_policy, rollout="air")}
        fits = [run(fastest_policy, chooser=m.chooser()) for m in models]
        c = {n: [r.cost for r in v] for n, v in res.items()}
        c["learned"] = [st.mean(f[d].cost for f in fits) for d in range(len(days))]
        late = {n: st.mean(r.late for r in v) for n, v in res.items()}
        late["learned"] = st.mean(r.late for f in fits for r in f)
        dec = st.mean(r.n_real_choices for r in res["fastest"])
        chg = st.mean(r.n_overrides for r in res["ceiling"])

        def pair(x, y):
            d = [u - v for u, v in zip(c[x], c[y])]
            m, h = ci95(d)
            return m, m - h, m + h, 100 * m / st.mean(c[y]), abs(m) > h
        rows.append(dict(name=name, ugv=ugv, dod=dod, scale=scale, bat=bat, n=len(days),
                         cost={k: st.mean(v) for k, v in c.items()}, late=late, dec=dec, chg=chg,
                         f_r=pair("fastest", "regret"), f_n=pair("fastest", "nearest"),
                         l_f=pair("learned", "fastest"), c_f=pair("ceiling", "fastest"),
                         a_f=pair("air", "fastest")))
        print(f"done {name} ugv {ugv} dod {dod} scale {scale}", file=sys.stderr, flush=True)

    lab = lambda r: (f"{r['name']}, {r['ugv']} UGV{'s' if r['ugv'] > 1 else ''}, "
                     f"{int(100 * r['dod'])}% dynamic"
                     + ("" if r["scale"] == 0.25 else ", original area, 45 Wh"))
    star = lambda p: "*" if p[4] else ""
    print("### Table A. Mean cost per day (late orders per day in brackets)\n")
    print("| Setting | Days | Regret | Nearest | Fastest | Learned | Ceiling |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        cell = lambda k: f"{r['cost'][k]:.1f} ({r['late'][k]:.2f})"
        print(f"| {lab(r)} | {r['n']} | {cell('regret')} | {cell('nearest')} | {cell('fastest')} "
              f"| {cell('learned')} | {cell('ceiling')} |")
    print("\n### Table B. Paired differences, percent of the reference rule's cost "
          "(95% CI of the difference in cost units; * = significant)\n")
    print("| Setting | Fastest vs regret | Fastest vs nearest | Learned vs fastest | Ceiling vs fastest "
          "| In-air rollout vs fastest | Decisions with a choice | Changed by ceiling |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        cell = lambda p: f"{p[3]:+.1f}%{star(p)} [{p[1]:+.0f}, {p[2]:+.0f}]"
        print(f"| {lab(r)} | {cell(r['f_r'])} | {cell(r['f_n'])} | {cell(r['l_f'])} | {cell(r['c_f'])} "
              f"| {cell(r['a_f'])} | {r['dec']:.1f} | {r['chg']:.1f} |")
    sig = lambda key, sign: sum(1 for r in rows if r[key][4] and r[key][0] * sign > 0)
    print(f"\nCounts over {len(rows)} settings: fastest significantly better than regret in "
          f"{sig('f_r', -1)}, worse in {sig('f_r', 1)}; than nearest in {sig('f_n', -1)}, worse in "
          f"{sig('f_n', 1)}; learned better than fastest in {sig('l_f', -1)}, worse in {sig('l_f', 1)}; "
          f"ceiling better than fastest in {sig('c_f', -1)}; in-air rollout better than fastest in "
          f"{sig('a_f', -1)}, worse in {sig('a_f', 1)}.")


if __name__ == "__main__":
    main()
