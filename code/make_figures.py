"""Figures for the manuscript (PNG at 300 dpi and PDF, in code/figures).

    python make_figures.py

Figure 1  service area of C75_R5 at the widened scale
Figure 3  paired cost difference against the fastest rule, three settings
Figure 2  attainable gain against lateness, twelve settings
"""
import json
import os
import pickle
import re
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drp.params import Params
from drp.policies import regret_policy, nearest_policy, fastest_policy
from drp.dynamic import execute
from drp.learn import ci95
from run_rollout import load_instance, HERE

OUT = os.path.join(HERE, "figures")
BLUE, ORANGE = "#2a78d6", "#eb6834"          # validated pair (light surface)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 300,
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)


def schedules(inst, P, ugv, dod=0.5):
    tag = f"{inst.name}_s{P.dist_scale}_E{P.E:g}_p1_u{ugv}_d{dod}_i20_e10"
    return {int(k): [(t, r) for t, r in v] for k, v in
            json.load(open(os.path.join(HERE, "results", f"schedules_{tag}.json"))).items()}


def figure1(name="C75_R5", title="Service area of C75_R5 at 0.25 km per unit", out="fig1_service_area",
            cust="Customer (75)", stat="Fixed charging station"):
    P = Params(dist_scale=0.25)
    inst = load_instance(name, None)
    km = lambda i: (P.km(inst.nodes[i].x), P.km(inst.nodes[i].y))
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    cx, cy = zip(*[km(i) for i in inst.customers])
    ax.scatter(cx, cy, s=14, color=MUTED, alpha=0.75, linewidths=0, label=cust)
    sx, sy = zip(*[km(i) for i in inst.stations if inst.d[i][inst.depot] > 1e-9])
    ax.scatter(sx, sy, s=70, marker="^", color=BLUE, edgecolors="white", linewidths=1.0,
               label=stat, zorder=3)
    dx, dy = km(inst.depot)
    ax.scatter([dx], [dy], s=80, marker="s", color=INK, edgecolors="white", linewidths=1.0,
               label="Depot (with a station)", zorder=4)
    for r, text, ls in ((P.E / (P.Pa * P.m + P.Pb) * P.speed_kmh / 2, "empty drone", "--"),
                        (P.E / (P.Pa * (P.m + P.Q) + P.Pb) * P.speed_kmh / 2, "full payload", ":")):
        ax.add_patch(plt.Circle((dx, dy), r, fill=False, color=ORANGE, lw=1.3, ls=ls, zorder=2))
        ax.plot([], [], color=ORANGE, lw=1.3, ls=ls,
                label=f"Out-and-back reach, {text}: {r:.1f} km")
    ax.set_aspect("equal")
    ax.set_xlabel("km")
    ax.set_ylabel("km")
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    save(fig, out)


def figure2():
    settings = [("C75_R5", "models_C75_R5_ugv1_fastest.pkl", 120, "C75_R5, 120 fresh days"),
                ("RC75_R4", "models_RC75_R4_ugv1_fastest_400.pkl", 120, "RC75_R4, 120 held-out days"),
                ("RC100_R5", "models_RC100_R5_ugv1_fastest.pkl", 100, "RC100_R5, 100 held-out days")]
    P = Params(dist_scale=0.25, pads=1, n_ugv=1, infeasible_penalty=200.0)
    order = ["Regret rule", "Nearest rule", "Learned policy", "Ceiling (knows the schedule)"]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9), sharey=True)
    summary = {}
    for ax, (name, mfile, ndays, title) in zip(axes, settings):
        inst = load_instance(name, None)
        sch = schedules(inst, P, 1)
        first = 300 if name == "C75_R5" else 0      # C75_R5: days no model variant ever saw
        days = [s for s in range(first, first + ndays) if s in sch]
        models = pickle.load(open(os.path.join(HERE, "results", mfile), "rb"))
        run = lambda pol, **kw: [execute(inst, P, pol(), sch[s], **kw).cost for s in days]
        fast = run(fastest_policy)
        fits = [run(fastest_policy, chooser=m.chooser()) for m in models]
        cost = {"Regret rule": run(lambda: regret_policy(P.xi1, P.xi2)),
                "Nearest rule": run(nearest_policy),
                "Learned policy": [st.mean(f[d] for f in fits) for d in range(len(days))],
                "Ceiling (knows the schedule)": run(fastest_policy, rollout="all")}
        base = st.mean(fast)
        summary[name] = {}
        for y, k in enumerate(order):
            m, h = ci95([100 * (a - b) / base for a, b in zip(cost[k], fast)])
            summary[name][k] = (m, m - h, m + h)
            col = BLUE if k == "Learned policy" else MUTED
            ax.plot([m - h, m + h], [y, y], color=col, lw=2, solid_capstyle="round")
            ax.plot([m], [y], "o", color=col, ms=7, mec="white", mew=1.2)
            ax.annotate(f"{m:+.1f}%", (m, y), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8, color=INK)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, color=INK)
        ax.set_ylim(len(order) - 0.5, -0.6)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    fig.supxlabel("Cost difference against the fastest rule (%)", fontsize=9, color=INK)
    fig.suptitle("Daily cost relative to the fastest rule: mean paired difference with 95% confidence interval",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "fig3_paired_differences")
    json.dump(summary, open(os.path.join(OUT, "fig3_values.json"), "w"), indent=1)
    return summary


def figure3():
    """Ceiling gain against lateness under the fastest rule, from results/paper_tables.md."""
    txt = open(os.path.join(HERE, "results", "paper_tables.md"), encoding="utf-8").read()
    a, b = txt.split("### Table B")
    late, gain = {}, {}
    for line in a.splitlines():
        p = [c.strip() for c in line.split("|")]
        if len(p) > 7 and p[2].isdigit():
            late[p[1]] = float(re.search(r"\(([\d.]+)\)", p[5]).group(1))      # fastest column
    for line in b.splitlines():
        p = [c.strip() for c in line.split("|")]
        if len(p) > 6 and p[1] in late:
            gain[p[1]] = -float(re.match(r"([-+][\d.]+)%", p[5]).group(1))      # ceiling vs fastest
    trained = {"C75_R5, 1 UGV, 50% dynamic": (2.52, 2.5), "RC75_R4, 1 UGV, 50% dynamic": (3.07, 3.1),
               "RC100_R5, 1 UGV, 50% dynamic": (6.69, 5.5)}
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    xs, ys = [late[k] for k in late], [gain[k] for k in late]
    ax.scatter(xs, ys, s=42, color=BLUE, edgecolors="white", linewidths=1.0, zorder=3,
               label="Ceiling: rollout that knows the schedule (12 settings, 30 to 60 days)")
    tx, ty = zip(*trained.values())
    ax.scatter(tx, ty, s=48, marker="D", color=ORANGE, edgecolors="white", linewidths=1.0, zorder=4,
               label="Learned policy trained in the setting (100 to 120 held-out days)")
    for k, (x, y) in trained.items():
        ax.annotate(k.split(",")[0], (x, y), xytext=(7, -3), textcoords="offset points",
                    fontsize=7.5, color=MUTED)
    ax.set_xlabel("Late orders per day under the fastest rule")
    ax.set_ylabel("Cost reduction against the fastest rule (%)")
    ax.set_xlim(left=-0.3)
    ax.set_ylim(bottom=0)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=7.5)
    ax.set_title("What a better charging decision can gain grows with congestion",
                 loc="left", fontsize=10, color=INK)
    save(fig, "fig2_gain_vs_lateness")
    return dict(late=late, gain=gain)


def figure4(day=300):
    """One simulated day of C75_R5 under the fastest rule: routes and charging stops."""
    from drp import simulate as S
    P = Params(dist_scale=0.25, pads=1, n_ugv=1, infeasible_penalty=200.0)
    inst = load_instance("C75_R5", None)
    sch = schedules(inst, P, 1)[day]
    stops = []
    original = S.DaySim._apply

    def logged(self, k, i, j, t_done, E, o):
        stops.append((i, j, o["kind"], o.get("f")))
        return original(self, k, i, j, t_done, E, o)
    S.DaySim._apply = logged
    try:
        res = execute(inst, P, fastest_policy(), sch)
    finally:
        S.DaySim._apply = original
    km = lambda i: (P.km(inst.nodes[i].x), P.km(inst.nodes[i].y))
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    for _, route in sch:
        xs, ys = zip(*[km(n) for n in [inst.depot] + list(route) + [inst.depot]])
        ax.plot(xs, ys, color="#c9c8c3", lw=0.6, zorder=1)
    ax.plot([], [], color="#c9c8c3", lw=0.8, label=f"Planned drone route ({len(sch)})")
    cx, cy = zip(*[km(i) for i in inst.customers])
    ax.scatter(cx, cy, s=10, color=MUTED, linewidths=0, zorder=2, label="Customer")
    n_f = n_u = 0
    for i, j, kind, f in stops:
        if kind == "fcs":
            xs, ys = zip(km(i), km(f), km(j))
            ax.plot(xs, ys, color=BLUE, lw=1.3, zorder=3)
            n_f += 1
        else:
            ax.scatter(*km(i), s=48, marker="D", color=ORANGE, edgecolors="white", linewidths=0.9, zorder=5)
            n_u += 1
    ax.plot([], [], color=BLUE, lw=1.3, label=f"Detour to a fixed station ({n_f})")
    ax.scatter([], [], s=48, marker="D", color=ORANGE, edgecolors="white", label=f"Battery swap by the UGV ({n_u})")
    sx, sy = zip(*[km(i) for i in inst.stations if inst.d[i][inst.depot] > 1e-9])
    ax.scatter(sx, sy, s=70, marker="^", color=BLUE, edgecolors="white", linewidths=1.0, zorder=4,
               label="Fixed charging station")
    ax.scatter(*km(inst.depot), s=80, marker="s", color=INK, edgecolors="white", linewidths=1.0, zorder=6,
               label="Depot")
    ax.set_aspect("equal")
    ax.set_xlabel("km")
    ax.set_ylabel("km")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=8)
    ax.set_title("One day of C75_R5 under the fastest rule", loc="left", fontsize=10, color=INK)
    save(fig, "fig4_one_day")
    return dict(routes=len(sch), station=n_f, ugv=n_u, cost=res.cost, late=res.late,
                visits=res.station_visits, swaps=res.swaps)


if __name__ == "__main__":
    import sys
    if "--case" in sys.argv:
        figure1("AMMAN75_R5", "Amman case: real locations, natural scale", "fig5_amman_case",
                "Pharmacy (75 customers)", "Charging station at a fuel station")
        sys.exit()
    if "--only4" in sys.argv:
        print("figure 4:", figure4())
        sys.exit()
    figure1()
    print("figure 1 done")
    s = figure2()
    for name, d in s.items():
        print(name, {k: f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]" for k, (m, lo, hi) in d.items()})
    f3 = figure3()
    print("figure 3 points:", sorted((round(f3["late"][k], 2), round(f3["gain"][k], 1)) for k in f3["late"]))
