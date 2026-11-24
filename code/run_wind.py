"""Robustness to wind (Section 6.7).

    python run_wind.py C75_R5 --days 100 --model results/models_C75_R5_ugv1_fastest.pkl
    python run_wind.py AMMAN75_R5 --days 100 --model results/models_AMMAN75_R5_ugv1_fastest.pkl

Wind: one year of hourly wind at 100 m for Amman (ERA5 through Open-Meteo, case/wind_amman_2025.json).
The drone holds its airspeed, so the flight time and the energy of a leg are both multiplied by
airspeed / ground speed.  Days whose wind exceeds 12 m/s in the flying window are not flown.

Part A  the cached plans of the paper (no wind in the model) flown under wind.
Part B  each day re-planned with a wind-aware energy model: the wind observed at the start of
        the day, held constant, with a 10% reserve.  The same model triggers charging during the
        day.  The charging rules are then compared on identical days and identical wind.
        The ceiling knows the schedule and the wind of the whole day.
A leg on which the battery runs out is counted and penalised like a late order.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import statistics as st

from drp.params import Params
from drp.policies import regret_policy, nearest_policy, fastest_policy
from drp.dynamic import release_times, plan_day
from drp.simulate import DaySim
from run_rollout import load_instance, ci95, HERE

VA, VG_MIN, GROUND_LIMIT, RESERVE = 22.0, 5.0, 12.0, 0.10
BASE = dict(dist_scale=0.25, E=100.0, pads=1, n_ugv=1, infeasible_penalty=200.0)
WIND = os.path.join(HERE, "case", "wind_amman_2025.json")


class Wind:
    def __init__(self, path=WIND):
        h = json.load(open(path))["hourly"]
        self.speed, self.dirn = h["wind_speed_100m"], h["wind_direction_100m"]
        self.n_days = len(self.speed) // 24

    def window_max(self, day, hour0, hours=5):
        return max(self.speed[day * 24 + hour0:day * 24 + hour0 + hours])

    def vector(self, day, hour0, t_min):
        """(east, north) in m/s, t_min minutes after hour0; linear between the hourly values."""
        x = day * 24 + hour0 + t_min / 60.0
        a = min(int(x), len(self.speed) - 2)
        w, out = x - a, [0.0, 0.0]
        for k, wt in ((a, 1 - w), (a + 1, w)):
            th = math.radians(self.dirn[k])          # direction the wind comes from
            out[0] -= wt * self.speed[k] * math.sin(th)
            out[1] -= wt * self.speed[k] * math.cos(th)
        return out


def multiplier(u, v, dx, dy):
    par, cross = u * dx + v * dy, u * dy - v * dx
    return VA / max(math.sqrt(max(VA * VA - cross * cross, 0.0)) + par, VG_MIN)


class D(float):
    """A distance that carries the energy factor the planner assumes for its leg."""
    def __new__(cls, v, q):
        o = float.__new__(cls, v)
        o.q = q
        return o


class QParams(Params):
    def fly_wh(self, d_units, load):
        return Params.fly_wh(self, d_units, load) * getattr(d_units, "q", 1.0)


def aware_instance(name, wind, wd, h0, reserve=RESERVE):
    """Instance whose distances carry the factor under the wind observed at the start of the day."""
    inst = load_instance(name, None)
    n = len(inst.nodes)
    u, v = wind.vector(wd, h0, 0.0)

    def q(i, j):
        a, b = inst.nodes[i], inst.nodes[j]
        dx, dy = b.x - a.x, b.y - a.y
        r = math.hypot(dx, dy)
        return 1.0 if r < 1e-12 else multiplier(u, v, dx / r, dy / r) * (1.0 + reserve)
    refuges = [inst.depot] + inst.stations
    ref = [min(refuges, key=lambda r: inst.d[i][r]) for i in range(n)]
    inst.ref_d = [D(inst.d[i][ref[i]], q(i, ref[i])) for i in range(n)]
    inst.d = [[D(inst.d[i][j], q(i, j)) for j in range(n)] for i in range(n)]
    return inst


class WindSim(DaySim):
    """self.P decides (planner's model); flights obey the true physics and the wind."""

    def setup(self, true_P, wind, wd, h0):
        self.true, self.wind, self.wd, self.h0 = true_P, wind, wd, h0
        self.inflight = 0

    def clone(self):
        c = super().clone()
        c.__class__ = WindSim
        c.true, c.wind, c.wd, c.h0, c.inflight = self.true, self.wind, self.wd, self.h0, self.inflight
        return c

    def _leg(self, frm, to, t, E, load):
        a, b = self.inst.nodes[frm], self.inst.nodes[to]
        dx, dy = b.x - a.x, b.y - a.y
        r = math.hypot(dx, dy)
        m = 1.0
        if r > 1e-12 and self.wind is not None:
            u, v = self.wind.vector(self.wd, self.h0, t)
            m = multiplier(u, v, dx / r, dy / r)
        dist = float(self.inst.d[frm][to])
        E2 = E - self.true.fly_wh(dist, load) * m
        if E2 < -1e-6:
            self.inflight += 1
            self.res.energy_fail += 1
            E2 = 0.0
        return E2, t + self.true.fly_min(dist) * m

    def _fly(self, k, frm, to, t, E):
        dr = self.drones[k]
        dr["dist"] += float(self.inst.d[frm][to])
        E2, t2 = self._leg(frm, to, t, E, dr["load"])
        self._push(t2, "arrive", k, (to, E2))

    def _apply(self, k, i, j, t_done, E, o):
        if o["kind"] == "ugv":
            return super()._apply(k, i, j, t_done, E, o)
        f, dr = o["f"], self.drones[k]
        dr["dist"] += float(self.inst.d[i][f])
        self.res.charged_after[i] = ("fcs", 0.0)
        E2, t2 = self._leg(i, f, t_done, E, dr["load"])
        self._push(t2, "station", k, (f, E2, i))


def fly(inst, P, policy, sched, wind, wd, h0, rollout=None, chooser=None, rel=None):
    sim = WindSim(inst, P, policy)
    sim.setup(Params(**BASE), wind, wd, h0)
    sim.rollout, sim.chooser = rollout, chooser
    if rel is not None:
        sim.pending_orders = [(rel[c], t, inst.nodes[c].e, c) for t, r in sched for c in r]
    sched = sorted(sched, key=lambda x: x[0])
    for n, (t, r) in enumerate(sched):
        sim.future = sched[n:]
        sim.advance(t - 1e-9)
        sim.future = sched[n + 1:]
        sim.dispatch(r, t)
    sim.future = []
    sim.advance()
    res = sim.result()
    return {"cost": res.cost, "late": res.late, "inflight": sim.inflight,
            "stranded": res.energy_fail - sim.inflight, "swaps": res.swaps,
            "stations": res.station_visits, "wait": res.station_wait + res.swap_wait}


_wind = None


def wind_data():
    global _wind
    _wind = _wind or Wind()
    return _wind


def plan_one(job):
    name, seed, wd, h0 = job
    inst = aware_instance(name, wind_data(), wd, h0)
    P = QParams(**BASE)
    rel = release_times(inst, P, 0.5, seed=seed, epoch=10.0)
    sched, _ = plan_day(inst, P, regret_policy(P.xi1, P.xi2), rel, 10.0, 20, seed)
    return seed, sched


def eval_one(job):
    name, seed, wd, h0, sched, model_path = job
    wind = wind_data()
    inst = aware_instance(name, wind, wd, h0)
    P = QParams(**BASE)
    out = {"regret": fly(inst, P, regret_policy(P.xi1, P.xi2), sched, wind, wd, h0),
           "nearest": fly(inst, P, nearest_policy(), sched, wind, wd, h0),
           "fastest": fly(inst, P, fastest_policy(), sched, wind, wd, h0),
           "ceiling": fly(inst, P, fastest_policy(), sched, wind, wd, h0, rollout="all")}
    if model_path:
        models = pickle.load(open(model_path, "rb"))
        models = models if isinstance(models, list) else [models]
        rel = release_times(inst, P, 0.5, seed=seed, epoch=10.0)
        fits = [fly(inst, P, fastest_policy(), sched, wind, wd, h0, chooser=m.chooser(), rel=rel)
                for m in models]
        out["learned"] = {k: st.mean(f[k] for f in fits) for k in fits[0]}
    return seed, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("--days", type=int, default=100)
    ap.add_argument("--first-day", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    wind = wind_data()
    days = list(range(a.first_day, a.first_day + a.days))
    draw = {}
    for d in days:                      # one real wind day and start hour per simulated day
        rng = random.Random(1000 + d)
        while True:
            wd, h0 = rng.randrange(wind.n_days - 1), rng.choice([8, 10, 12, 14])
            if wind.window_max(wd, h0) <= GROUND_LIMIT:
                break
        draw[d] = (wd, h0)
    print(f"{a.instance} | days {days[0]}..{days[-1]} | strongest wind in the flying window, m/s: "
          f"median {st.median(wind.window_max(*draw[d]) for d in days):.1f}, "
          f"max {max(wind.window_max(*draw[d]) for d in days):.1f}")

    # ---- Part A: the paper's plans, no wind in the model --------------------------------
    base = json.load(open(os.path.join(HERE, "results",
                                       f"schedules_{a.instance}_s0.25_E100_p1_u1_d0.5_i20_e10.json")))
    inst0, P0 = load_instance(a.instance, None), Params(**BASE)
    calm = [fly(inst0, P0, fastest_policy(), base[str(d)], None, 0, 0) for d in days]
    rows = [fly(inst0, P0, fastest_policy(), base[str(d)], wind, *draw[d]) for d in days]
    hit = sum(1 for r in rows if r["inflight"])
    print(f"\nA. plans and charging rule without wind in the model (rule: fastest)\n"
          f"   no wind  : cost {st.mean(r['cost'] for r in calm):8.1f}, late/day {st.mean(r['late'] for r in calm):.2f}\n"
          f"   with wind: cost {st.mean(r['cost'] for r in rows):8.1f}, late/day {st.mean(r['late'] for r in rows):.2f}, "
          f"battery ran out in flight {st.mean(r['inflight'] for r in rows):.2f} times a day, "
          f"on {hit} of {len(rows)} days ({100 * hit / len(rows):.0f}%)")

    # ---- Part B: wind-aware plans, rules compared -----------------------------------------
    cache = os.path.join(HERE, "results", f"wind_schedules_{a.instance}.json")
    plans = json.load(open(cache)) if os.path.exists(cache) else {}
    todo = [(a.instance, d, *draw[d]) for d in days if str(d) not in plans]
    with mp.Pool(a.workers) as pool:
        if todo:
            for d, s in pool.imap_unordered(plan_one, todo):
                plans[str(d)] = s
            json.dump(plans, open(cache, "w"))
        res = dict(pool.imap_unordered(eval_one, [(a.instance, d, *draw[d], plans[str(d)], a.model)
                                                  for d in days]))
    json.dump({str(d): res[d] for d in days},
              open(os.path.join(HERE, "results", f"wind_eval_{a.instance}.json"), "w"))
    rules = list(res[days[0]])
    print(f"\nB. wind-aware plans (wind at the start of the day, {int(100 * RESERVE)}% reserve), {len(days)} days")
    print(f"   {'rule':8} {'cost':>8} {'late/day':>8} {'out in flight/day':>17} {'days hit':>8} "
          f"{'no option/day':>13} {'stations':>8} {'swaps':>6} {'wait min':>8}")
    for n in rules:
        f = lambda k: st.mean(res[d][n][k] for d in days)
        hit = sum(1 for d in days if res[d][n]["inflight"] > 0)
        print(f"   {n:8} {f('cost'):8.1f} {f('late'):8.2f} {f('inflight'):17.3f} {hit:8d} "
              f"{f('stranded'):13.2f} {f('stations'):8.1f} {f('swaps'):6.1f} {f('wait'):8.1f}")

    def pair(x, y):
        dif = [res[d][x]["cost"] - res[d][y]["cost"] for d in days]
        m, h = ci95(dif)
        better = sum(1 for v in dif if v < -1e-9)
        worse = sum(1 for v in dif if v > 1e-9)
        ref = st.mean(res[d][y]["cost"] for d in days)
        return (f"{m:+8.1f} ({100 * m / ref:+5.1f}%) [{m - h:+7.1f}, {m + h:+7.1f}]"
                f"{' significant' if abs(m) > h else ' n.s.'}; better on {better}, worse on {worse}")
    print("\n   paired differences in cost (same day, same plan, same wind):")
    pairs = [("fastest", "regret"), ("fastest", "nearest"), ("ceiling", "fastest")]
    if "learned" in rules:
        pairs.append(("learned", "fastest"))
    for x, y in pairs:
        print(f"   {x:8} - {y:8}: {pair(x, y)}")


if __name__ == "__main__":
    main()
