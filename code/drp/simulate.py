"""Event-driven simulation of one planning day.

Given drone routes (customer sequences), the simulator flies the drones in
parallel from their dispatch times, applies the charging policy whenever a
drone cannot safely fly its next leg, and returns the objective value with a
trace.

Modelling choices that follow the base paper:
  * a drone leaves the depot with a full battery;
  * service starts at max(arrival, tolerable earliest); satisfaction is the
    fuzzy membership of the service start time (Eq. 13);
  * energy per leg depends on payload (Eq. 33); full, linear recharge (Eq. 30);
  * a UGV swaps the battery at the customer the drone is standing on, after
    the delivery is completed (Assumptions 4 and 5).
Simplifications, stated in the README:
  * charging is triggered reactively: only when the next leg plus a safety
    reserve to the nearest refuge is not covered;
  * UGVs are dispatched on request rather than on a pre-planned tour;
  * a landed drone uses no energy while waiting.
"""
from dataclasses import dataclass, field, replace
import heapq
import math

EPS = 1e-9


@dataclass
class SimResult:
    cost: float = 0.0
    travel_km: float = 0.0
    sat_loss: float = 0.0
    n_routes: int = 0
    violations: int = 0
    late: int = 0
    energy_fail: int = 0
    overload: int = 0
    station_visits: int = 0
    station_wait: float = 0.0
    swaps: int = 0
    swap_wait: float = 0.0
    n_decisions: int = 0   # charging decisions taken
    n_real_choices: int = 0  # ... of which had more than one option
    n_overrides: int = 0   # ... where rollout chose differently from the base rule
    filtered: int = 0    # charging options dropped because they would cause lateness
    ugv_km: float = 0.0
    makespan: float = 0.0
    mu: dict = field(default_factory=dict)            # customer -> satisfaction
    arr_energy: dict = field(default_factory=dict)    # customer -> Wh on arrival
    charged_after: dict = field(default_factory=dict) # customer -> (kind, wait)
    decisions: list = field(default_factory=list)     # trace of charging decisions

    @property
    def feasible(self) -> bool:
        return self.violations == 0


def satisfaction(t, e, l, tol):
    """Paper Eq. 13 with tolerable window [e - tol, l + tol]."""
    if e <= t <= l:
        return 1.0
    if e - tol <= t < e:
        return 1.0 - (e - t) / tol
    if l < t <= l + tol:
        return 1.0 - (t - l) / tol
    return 0.0


class DaySim:
    """One day of operations.  Routes can be dispatched at any time; stations
    and UGVs are shared by every drone in the air."""

    def __init__(self, inst, P, policy, record=False):
        self.inst, self.P, self.policy, self.record = inst, P, policy, record
        self.res = SimResult()
        self.events = []         # (time, seq, kind, drone, payload)
        self.seq = 0
        self.now = 0.0
        self.pads = None
        if P.pads is not None:
            self.pads = {f: [0.0] * P.pads for f in inst.stations}
        self.ugvs = [{"pos": inst.depot, "free": 0.0, "stock": P.ugv_stock, "km": 0.0}
                     for _ in range(P.n_ugv)]
        self.drones = []
        self.rollout = None      # None, "air" or "all" (see _rollout_choice)
        self.future = []         # departures not yet dispatched: [(time, route), ...]
        self.rollout_follow = True   # False: compute rollout costs but keep the base choice
        self.chooser = None      # learned policy: f(sim, k, i, j, t_done, E, options, base) -> index
        self.collector = None    # called with the rollout cost of every option
        self.pending_orders = None   # [(release, departure, window start, customer)]

    # ---- rollout -------------------------------------------------------------
    def clone(self):
        """Independent copy of the day's state; it continues with the base
        policy and without rollout."""
        c = DaySim.__new__(DaySim)
        c.inst, c.P, c.policy, c.record = self.inst, self.P, self.policy, False
        c.res = replace(self.res, mu={}, arr_energy={}, charged_after={}, decisions=[])
        c.events = list(self.events)
        c.seq, c.now = self.seq, self.now
        c.pads = None if self.pads is None else {f: list(v) for f, v in self.pads.items()}
        c.ugvs = [dict(u) for u in self.ugvs]
        c.drones = [dict(x) for x in self.drones]
        c.rollout, c.future = None, []
        c.rollout_follow, c.chooser, c.collector = True, None, None
        return c

    def _rollout_choice(self, k, i, j, t_done, E, options, base_choice):
        """Try every option, play the rest of the day with the base policy, and
        return the option with the lowest final cost.

        rollout == "air": only drones already dispatched are simulated.
        rollout == "all": departures scheduled for later are simulated too
                          (information a real controller would not fully have).
        Ties keep the base policy's choice."""
        future = self.future if self.rollout == "all" else []
        best, best_cost = base_choice, None
        costs = [0.0] * len(options)
        order = [base_choice] + [x for x in range(len(options)) if x != base_choice]
        for x in order:
            c = self.clone()
            c._apply(k, i, j, t_done, E, options[x])
            for t, r in future:
                c.advance(t - 1e-9)
                c.dispatch(r, t)
            c.advance()
            cost = costs[x] = c.result().cost
            if best_cost is None or cost < best_cost - 1e-9:
                best, best_cost = x, cost
        if self.collector is not None:      # training data: final cost of every option
            self.collector(self, k, i, j, t_done, E, options, base_choice, costs)
        return best if self.rollout_follow else base_choice

    # ---- helpers -----------------------------------------------------------
    def _push(self, t, kind, k, data):
        heapq.heappush(self.events, (t, self.seq, kind, k, data))
        self.seq += 1

    def _next_node(self, dr):
        i = dr["idx"] + 1
        return dr["route"][i] if i < len(dr["route"]) else self.inst.depot

    def _fly(self, k, frm, to, t, E):
        P, dr = self.P, self.drones[k]
        dist = self.inst.d[frm][to]
        dr["dist"] += dist
        E2 = E - P.fly_wh(dist, dr["load"])
        if E2 < -1e-6:
            self.res.energy_fail += 1
            E2 = 0.0
        self._push(t + P.fly_min(dist), "arrive", k, (to, E2))

    def _snapshot(self, k, i, t, E):
        """Fleet state handed to the policy (used by the learned estimate)."""
        dr = self.drones[k]
        return {
            "time": t, "drone": k, "node": i, "energy": E, "load": dr["load"],
            "remaining": len(dr["route"]) - dr["idx"] - 1,
            "pads_free_at": None if self.pads is None
                            else {f: list(v) for f, v in self.pads.items()},
            "ugvs": [dict(u) for u in self.ugvs],
            "in_air": sum(1 for x in self.drones if not x["done"]),
        }

    # ---- the charging decision ----------------------------------------------
    def _depart(self, k, i, t_done, t_req, E):
        """Drone k is ready to leave node i at t_done with E Wh on board."""
        inst, P, res = self.inst, self.P, self.res
        d, nodes, depot = inst.d, inst.nodes, inst.depot
        dr = self.drones[k]
        j = self._next_node(dr)
        load = dr["load"]
        e_ij = P.fly_wh(d[i][j], load)
        reserve = 0.0 if j == depot else P.fly_wh(inst.ref_d[j], load - nodes[j].demand)
        if E + EPS >= e_ij + reserve:
            self._fly(k, i, j, t_done, E)
            return

        options = []
        if i != depot:
            for ui, u in enumerate(self.ugvs):
                if u["stock"] <= 0:
                    continue
                d_u = d[u["pos"]][i]
                if u["km"] + P.km(d_u) > P.ugv_range_km + EPS:
                    continue
                if P.E + EPS < e_ij + reserve:
                    continue
                arr = max(u["free"], t_req) + P.ugv_min(d_u)
                leave = max(arr, t_done) + P.Ts
                options.append({"kind": "ugv", "u": ui, "leave": leave,
                                "wait": leave - t_done - P.Ts,
                                "t": leave - t_done,
                                "e": P.ugv_kwh_per_km * P.km(d_u),
                                "dist": P.km(d_u)})
        for f in inst.stations:
            if f == i:
                continue
            e_if = P.fly_wh(d[i][f], load)
            if E + EPS < e_if:
                continue
            e_fj = P.fly_wh(d[f][j], load)
            if P.E + EPS < e_fj + reserve:
                continue
            t_arr = t_done + P.fly_min(d[i][f])
            if self.pads is None:
                wait = P.exo_wait
            elif P.live_info:
                wait = max(0.0, min(self.pads[f]) - t_arr)
            else:
                wait = P.avg_wait_est
            ct = (P.E - (E - e_if)) / P.charge_rate
            options.append({"kind": "fcs", "f": f, "e_if": e_if, "wait": wait,
                            "t": P.fly_min(d[i][f]) + wait + ct
                                 + P.fly_min(d[f][j]) - P.fly_min(d[i][j]),
                            "e": e_if + e_fj - e_ij,
                            "dist": P.km(d[i][f] + d[f][j] - d[i][j])})
        if not options:
            res.energy_fail += 1          # stranded: penalise, then carry on
            self._fly(k, i, j, t_done, P.E)
            return
        if P.filter_late and j != depot:
            # drop options that would make the next customer late, if any other remains
            limit = nodes[j].l + P.tol - t_done - P.fly_min(d[i][j])
            ok = [o for o in options if o["t"] <= limit + EPS]
            if ok and len(ok) < len(options):
                res.filtered += len(options) - len(ok)
                options = ok

        need_state = self.record or getattr(self.policy, "needs_state", False)
        state = self._snapshot(k, i, t_done, E) if need_state else None
        base_choice = self.policy(options, state)
        choice = base_choice
        res.n_decisions += 1
        res.n_real_choices += len(options) > 1
        if self.chooser is not None and len(options) > 1:
            choice = self.chooser(self, k, i, j, t_done, E, options, base_choice)
            res.n_overrides += choice != base_choice
        elif self.rollout is not None and len(options) > 1:
            choice = self._rollout_choice(k, i, j, t_done, E, options, base_choice)
            res.n_overrides += choice != base_choice
        o = options[choice]
        if self.record:
            res.decisions.append({"time": t_done, "drone": k, "node": i,
                                  "chosen": o["kind"], "n_options": len(options),
                                  "options": [(x["kind"], round(x["t"], 3), round(x["e"], 3))
                                              for x in options]})
        self._apply(k, i, j, t_done, E, o)

    def _apply(self, k, i, j, t_done, E, o):
        """Carry out charging option o for drone k standing at node i."""
        P, res, d, dr = self.P, self.res, self.inst.d, self.drones[k]
        if o["kind"] == "ugv":
            u = self.ugvs[o["u"]]
            u["km"] += o["dist"]
            u["pos"] = i
            u["free"] = o["leave"]
            u["stock"] -= 1
            res.swaps += 1
            res.swap_wait += o["t"] - P.Ts
            res.charged_after[i] = ("ugv", o["t"])
            self._fly(k, i, j, o["leave"], P.E)
        else:
            f = o["f"]
            dr["dist"] += d[i][f]
            res.charged_after[i] = ("fcs", 0.0)
            self._push(t_done + P.fly_min(d[i][f]), "station", k, (f, E - o["e_if"], i))

    # ---- public interface -------------------------------------------------------
    def dispatch(self, route, t0=0.0):
        """A drone leaves the depot at t0 with a full battery to serve `route`."""
        if not route:
            return
        load = sum(self.inst.nodes[c].demand for c in route)
        if load > self.P.Q + EPS:
            self.res.overload += 1
        self.drones.append({"route": list(route), "idx": -1, "load": load,
                            "dist": 0.0, "done": False, "t0": t0,
                            "node": self.inst.depot, "E": self.P.E})
        self.res.n_routes += 1
        self._depart(len(self.drones) - 1, self.inst.depot, t0, t0, self.P.E)

    def advance(self, t_end=math.inf):
        """Process every event up to and including t_end."""
        inst, P, res = self.inst, self.P, self.res
        nodes, depot = inst.nodes, inst.depot
        while self.events and self.events[0][0] <= t_end:
            t, _, kind, k, data = heapq.heappop(self.events)
            self.now = t
            dr = self.drones[k]
            if kind == "arrive":
                node, E = data
                if node == depot:
                    dr["done"] = True
                    res.makespan = max(res.makespan, t)
                    continue
                dr["idx"] += 1
                dr["node"], dr["E"] = node, E      # last known position and charge
                n = nodes[node]
                start = max(t, n.e - P.tol)
                if start > n.l + P.tol + EPS:
                    res.late += 1
                mu = satisfaction(start, n.e, n.l, P.tol)
                res.mu[node] = mu
                res.arr_energy[node] = E
                res.sat_loss += 1.0 - mu
                dr["load"] -= n.demand
                self._depart(k, node, start + n.st, t, E)
            else:  # station
                f, E, came_from = data
                ct = (P.E - E) / P.charge_rate
                if self.pads is None:
                    wait = P.exo_wait
                else:
                    free = heapq.heappop(self.pads[f])
                    wait = max(0.0, free - t)
                    heapq.heappush(self.pads[f], t + wait + ct)
                res.station_visits += 1
                res.station_wait += wait
                res.charged_after[came_from] = ("fcs", wait)
                self._fly(k, f, self._next_node(dr), t + wait + ct, P.E)

    def result(self):
        P, res = self.P, self.res
        res.travel_km = P.km(sum(dr["dist"] for dr in self.drones))
        res.ugv_km = sum(u["km"] for u in self.ugvs)
        used = sum(1 for u in self.ugvs if u["km"] > 0)
        res.violations = res.late + res.energy_fail + res.overload
        res.cost = (P.Ct * res.travel_km + P.Cp * res.sat_loss + P.Cfixed * res.n_routes
                    + P.Cugv_km * res.ugv_km + P.Cugv_swap * res.swaps + P.Cugv_use * used
                    + P.infeasible_penalty * res.violations)
        return res


def simulate(inst, routes, P, policy, record=False, start=0.0):
    """All routes leave the depot at `start`; returns the day's SimResult."""
    sim = DaySim(inst, P, policy, record)
    for r in routes:
        sim.dispatch(r, start)
    sim.advance()
    return sim.result()
