"""Learned estimate of the future cost of a charging option.

Training signal.  For a training day, rollout with knowledge of later
departures gives, for every charging option, the final cost of the day if
that option is taken and the regret rule decides afterwards.  The label of an
option is that cost minus the cost of the regret rule's own choice: its
*advantage* (negative = better than the regret rule).

Deployment.  At a decision the model sees only what a real controller has
(the option, the drone, the fleet in the air, the chargers, the time of day),
predicts the advantage of every alternative to the regret choice, and switches
to the best alternative only if the predicted gain exceeds a margin.  With an
estimate of zero it therefore reproduces the regret rule exactly.
"""
import math

from .policies import regret_scores, weighted_scores

FEATURES = [
    "is_ugv", "time_cost", "wait", "detour_km", "time_minus_best", "time_minus_base",
    "detour_minus_base",
    "regret_score", "weighted_score", "slack_after", "n_options", "n_ugv_options",
    "n_fcs_options", "drone_energy", "drone_remaining", "drone_load", "time_of_day",
    "in_air", "in_air_low", "remaining_all", "recent_departures", "charger_busy",
    "low_near_new_ugv_pos", "low_near_old_ugv_pos", "low_nearest_to_station",
    "base_is_ugv", "base_time_cost", "base_wait",
    # forecast features: orders already known but not yet dispatched (zero when the
    # release times are not supplied to execute())
    "pending_all", "pending_30", "pending_60", "pending_60_near_option",
]
N_BASIC = 28       # features before the forecast block (older models use only these)
LOW = 0.6          # share of battery below which a drone counts as "low"
NEAR_KM = 6.0      # radius for "near"
SLACK_CAP = 120.0


def _horizon(inst, P):
    h = getattr(inst, "_horizon", None)
    if h is None:
        h = inst._horizon = max(inst.nodes[c].l for c in inst.customers) + P.tol
    return h


def features(sim, k, i, j, t_done, E, options, base):
    """One feature vector per option, from information available at decision time."""
    inst, P = sim.inst, sim.P
    d = inst.d
    reg = regret_scores(options, P.xi1, P.xi2)
    wsum = weighted_scores(options, P.xi1, P.xi2)
    tmin = min(o["t"] for o in options)
    ob = options[base]
    n_ugv = sum(o["kind"] == "ugv" for o in options)
    dr = sim.drones[k]
    others = [x for n, x in enumerate(sim.drones) if n != k and not x["done"]]
    low = [x for x in others if x["E"] < LOW * P.E and x["idx"] + 1 < len(x["route"])]
    slack0 = SLACK_CAP if j == inst.depot else \
        inst.nodes[j].l + P.tol - t_done - P.fly_min(d[i][j])
    recent = sum(1 for x in sim.drones if t_done - x["t0"] <= 20.0)
    common = [
        E, len(dr["route"]) - dr["idx"] - 1, dr["load"], t_done / _horizon(inst, P),
        len(others), len(low),
        sum(len(x["route"]) - x["idx"] - 1 for x in others), recent,
    ]
    # orders the controller already knows about but whose drone has not left yet
    pend = [(e, c) for (rel, dep, e, c) in (getattr(sim, "pending_orders", None) or ())
            if rel <= t_done < dep]
    p_all = len(pend)
    p30 = sum(1 for e, c in pend if e - t_done <= 30.0)
    p60 = [c for e, c in pend if e - t_done <= 60.0]
    rows = []
    for n, o in enumerate(options):
        is_ugv = o["kind"] == "ugv"
        near_new = near_old = nearest = 0
        spot = i if is_ugv else o["f"]
        p_near = sum(1 for c in p60 if P.km(d[c][spot]) <= NEAR_KM)
        if is_ugv:
            u = sim.ugvs[o["u"]]
            busy = max(0.0, u["free"] - t_done)
            near_new = sum(1 for x in low if P.km(d[x["node"]][i]) <= NEAR_KM)
            near_old = sum(1 for x in low if P.km(d[x["node"]][u["pos"]]) <= NEAR_KM)
        else:
            f = o["f"]
            busy = 0.0 if sim.pads is None else max(0.0, min(sim.pads[f]) - t_done)
            nearest = sum(1 for x in low
                          if min(inst.stations, key=lambda s: d[x["node"]][s]) == f)
        rows.append([
            float(is_ugv), o["t"], o["wait"], o["dist"], o["t"] - tmin, o["t"] - ob["t"],
            o["dist"] - ob["dist"], reg[n], wsum[n], max(-SLACK_CAP, min(SLACK_CAP, slack0 - o["t"])),
            len(options), n_ugv, len(options) - n_ugv,
            *common, busy, near_new, near_old, nearest,
            float(ob["kind"] == "ugv"), ob["t"], ob["wait"],
            p_all, p30, len(p60), p_near,
        ])
    return rows


class Dataset:
    """Rows = non-base options; label = rollout cost minus the base option's cost."""

    def __init__(self):
        self.X, self.y, self.day, self.dec = [], [], [], []
        self._day = 0
        self._dec = 0

    def start_day(self, day):
        self._day = day

    def collector(self, sim, k, i, j, t_done, E, options, base, costs):
        rows = features(sim, k, i, j, t_done, E, options, base)
        self._dec += 1
        for n, r in enumerate(rows):
            if n == base:
                continue
            self.X.append(r)
            self.y.append(costs[n] - costs[base])
            self.day.append(self._day)
            self.dec.append(self._dec)

    def __len__(self):
        return len(self.y)


class FutureCostModel:
    """Gradient-boosted regression of the advantage, plus a switching margin."""

    def __init__(self, margin=0.0, **kw):
        import lightgbm as lgb
        params = dict(objective="huber", alpha=50.0, n_estimators=300, learning_rate=0.03,
                      num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=0)
        params.update(kw)
        self.model = lgb.LGBMRegressor(**params)
        self.margin = margin

    def fit(self, X, y, forecast=True):
        import numpy as np
        self.nf = len(FEATURES) if forecast else N_BASIC
        self.model.fit(np.asarray(X, dtype=float)[:, :self.nf], np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        import numpy as np
        nf = getattr(self, "nf", N_BASIC)       # models saved before the forecast block
        return self.model.predict(np.asarray(X, dtype=float)[:, :nf])

    def chooser(self):
        """Decision function for DaySim.chooser."""
        def choose(sim, k, i, j, t_done, E, options, base):
            rows = features(sim, k, i, j, t_done, E, options, base)
            alt = [n for n in range(len(options)) if n != base]
            pred = self.predict([rows[n] for n in alt])
            b = min(range(len(alt)), key=lambda x: pred[x])
            return alt[b] if pred[b] < -self.margin else base
        return choose

    def labelled_chooser(self):
        """As chooser(), but also triggers rollout so that the day yields labels
        along the learned policy's own trajectory (needs sim.rollout == 'all')."""
        inner = self.chooser()

        def choose(sim, k, i, j, t_done, E, options, base):
            sim._rollout_choice(k, i, j, t_done, E, options, base)
            return inner(sim, k, i, j, t_done, E, options, base)
        return choose

    def importance(self):
        gain = self.model.booster_.feature_importance(importance_type="gain")
        tot = float(sum(gain)) or 1.0
        names = FEATURES[:getattr(self, "nf", N_BASIC)]
        return sorted(((g / tot, n) for g, n in zip(gain, names)), reverse=True)


def t_crit(df):
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
             8: 2.306, 9: 2.262, 10: 2.228, 14: 2.145, 19: 2.093, 24: 2.064,
             29: 2.045, 39: 2.023, 49: 2.010, 59: 2.001, 99: 1.984, 149: 1.976}
    return table.get(df) or table[max(k for k in table if k <= df)]


def ci95(xs):
    import statistics as st
    n = len(xs)
    if n < 2:
        return (xs[0] if xs else 0.0), float("nan")
    return st.mean(xs), t_crit(n - 1) * st.stdev(xs) / math.sqrt(n)
