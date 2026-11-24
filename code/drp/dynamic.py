"""Rolling-horizon operation of one day with orders revealed over time.

Parcels are loaded at the depot, so a drone that has left cannot take new
orders.  The dynamic decisions are therefore:
  1. which known orders go together on a route (ALNS, re-run when orders arrive);
  2. when each route leaves the depot (hold while waiting costs nothing);
  3. where each drone recharges on the way (the charging policy, in the
     shared day simulation where queues build up from the fleet's own choices).

Decisions 1 and 2 form the *schedule* and are made with a fixed planning
policy.  Decision 3 is then executed under any charging policy on the same
schedule, so that charging rules are compared on identical days.
"""
import random

from .alns import ALNS
from .simulate import DaySim, simulate


def release_times(inst, P, dod, seed=0, epoch=10.0):
    """Release time per customer.  A share `dod` of the customers is revealed
    during the day, each at a random time that still leaves one epoch plus the
    flight from the depot before its tolerable window opens."""
    rng = random.Random(seed)
    rel = {c: 0.0 for c in inst.customers}
    for c in rng.sample(inst.customers, int(round(dod * len(inst.customers)))):
        n = inst.nodes[c]
        latest = n.e - P.tol - P.fly_min(inst.d[inst.depot][c]) - epoch
        if latest > 0:
            rel[c] = rng.uniform(0.0, latest)
    return rel


def effective_dod(rel):
    return sum(1 for t in rel.values() if t > 0) / max(1, len(rel))


def plan_day(inst, P, plan_policy, rel, epoch=10.0, iters=30, seed=0, first_iters=None):
    """Decide routes and departure times.  Returns (schedule, info) where
    schedule is a list of (departure time, route), in order of departure.

    The routes still at the depot are re-optimised at every epoch, with or
    without new orders.  `first_iters` gives the very first plan a larger
    budget (used by the perfect-information benchmark)."""
    horizon = max(inst.nodes[c].l for c in inst.customers) + P.tol
    unseen = sorted(inst.customers, key=lambda c: rel[c])
    plan, schedule = [], []
    t, epochs, replans = 0.0, 0, 0
    while True:
        new = []
        while unseen and rel[unseen[0]] <= t:
            new.append(unseen.pop(0))
        if new or plan:
            pool = [c for r in plan for c in r] + new
            n_it = first_iters if (epochs == 0 and first_iters) else iters
            alns = ALNS(inst, P, plan_policy, seed=seed + epochs, iters=n_it,
                        customers=pool, start=t, flex_start=True)
            for c in new:           # warm start: keep the old plan, insert the new orders
                _, ri, pos = alns.best_insertion(plan, c)
                alns.insert(plan, c, ri, pos)
            plan, _ = alns.run(initial=plan)
            replans += 1
        keep = []
        for r in plan:
            now = simulate(inst, [r], P, plan_policy, start=t).cost
            later = simulate(inst, [r], P, plan_policy, start=t + epoch).cost
            if later > now + 1e-9 or t >= horizon:
                schedule.append((t, list(r)))
            else:
                keep.append(r)
        plan = keep
        t += epoch
        epochs += 1
        if not unseen and not plan:
            break
    info = {"epochs": epochs, "replans": replans, "dod": effective_dod(rel)}
    return schedule, info


def execute(inst, P, policy, schedule, record=False, rollout=None, chooser=None,
            collector=None, follow=True, rel=None):
    """Fly a schedule under a charging policy; returns the day's SimResult.

    rollout=None  : the policy decides alone.
    rollout="air" : at each charging decision every option is simulated to the
                    end of the day (drones already in the air only) with
                    `policy` deciding afterwards; the cheapest option is taken.
    rollout="all" : the same, but departures scheduled for later are simulated
                    too.  This uses knowledge of the future schedule.
    chooser       : a learned decision function; it overrides rollout.
    collector     : receives the rollout cost of every option (training data).
    follow=False  : compute rollout costs but keep the policy's own choice."""
    sim = DaySim(inst, P, policy, record)
    sim.rollout = rollout
    sim.chooser, sim.collector, sim.rollout_follow = chooser, collector, follow
    if rel is not None:      # what a controller knows about orders not yet dispatched
        sim.pending_orders = [(rel[c], t, inst.nodes[c].e, c)
                              for t, r in schedule for c in r]
    sched = sorted(schedule, key=lambda x: x[0])
    for n, (t, r) in enumerate(sched):
        sim.future = sched[n:]
        sim.advance(t - 1e-9)        # everything strictly before this departure
        sim.future = sched[n + 1:]
        sim.dispatch(r, t)
    sim.future = []
    sim.advance()
    return sim.result()


def run_day(inst, P, policy, rel, epoch=10.0, iters=30, seed=0, record=False,
            plan_policy=None):
    """Plan and execute one day.  Returns (SimResult, info)."""
    schedule, info = plan_day(inst, P, plan_policy or policy, rel, epoch, iters, seed)
    info["departures"] = schedule
    return execute(inst, P, policy, schedule, record), info
