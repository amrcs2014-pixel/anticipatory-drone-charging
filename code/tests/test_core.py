"""Core checks. Run from the code folder:  python tests/test_core.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from drp.instance import Instance, Node, load
from drp.params import Params
from drp.policies import (regret_scores, weighted_scores, regret_policy,
                          cost_policy, anticipatory_policy)
from drp.simulate import simulate, satisfaction
from drp.alns import ALNS

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "figshare_instances",
                    "extracted", "DATA", "DATA")


def approx(a, b, tol=1e-2):
    return abs(a - b) <= tol


def test_paper_regret_example():
    # Section 3.2: A, B, C with t = 3, 5, 10 and e = 10, 7, 1; weights 0.4 / 0.6.
    opts = [{"t": 3, "e": 10}, {"t": 5, "e": 7}, {"t": 10, "e": 1}]
    ws = weighted_scores(opts, 0.4, 0.6)
    rs = regret_scores(opts, 0.4, 0.6)
    assert approx(ws[0], 0.60) and approx(ws[1], 0.51) and approx(ws[2], 0.40), ws
    assert approx(rs[0], 0.80) and approx(rs[1], 0.51) and approx(rs[2], 0.69), rs
    assert cost_policy(0.4, 0.6)(opts, None) == 2      # paper: C preferred
    assert regret_policy(0.4, 0.6)(opts, None) == 1    # paper: B preferred


def test_anticipatory_nests_regret():
    opts = [{"t": 3, "e": 10}, {"t": 5, "e": 7}, {"t": 10, "e": 1}]
    zero = anticipatory_policy(lambda o, s: 0.0, 0.4, 0.6, base="regret")
    assert zero(opts, None) == regret_policy(0.4, 0.6)(opts, None)
    zero_c = anticipatory_policy(lambda o, s: 0.0, 0.4, 0.6, base="cost")
    assert zero_c(opts, None) == cost_policy(0.4, 0.6)(opts, None)
    # a large future cost on option B must move the choice away from it
    pushed = anticipatory_policy(lambda o, s: 5.0 if o["t"] == 5 else 0.0, 0.4, 0.6)
    assert pushed(opts, None) != 1


def test_satisfaction():
    assert satisfaction(50, 40, 60, 30) == 1.0
    assert approx(satisfaction(25, 40, 60, 30), 0.5)
    assert approx(satisfaction(75, 40, 60, 30), 0.5)
    assert satisfaction(5, 40, 60, 30) == 0.0
    assert satisfaction(95, 40, 60, 30) == 0.0


def tiny():
    nodes = [Node("0", "d", 0, 0, 0, 1000, 0, 0),
             Node("1", "c", 3, 4, 0, 500, 5, 1.0),      # 5 units from the depot
             Node("2", "c", 6, 8, 0, 500, 5, 1.0),      # 10 units from the depot
             Node("3", "r", 6, 4, 0, 1000, 0, 0)]
    return Instance(nodes, "tiny")


def test_single_route_by_hand():
    inst, P = tiny(), Params(dist_scale=1.0)
    res = simulate(inst, [[1, 2]], P, regret_policy())
    assert res.feasible, res
    assert approx(res.travel_km, 5 + 5 + 10)
    assert approx(res.cost, 20 + 100)                 # travel + one drone, full satisfaction
    # arrival energy at customer 1: 5 km with 2 kg on board
    e1 = 100 - 5 / 79.2 * (43.68 * 7 + 113.02)
    assert approx(res.arr_energy[1], e1), (res.arr_energy[1], e1)


def test_overload_and_lateness_are_penalised():
    inst = tiny()
    assert simulate(inst, [[1, 2]], Params(Q=1.5), regret_policy()).overload == 1
    late = Instance([Node("0", "d", 0, 0, 0, 1000, 0, 0),
                     Node("1", "c", 30, 40, 0, 1, 5, 1.0),
                     Node("2", "r", 5, 5, 0, 1000, 0, 0)], "late")
    res = simulate(late, [[1]], Params(dist_scale=0.2, tol=1.0), regret_policy())
    assert res.late == 1 and not res.feasible


def test_charging_is_triggered_and_shared_pads_queue():
    # stretch distances so that the route needs a recharge
    inst = tiny()
    P = Params(dist_scale=1.6, n_ugv=0)
    res = simulate(inst, [[1, 2]], P, regret_policy(), record=True)
    assert res.station_visits >= 1, res
    assert res.feasible, res
    # two identical drones, one pad, slow charger: the second one must wait
    P2 = Params(dist_scale=1.6, n_ugv=0, pads=1, charge_rate=2.0)
    inst2 = Instance(inst.nodes + [Node("4", "c", 3, 4, 0, 500, 5, 1.0),
                                   Node("5", "c", 6, 8, 0, 500, 5, 1.0)], "tiny2")
    c = inst2.customers
    res2 = simulate(inst2, [[c[0], c[1]], [c[2], c[3]]], P2, regret_policy())
    assert res2.station_visits >= 2 and res2.station_wait > 0, res2


def test_ugv_swap_is_available():
    inst = tiny()
    P = Params(dist_scale=1.6, n_ugv=1)
    with_ugv = simulate(inst, [[1, 2]], P, regret_policy(), record=True)
    assert with_ugv.feasible
    kinds = {k for dec in with_ugv.decisions for (k, _, _) in dec["options"]}
    assert "ugv" in kinds, with_ugv.decisions


def test_alns_serves_everyone_and_improves():
    path = os.path.join(DATA, "C", "C75_R5.txt")
    if not os.path.exists(path):
        print("  (skipped: Figshare data not found)")
        return
    inst = load(path).subset(20)
    P = Params(dist_scale=0.2)
    alns = ALNS(inst, P, regret_policy(), seed=1, iters=40)
    init = alns.construct()
    init_cost = alns.evaluate(init).cost
    best, res = alns.run(initial=init)
    served = sorted(c for r in best for c in r)
    assert served == inst.customers, "every customer exactly once"
    assert res.cost <= init_cost + 1e-9
    assert res.feasible, res


def test_reproduces_two_gurobi_optima_from_the_paper():
    # Table 6 of the base paper: R11_R3 = 329.76 (proven optimal), R12_R4 = 335.05.
    # Both are reproduced to the cent by taking the first n customers of the
    # 75-customer file, at 0.1 km per coordinate unit.
    for fname, n, target in [("R75_R3", 11, 329.76), ("R75_R4", 12, 335.05)]:
        path = os.path.join(DATA, "R", fname + ".txt")
        if not os.path.exists(path):
            print("  (skipped: Figshare data not found)")
            return
        inst = load(path).subset(n)
        best = min(ALNS(inst, Params(), regret_policy(), seed=s, iters=150).run()[1].cost
                   for s in range(4))
        assert approx(best, target, 0.01), (fname, n, best, target)


def test_dynamic_day_serves_everyone_after_release():
    from drp.dynamic import release_times, run_day, effective_dod
    path = os.path.join(DATA, "C", "C75_R5.txt")
    if not os.path.exists(path):
        print("  (skipped: Figshare data not found)")
        return
    inst = load(path).subset(25)
    P = Params(dist_scale=0.25, pads=1, n_ugv=1)
    assert effective_dod(release_times(inst, P, 0.0)) == 0.0
    rel = release_times(inst, P, 0.6, seed=3)
    assert 0.0 < effective_dod(rel) <= 0.6
    res, info = run_day(inst, P, regret_policy(), rel, iters=10, seed=3)
    assert sorted(res.mu) == inst.customers, "every customer served exactly once"
    served = sorted(c for _, r in info["departures"] for c in r)
    assert served == inst.customers
    for t, r in info["departures"]:
        assert all(rel[c] <= t + 1e-9 for c in r), "no drone leaves with an unreleased order"
    assert res.overload == 0


def test_rules_are_compared_on_the_same_schedule():
    from drp.dynamic import release_times, plan_day, execute
    from drp.policies import nearest_policy
    path = os.path.join(DATA, "C", "C75_R5.txt")
    if not os.path.exists(path):
        print("  (skipped: Figshare data not found)")
        return
    inst = load(path).subset(25)
    P = Params(dist_scale=0.25, pads=1, n_ugv=1, infeasible_penalty=200.0)
    rel = release_times(inst, P, 0.5, seed=1)
    sched, _ = plan_day(inst, P, regret_policy(), rel, iters=10, seed=1)
    a = execute(inst, P, regret_policy(), sched)
    b = execute(inst, P, nearest_policy(), sched)
    again = execute(inst, P, regret_policy(), sched)
    assert a.n_routes == b.n_routes == len(sched)
    assert sorted(a.mu) == sorted(b.mu) == inst.customers
    assert a.cost == again.cost, "execution is deterministic for a given schedule and rule"


def test_rollout_with_full_schedule_never_loses_to_its_base_rule():
    from drp.dynamic import release_times, plan_day, execute
    path = os.path.join(DATA, "C", "C75_R5.txt")
    if not os.path.exists(path):
        print("  (skipped: Figshare data not found)")
        return
    inst = load(path).subset(30)
    P = Params(dist_scale=0.25, pads=1, n_ugv=1, infeasible_penalty=200.0)
    for seed in (0, 1, 2):
        rel = release_times(inst, P, 0.5, seed=seed)
        sched, _ = plan_day(inst, P, regret_policy(), rel, iters=8, seed=seed)
        base = execute(inst, P, regret_policy(), sched)
        roll = execute(inst, P, regret_policy(), sched, rollout="all")
        assert sorted(roll.mu) == inst.customers
        assert roll.cost <= base.cost + 1e-6, (seed, roll.cost, base.cost)
        # the clone used inside rollout must not disturb the real day
        assert execute(inst, P, regret_policy(), sched).cost == base.cost


def test_learned_chooser_with_zero_estimate_is_the_regret_rule():
    from drp.dynamic import release_times, plan_day, execute
    from drp.learn import FutureCostModel, Dataset, features, FEATURES
    path = os.path.join(DATA, "C", "C75_R5.txt")
    if not os.path.exists(path):
        print("  (skipped: Figshare data not found)")
        return
    inst = load(path).subset(30)
    P = Params(dist_scale=0.25, pads=1, n_ugv=1, infeasible_penalty=200.0)
    rel = release_times(inst, P, 0.5, seed=0)
    sched, _ = plan_day(inst, P, regret_policy(), rel, iters=8, seed=0)
    base = execute(inst, P, regret_policy(), sched)

    class Zero(FutureCostModel):
        def __init__(self):
            self.margin = 0.0

        def predict(self, X):
            assert all(len(r) == len(FEATURES) for r in X)
            return [0.0] * len(X)

    same = execute(inst, P, regret_policy(), sched, chooser=Zero().chooser())
    assert same.cost == base.cost and same.n_overrides == 0
    # labels: one row per alternative, and the regret day itself is not disturbed
    data = Dataset()
    kept = execute(inst, P, regret_policy(), sched, rollout="all",
                   collector=data.collector, follow=False)
    assert kept.cost == base.cost
    assert len(data) > 0 and all(len(r) == len(FEATURES) for r in data.X)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as exc:                      # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
