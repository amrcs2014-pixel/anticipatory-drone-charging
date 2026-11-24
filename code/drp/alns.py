"""Adaptive large neighborhood search used as the routing engine.

Operator selection uses the score-based adaptive weights of Ropke and
Pisinger, as in the base paper (Eq. 45, scores 33/13/9).  Nothing here is
learned.  Insertion positions are scored with a single-route simulation
(cheap, cached); every candidate solution is then scored with the full
multi-drone simulation, which is the objective.
"""
import math
import random
import time

from .simulate import simulate, DaySim


class ALNS:
    def __init__(self, inst, P, policy, seed=0, iters=300, rmin=0.1, rmax=0.4,
                 sigma=(33.0, 13.0, 9.0), rho=0.1, cooling=0.99, T0=None,
                 start_accept=0.05, time_limit=None, verbose=False,
                 destroy=None, repair=None, customers=None, start=0.0, flex_start=False):
        self.inst, self.P, self.policy = inst, P, policy
        self.rng = random.Random(seed)
        self.iters, self.rmin, self.rmax = iters, rmin, rmax
        self.sigma, self.rho, self.cooling = sigma, rho, cooling
        self.T0, self.start_accept = T0, start_accept
        self.time_limit, self.verbose = time_limit, verbose
        self._cache = {}
        self.customers = list(customers) if customers is not None else list(inst.customers)
        self.start = start          # earliest time a planned route may leave the depot
        self.flex_start = flex_start  # let each route leave later if that is cheaper
        self._start_of = {}
        self.destroy_ops = {
            "random": self.d_random, "worst": self.d_worst, "route": self.d_route,
            "shaw": self.d_shaw, "energy": self.d_energy, "queue": self.d_queue,
        }
        self.repair_ops = {
            "greedy": self.r_greedy, "regret2": self.r_regret2, "random": self.r_random,
        }
        # optional operator subsets, e.g. to mimic the base paper's 3 + 2 operators
        all_destroy = dict(self.destroy_ops, keep_longest=self.d_keep_longest)
        if destroy is not None:
            self.destroy_ops = {n: all_destroy[n] for n in destroy}
        if repair is not None:
            self.repair_ops = {n: self.repair_ops[n] for n in repair}
        self.history = []

    # ---- evaluation -------------------------------------------------------
    def route_cost(self, route):
        key = tuple(route)
        c = self._cache.get(key)
        if c is None:
            if not route:
                c = 0.0
            else:
                c = simulate(self.inst, [route], self.P, self.policy, start=self.start).cost
                self._start_of[key] = self.start
                if self.flex_start:
                    # candidate: leave so that the first customer is reached at the
                    # start of its preferred window
                    first = self.inst.nodes[route[0]]
                    t2 = first.e - self.P.fly_min(self.inst.d[self.inst.depot][route[0]])
                    if t2 > self.start + 1e-9:
                        c2 = simulate(self.inst, [route], self.P, self.policy, start=t2).cost
                        if c2 < c:
                            c, self._start_of[key] = c2, t2
            self._cache[key] = c
        return c

    def start_time(self, route):
        self.route_cost(route)
        return self._start_of[tuple(route)]

    def evaluate(self, routes):
        if not self.flex_start:
            return simulate(self.inst, routes, self.P, self.policy, start=self.start)
        sim = DaySim(self.inst, self.P, self.policy)
        for r in sorted((r for r in routes if r), key=self.start_time):
            sim.dispatch(r, self.start_time(r))
        sim.advance()
        return sim.result()

    # ---- construction -------------------------------------------------------
    def best_insertion(self, routes, c):
        """(delta, route index or -1 for a new route, position)."""
        best = (self.route_cost([c]), -1, 0)
        for ri, r in enumerate(routes):
            base = self.route_cost(r)
            for pos in range(len(r) + 1):
                delta = self.route_cost(r[:pos] + [c] + r[pos:]) - base
                if delta < best[0]:
                    best = (delta, ri, pos)
        return best

    def insert(self, routes, c, ri, pos):
        if ri < 0:
            routes.append([c])
        else:
            routes[ri].insert(pos, c)

    def construct(self):
        routes = []
        order = sorted(self.customers, key=lambda c: self.inst.nodes[c].e)
        for c in order:
            _, ri, pos = self.best_insertion(routes, c)
            self.insert(routes, c, ri, pos)
        return routes

    # ---- destroy operators ----------------------------------------------------
    def _q(self):
        n = len(self.customers)
        return min(n, max(1, int(round(n * self.rng.uniform(self.rmin, self.rmax)))))

    def _strip(self, routes, removed):
        rs = set(removed)
        return [[c for c in r if c not in rs] for r in routes if any(c not in rs for c in r)]

    def d_random(self, routes, res):
        removed = self.rng.sample(self.customers, self._q())
        return self._strip(routes, removed), removed

    def d_worst(self, routes, res):
        gains = []
        for r in routes:
            base = self.route_cost(r)
            for c in r:
                gains.append((base - self.route_cost([x for x in r if x != c]), c))
        gains.sort(reverse=True)
        removed = []
        q = self._q()
        while gains and len(removed) < q:        # randomised: bias towards the worst
            idx = int(len(gains) * self.rng.random() ** 3)
            removed.append(gains.pop(idx)[1])
        return self._strip(routes, removed), removed

    def d_route(self, routes, res):
        q = self._q()
        removed = []
        for r in sorted(routes, key=len):
            if removed and len(removed) + len(r) > q:
                break
            removed.extend(r)
        return self._strip(routes, removed), removed

    def d_keep_longest(self, routes, res):
        """Base paper's 'shorter path removal': keep only the longest route."""
        if len(routes) < 2:
            return self.d_random(routes, res)
        keep = max(range(len(routes)), key=lambda i: len(routes[i]))
        removed = [c for i, r in enumerate(routes) if i != keep for c in r]
        return [list(routes[keep])], removed

    def d_shaw(self, routes, res):
        inst = self.inst
        seed = self.rng.choice(self.customers)
        dmax = max(max(row) for row in inst.d) or 1.0
        tmax = max(n.l for n in inst.nodes) or 1.0
        rel = sorted(self.customers, key=lambda c: inst.d[seed][c] / dmax
                     + abs(inst.nodes[seed].e - inst.nodes[c].e) / tmax)
        removed = rel[:self._q()]
        return self._strip(routes, removed), removed

    def d_energy(self, routes, res):
        """Customers reached with the least energy left."""
        if not res.arr_energy:
            return self.d_random(routes, res)
        order = sorted(res.arr_energy, key=res.arr_energy.get)
        removed = order[:self._q()]
        return self._strip(routes, removed), removed

    def d_queue(self, routes, res):
        """Customers whose drones charged next, longest waits first, plus neighbours."""
        if not res.charged_after:
            return self.d_random(routes, res)
        q = self._q()
        hot = sorted(res.charged_after, key=lambda c: -res.charged_after[c][1])
        removed = []
        for c in hot:
            for r in routes:
                if c in r:
                    i = r.index(c)
                    for x in r[max(0, i - 1): i + 2]:
                        if x not in removed and len(removed) < q:
                            removed.append(x)
        if len(removed) < q:
            rest = [c for c in self.customers if c not in removed]
            removed += self.rng.sample(rest, min(len(rest), q - len(removed)))
        return self._strip(routes, removed), removed

    # ---- repair operators --------------------------------------------------------
    def r_greedy(self, routes, removed):
        self.rng.shuffle(removed)
        for c in removed:
            _, ri, pos = self.best_insertion(routes, c)
            self.insert(routes, c, ri, pos)
        return routes

    def r_regret2(self, routes, removed):
        pending = list(removed)
        while pending:
            pick = None
            for c in pending:
                per_route = [(self.route_cost([c]), -1, 0)]
                for ri, r in enumerate(routes):
                    base = self.route_cost(r)
                    b = min((self.route_cost(r[:p] + [c] + r[p:]) - base, ri, p)
                            for p in range(len(r) + 1))
                    per_route.append(b)
                per_route.sort()
                regret = (per_route[1][0] - per_route[0][0]) if len(per_route) > 1 else 0.0
                if pick is None or regret > pick[0]:
                    pick = (regret, c, per_route[0])
            _, c, (_, ri, pos) = pick
            self.insert(routes, c, ri, pos)
            pending.remove(c)
        return routes

    def r_random(self, routes, removed):
        """Random position; falls back to a new route if that position is infeasible."""
        pen = self.P.infeasible_penalty
        self.rng.shuffle(removed)
        for c in removed:
            placed = False
            for _ in range(5):
                if not routes:
                    break
                ri = self.rng.randrange(len(routes))
                pos = self.rng.randint(0, len(routes[ri]))
                r = routes[ri]
                if self.route_cost(r[:pos] + [c] + r[pos:]) - self.route_cost(r) < pen:
                    r.insert(pos, c)
                    placed = True
                    break
            if not placed:
                routes.append([c])
        return routes

    # ---- main loop -------------------------------------------------------------------
    def _pick(self, weights):
        names = list(weights)
        return self.rng.choices(names, [weights[n] for n in names])[0]

    def run(self, initial=None):
        t0 = time.time()
        cur = [list(r) for r in (initial or self.construct())]
        cur_res = self.evaluate(cur)
        best, best_res = [list(r) for r in cur], cur_res
        T = self.T0 if self.T0 is not None else \
            self.start_accept * cur_res.cost / math.log(2.0)
        wd = {n: 1.0 for n in self.destroy_ops}
        wr = {n: 1.0 for n in self.repair_ops}
        self.history = [(0, cur_res.cost, best_res.cost)]
        for it in range(1, self.iters + 1):
            if self.time_limit and time.time() - t0 > self.time_limit:
                break
            dn, rn = self._pick(wd), self._pick(wr)
            partial, removed = self.destroy_ops[dn]([list(r) for r in cur], cur_res)
            cand = [r for r in self.repair_ops[rn](partial, list(removed)) if r]
            cand_res = self.evaluate(cand)
            score = 0.0
            delta = cand_res.cost - cur_res.cost
            if delta < 0 or self.rng.random() < math.exp(-delta / max(T, 1e-9)):
                if cand_res.cost < best_res.cost - 1e-9:
                    best, best_res = [list(r) for r in cand], cand_res
                    score = self.sigma[0]
                elif delta < 0:
                    score = self.sigma[1]
                else:
                    score = self.sigma[2]
                cur, cur_res = cand, cand_res
            wd[dn] = (1 - self.rho) * wd[dn] + self.rho * score
            wr[rn] = (1 - self.rho) * wr[rn] + self.rho * score
            wd[dn], wr[rn] = max(wd[dn], 0.05), max(wr[rn], 0.05)
            T *= self.cooling
            self.history.append((it, cur_res.cost, best_res.cost))
            if self.verbose and it % 25 == 0:
                print(f"  it {it:4d}  current {cur_res.cost:9.2f}  best {best_res.cost:9.2f}"
                      f"  routes {best_res.n_routes}  {time.time() - t0:6.1f}s")
        self.weights = {"destroy": wd, "repair": wr}
        self.seconds = time.time() - t0
        return best, best_res
