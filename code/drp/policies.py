"""Charging decision rules.

A policy is a callable  policy(options, state) -> index of the chosen option.

Each option is a dict with at least:
    kind  'fcs' or 'ugv'
    t     time cost in minutes   (paper Eq. 2-5: detour + queue, or UGV travel + wait)
    e     energy cost            (paper Eq. 6-9)
    dist  km of detour (station) or of UGV travel
`state` is a snapshot of the fleet at the moment of the decision; the myopic
rules ignore it, the anticipatory policy passes it to a learned estimate.
"""

DELTA = 1e-6


def _ranges(options):
    ts = [o["t"] for o in options]
    es = [o["e"] for o in options]
    return min(ts), max(ts), min(es), max(es)


def regret_scores(options, xi1=0.5, xi2=0.5):
    """Total regret of each option, paper Eq. 1."""
    tmin, tmax, emin, emax = _ranges(options)
    out = []
    for o in options:
        r = 0.0
        for o2 in options:
            if o2 is o:
                continue
            r += xi1 * max(0.0, o["t"] - o2["t"]) / (tmax - tmin + DELTA)
            r += xi2 * max(0.0, o["e"] - o2["e"]) / (emax - emin + DELTA)
        out.append(r)
    return out


def weighted_scores(options, xi1=0.5, xi2=0.5):
    """Weighted-sum cost of each option, paper Eq. 11."""
    tmin, tmax, emin, emax = _ranges(options)
    return [xi1 * (o["t"] - tmin) / (tmax - tmin + DELTA)
            + xi2 * (o["e"] - emin) / (emax - emin + DELTA) for o in options]


def _argmin(scores):
    return min(range(len(scores)), key=scores.__getitem__)


def regret_policy(xi1=0.5, xi2=0.5):
    """The base paper's rule (Eq. 10)."""
    return lambda options, state: _argmin(regret_scores(options, xi1, xi2))


def cost_policy(xi1=0.5, xi2=0.5):
    """The cost-based comparison rule (Eq. 12)."""
    return lambda options, state: _argmin(weighted_scores(options, xi1, xi2))


def nearest_policy():
    """Shortest detour / shortest UGV trip."""
    return lambda options, state: _argmin([o["dist"] for o in options])


def fcs_only(policy):
    """Wrap a policy so that it never uses a UGV when a station is available."""
    def wrapped(options, state):
        idx = [i for i, o in enumerate(options) if o["kind"] == "fcs"]
        if not idx:
            return policy(options, state)
        return idx[policy([options[i] for i in idx], state)]
    return wrapped


def anticipatory_policy(future_cost, xi1=0.5, xi2=0.5, base="regret"):
    """Immediate score plus a learned estimate of downstream cost.

    future_cost(option, state) -> float, on the same scale as the immediate
    score.  With future_cost == 0 this reduces exactly to the regret rule
    (base='regret') or to the cost rule (base='cost').
    The estimate itself is trained in a later phase; this is the hook.
    """
    scorer = regret_scores if base == "regret" else weighted_scores

    def policy(options, state):
        now = scorer(options, xi1, xi2)
        return _argmin([n + future_cost(o, state) for n, o in zip(now, options)])
    return policy


def fastest_policy():
    """Least time cost (detour or UGV travel, plus waiting and charging)."""
    return lambda options, state: _argmin([o["t"] for o in options])
