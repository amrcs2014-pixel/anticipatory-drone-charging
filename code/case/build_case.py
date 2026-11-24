"""Build the Amman case instance from real OpenStreetMap locations.

    python case/build_case.py

Real:       locations of the depot (a hospital), the customers (pharmacies) and the
            fixed charging stations (fuel stations), from case/osm_amman.json.
Synthetic:  time windows, service times and demands, which are those of the benchmark
            instance C75_R5 assigned to the pharmacies at random (seed 0).
Coordinates are written in units of 0.25 km so that the scripts run with their defaults.
"""
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from drp.instance import load          # noqa: E402

UNIT = 0.25          # km per coordinate unit
N_CUST, N_OUTER, SEED = (int(sys.argv[1]) if len(sys.argv) > 1 else 75), 4, 0


def main():
    osm = json.load(open(os.path.join(HERE, "osm_amman.json"), encoding="utf-8"))
    s, w, n, e = osm["bbox"]
    lat0 = math.radians((s + n) / 2)

    def xy(el):          # km east and north of the south-west corner
        return ((el["lon"] - w) * 111.320 * math.cos(lat0), (el["lat"] - s) * 110.574)
    dist = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
    by = lambda k: [dict(el, p=xy(el)) for el in osm["elements"] if el["kind"] == k]
    rng = random.Random(SEED)

    pharm = []
    for el in sorted(by("pharmacy"), key=lambda q: q["osm"]):
        if all(dist(el["p"], q["p"]) > 0.15 for q in pharm):       # drop duplicates within 150 m
            pharm.append(el)
    cx = sum(q["p"][0] for q in pharm) / len(pharm)
    cy = sum(q["p"][1] for q in pharm) / len(pharm)
    depot = min(by("hospital"), key=lambda q: dist(q["p"], (cx, cy)))
    cust = rng.sample(pharm, N_CUST)

    fuel = [f for f in by("fuel") if 5.0 <= dist(f["p"], depot["p"]) <= 11.0]
    outer = []
    while len(outer) < N_OUTER:           # farthest-point selection, starting from the depot
        chosen = [depot] + outer
        outer.append(max(fuel, key=lambda f: (min(dist(f["p"], c["p"]) for c in chosen), f["osm"])))

    ref = load(os.path.join(HERE, "..", "..", "figshare_instances", "extracted", "DATA", "DATA", "C",
                            "C75_R5.txt"))
    attrs = [(ref.nodes[i].e, ref.nodes[i].l, ref.nodes[i].st, ref.nodes[i].demand) for i in ref.customers]
    if N_CUST > len(attrs):            # larger case: add the windows and demands of R75_R3
        ref2 = load(os.path.join(HERE, "..", "..", "figshare_instances", "extracted", "DATA", "DATA", "R",
                                 "R75_R3.txt"))
        attrs += [(min(ref2.nodes[i].e, ref.nodes[0].l), min(ref2.nodes[i].l, ref.nodes[0].l),
                   ref2.nodes[i].st, ref2.nodes[i].demand) for i in ref2.customers]
        attrs = attrs[:N_CUST]
    rng.shuffle(attrs)
    horizon = ref.nodes[0].l
    u = lambda p: (round(p[0] / UNIT, 2), round(p[1] / UNIT, 2))
    rows = [("0", "d", *u(depot["p"]), 0, horizon, 0, 0.0)]
    for i, (c, a) in enumerate(zip(cust, attrs), 1):
        rows.append((str(i), "c", *u(c["p"]), *a))
    for k, st in enumerate([depot] + outer):
        rows.append((str(N_CUST + 1 + k), "r", *u(st["p"]), 0, horizon, 0, 0.0))
    path = os.path.join(HERE, f"AMMAN{N_CUST}_R5.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("StringID\tType\tX\tY\tEarliestStart\tLatestStart\tServiceTime\tDemand\n")
        for r in rows:
            fh.write("\t".join(f"{v:g}" if isinstance(v, float) else str(v) for v in r) + "\t\n")
    meta = {"osm_timestamp": osm.get("timestamp"), "bbox": osm["bbox"], "unit_km": UNIT, "seed": SEED,
            "depot": {k: depot[k] for k in ("osm", "name", "lat", "lon")},
            "stations": [{k: f[k] for k in ("osm", "name", "lat", "lon")} for f in outer],
            "customers": [{k: c[k] for k in ("osm", "name", "lat", "lon")} for c in cust]}
    json.dump(meta, open(os.path.join(HERE, f"AMMAN{N_CUST}_R5_sources.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    dd = sorted(dist(c["p"], depot["p"]) for c in cust)
    print(f"pharmacies after de-duplication: {len(pharm)}; depot: {depot['name']!r} ({depot['osm']})")
    print(f"customer distance from depot, km: min {dd[0]:.1f}, median {dd[len(dd) // 2]:.1f}, max {dd[-1]:.1f}")
    print("outer stations, km from depot:", [round(dist(f["p"], depot["p"]), 1) for f in outer],
          [f["name"] for f in outer])
    xs = [c["p"][0] for c in cust]; ys = [c["p"][1] for c in cust]
    print(f"extent of customers: {max(xs) - min(xs):.1f} km east-west, {max(ys) - min(ys):.1f} km north-south")
    print("written", path)


if __name__ == "__main__":
    main()
