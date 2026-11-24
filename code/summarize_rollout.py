"""One table from every results/rollout_*.txt file.

Usage:  python summarize_rollout.py
"""
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def parse(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    if "Paired difference" not in txt:
        return None
    head = txt.splitlines()[0]
    m = re.search(r"Instance\('([^']+)'.*scale ([\d.]+) battery ([\d.]+) pads (\d+) ugv (\d+) dod ([\d.]+)", head)
    row = {"file": os.path.basename(path), "inst": m.group(1), "scale": m.group(2),
           "battery": m.group(3), "ugv": m.group(5), "dod": m.group(6)}
    for line in txt.splitlines():
        p = line.split()
        if len(p) == 11 and p[0] in ("regret", "nearest", "rollout-air", "rollout-all"):
            row[p[0]] = {"cost": float(p[1]), "late": float(p[3]), "decisions": float(p[8]),
                         "choices": float(p[9]), "overrides": float(p[10])}
        m = re.match(r"\s+(nearest|rollout-air|rollout-all)\s+mean\s+([-+\d.]+) \(([-+\d.]+)%\)\s+"
                     r"95% CI \[([-+\d.]+), ([-+\d.]+)\]\s+better on (\d+), worse on (\d+)", line)
        if m:
            row[m.group(1)].update(diff=float(m.group(2)), pct=float(m.group(3)),
                                   lo=float(m.group(4)), hi=float(m.group(5)),
                                   better=int(m.group(6)), worse=int(m.group(7)),
                                   sig=float(m.group(5)) < 0 or float(m.group(4)) > 0)
    return row


def main():
    rows = [r for r in map(parse, sorted(glob.glob(os.path.join(HERE, "results", "rollout_*.txt")))) if r]
    star = lambda d: "*" if d["sig"] else " "
    print(f"{'instance':10} {'scale':>5} {'Wh':>4} {'ugv':>3} {'dod':>4} | {'regret':>8} {'late':>5} |"
          f" {'nearest %':>9} | {'roll-air %':>10} | {'roll-all %':>10} {'95% CI of difference':>22} "
          f"{'worse days':>10} | {'choices':>7} {'changed':>7}")
    for r in rows:
        a, n, x = r["rollout-all"], r["nearest"], r["rollout-air"]
        print(f"{r['inst']:10} {r['scale']:>5} {float(r['battery']):4.0f} {r['ugv']:>3} {r['dod']:>4} | "
              f"{r['regret']['cost']:8.1f} {r['regret']['late']:5.2f} | "
              f"{n['pct']:+8.1f}{star(n)} | {x['pct']:+9.1f}{star(x)} | {a['pct']:+9.1f}{star(a)} "
              f"{'[%+.0f, %+.0f]' % (a['lo'], a['hi']):>22} {a['worse']:>10} | "
              f"{a['choices']:7.1f} {a['overrides']:7.1f}")
    print("\n* = significant at 5% (paired, 30 days).  Percentages are relative to the regret rule;"
          "\nnegative = cheaper.  'choices' = charging decisions per day with more than one option;"
          "\n'changed' = how many of them rollout-all decided differently.")


if __name__ == "__main__":
    main()
