"""Loader for the Figshare instances of Fang et al. (2026).

File format (tab separated):
    StringID  Type  X  Y  EarliestStart  LatestStart  ServiceTime  Demand
Type is d (depot), c (customer) or r (charging station).
Node 0 is always the depot; customers follow, then stations.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Node:
    name: str
    kind: str      # 'd', 'c', 'r'
    x: float
    y: float
    e: float       # preferred earliest service start
    l: float       # preferred latest service start
    st: float      # service time
    demand: float


class Instance:
    def __init__(self, nodes: list[Node], name: str = ""):
        depots = [n for n in nodes if n.kind == "d"]
        if len(depots) != 1:
            raise ValueError(f"expected exactly one depot, found {len(depots)}")
        custs = [n for n in nodes if n.kind == "c"]
        stats = [n for n in nodes if n.kind == "r"]
        self.name = name
        self.nodes = depots + custs + stats
        self.depot = 0
        self.customers = list(range(1, 1 + len(custs)))
        self.stations = list(range(1 + len(custs), len(self.nodes)))
        n = len(self.nodes)
        self.d = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                dij = math.hypot(self.nodes[i].x - self.nodes[j].x,
                                 self.nodes[i].y - self.nodes[j].y)
                self.d[i][j] = self.d[j][i] = dij
        refuges = [self.depot] + self.stations
        # distance from every node to its nearest refuge (depot or station)
        self.ref_d = [min(self.d[i][r] for r in refuges) for i in range(n)]

    def subset(self, n_customers: int, name: str = "") -> "Instance":
        """Depot, the first n_customers customers, and all stations."""
        keep = ([self.nodes[0]] + [self.nodes[i] for i in self.customers[:n_customers]]
                + [self.nodes[i] for i in self.stations])
        return Instance(keep, name or f"{self.name}[:{n_customers}]")

    def __repr__(self):
        return (f"Instance({self.name!r}, customers={len(self.customers)}, "
                f"stations={len(self.stations)})")


def load(path: str) -> Instance:
    nodes = []
    with open(path, encoding="utf-8") as fh:
        header = fh.readline()
        if not header.lower().startswith("stringid"):
            raise ValueError(f"unexpected header in {path}: {header!r}")
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 8:
                raise ValueError(f"short line in {path}: {line!r}")
            nodes.append(Node(parts[0], parts[1], float(parts[2]), float(parts[3]),
                              float(parts[4]), float(parts[5]), float(parts[6]),
                              float(parts[7])))
    import os
    return Instance(nodes, os.path.splitext(os.path.basename(path))[0])
