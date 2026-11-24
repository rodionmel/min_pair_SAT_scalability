"""Random graph generators. All are deterministic for a given ``seed``.

* ``gnp(n, p)``            -- uniform random graph G(n, p).
* ``triangular(n, p, e)``  -- e-th power of a triangle-shaped piece of the
  triangular lattice, with random edge exclusion.
"""

from __future__ import annotations

import itertools
import logging
import math
import random
from collections import deque
from typing import Optional

from .graph import Graph

log = logging.getLogger(__name__)


def gnp(n: int, p: float, seed: int = 0) -> Graph:
    """Erdos-Renyi G(n, p): each of the n(n-1)/2 possible edges is included
    independently with probability ``p``."""
    _check_prob(p)
    rng = random.Random(seed)
    edges = [e for e in itertools.combinations(range(n), 2) if rng.random() < p]
    g = Graph.from_edges(n, edges, name=f"gnp n={n} p={p} seed={seed}")
    log.debug("gnp: n=%d p=%g seed=%d -> m=%d", n, p, seed, g.m)
    return g


def triangular_lattice(n: int) -> Graph:
    """Triangle-shaped piece of the triangular lattice with ``n`` vertices per side.

    Row r (0 <= r < n) has r+1 vertices (r, 0..r), numbered row by row.
    Vertex (r, c) is adjacent to (r, c+1) in its row and to (r+1, c) and
    (r+1, c+1) below it, so every unit cell is a triangle. Total:
    n(n+1)/2 vertices, 3n(n-1)/2 edges; planar, max degree 6.
    """
    def vid(r: int, c: int) -> int:
        return r * (r + 1) // 2 + c

    edges = []
    for r in range(n):
        for c in range(r + 1):
            if c < r:
                edges.append((vid(r, c), vid(r, c + 1)))
            if r + 1 < n:
                edges.append((vid(r, c), vid(r + 1, c)))
                edges.append((vid(r, c), vid(r + 1, c + 1)))
    return Graph.from_edges(n * (n + 1) // 2, edges, name=f"trilattice n={n}")


def power(g: Graph, e: int) -> Graph:
    """The e-th power of ``g``: u ~ v iff 1 <= dist_g(u, v) <= e (e=1 returns g's edges).

    Computed by a breadth-first search of depth ``e`` from every vertex.
    """
    if e < 1:
        raise ValueError("power e must be >= 1")
    adj = g.adjacency()
    edges = []
    for src in range(g.n):
        dist = {src: 0}
        queue = deque([src])
        while queue:
            v = queue.popleft()
            if dist[v] == e:
                continue  # do not expand beyond depth e
            for w in adj[v]:
                if w not in dist:
                    dist[w] = dist[v] + 1
                    queue.append(w)
        edges += [(src, v) for v in dist if v > src]  # each pair once
    return Graph.from_edges(g.n, edges, name=g.name)


def triangular(n: int, p: float, e: int = 1, seed: int = 0) -> Graph:
    """Random planar-based graph: the e-th power (see ``power``) of the
    triangular lattice with ``n`` vertices per side (see ``triangular_lattice``),
    then each edge of that power graph excluded independently with probability ``p``.

    The vertex count is always n(n+1)/2; exclusion can leave vertices isolated.
    """
    _check_prob(p)
    rng = random.Random(seed)
    full = power(triangular_lattice(n), e)
    kept = [edge for edge in full.edges if rng.random() >= p]
    g = Graph(full.n, tuple(kept), name=f"triangular n={n} p={p} e={e} seed={seed}")
    log.debug("triangular: n=%d p=%g e=%d seed=%d -> vertices=%d power m=%d kept m=%d",
              n, p, e, seed, g.n, full.m, g.m)
    return g


def _check_prob(p: float) -> None:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {p}")


GENERATORS = {"gnp": gnp, "triangular": triangular}


def generate(model: str, n: int, p: float, e: Optional[int] = None, seed: int = 0) -> tuple[Graph, str]:
    """Build a graph of the given model and return it with its canonical file
    name, e.g. ``gnp_n50_p0.3_s2.dimacs`` / ``tri_n8_p0.1_e2_s0.dimacs``."""
    if model == "gnp":
        return gnp(n, p, seed), f"gnp_n{n}_p{p}_s{seed}.dimacs"
    if model == "triangular":
        e = 1 if e is None else e
        return triangular(n, p, e, seed), f"tri_n{n}_p{p}_e{e}_s{seed}.dimacs"
    raise ValueError(f"unknown model {model!r}; known: {sorted(GENERATORS)}")


def geometric_sizes(start: int, stop: int, factor: float = math.sqrt(2)) -> list[int]:
    """Sizes start, start*f, start*f^2, ... <= stop, rounded half up, duplicates dropped.

    With the default factor sqrt(2), start=8 gives 8, 11, 16, 23, 32, 45, 64, 91, 128.
    """
    if start < 1 or factor <= 1:
        raise ValueError("need start >= 1 and factor > 1")
    sizes: list[int] = []
    i = 0
    while True:
        s = int(start * factor ** i + 0.5)
        if s > stop:
            return sizes
        if not sizes or s != sizes[-1]:
            sizes.append(s)
        i += 1
