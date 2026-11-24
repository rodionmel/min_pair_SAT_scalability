"""Random graph generators. All are deterministic for a given ``seed``.

* ``gnp(n, p)``            -- uniform random graph G(n, p).
* ``triangular(n, p, e)``  -- triangle-shaped piece of the triangular lattice
  with random edge exclusion, raised to the e-th power.
"""

from __future__ import annotations

import itertools
import logging
import random
from collections import deque

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
    """Random planar-based graph: the triangular lattice with ``n`` vertices per
    side (see ``triangular_lattice``), each edge excluded independently with
    probability ``p``, then raised to the power ``e`` (see ``power``).

    The power is taken on the thinned graph, so e=2 links two vertices iff a
    path of length <= 2 survived the exclusion. The vertex count is always
    n(n+1)/2; exclusion can leave vertices isolated.
    """
    _check_prob(p)
    rng = random.Random(seed)
    base = triangular_lattice(n)
    kept = [edge for edge in base.edges if rng.random() >= p]
    g = Graph.from_edges(base.n, kept)
    if e > 1:
        g = power(g, e)
    g = Graph(g.n, g.edges, name=f"triangular n={n} p={p} e={e} seed={seed}")
    log.debug("triangular: n=%d p=%g e=%d seed=%d -> vertices=%d lattice m=%d kept=%d final m=%d",
              n, p, e, seed, g.n, base.m, len(kept), g.m)
    return g


def _check_prob(p: float) -> None:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {p}")


GENERATORS = {"gnp": gnp, "triangular": triangular}
