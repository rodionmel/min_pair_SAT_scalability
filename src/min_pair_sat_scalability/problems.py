"""Problem encodings: graph -> CNF + objective.

An encoding is solver-agnostic: hard clauses plus a list of objective
literals. The cost of a model is the number of objective literals it makes
true; the minimizer (``solve.py``) finds a model of minimum cost.

To add a problem, write ``def encode_x(graph, **params) -> Encoding`` and
register it in ``PROBLEMS``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from math import comb
from typing import Any, Callable

from pysat.formula import IDPool

from .graph import Graph

log = logging.getLogger(__name__)


@dataclass
class Encoding:
    """A minimization problem in CNF form, independent of any solver.

    ``clauses`` are the hard constraints over variables allocated from
    ``pool`` (so ``pool.top`` is the highest variable id in use). The cost of
    a model is the number of ``objective`` literals it satisfies.
    """

    clauses: list[list[int]]
    objective: list[int]
    pool: IDPool
    decode: Callable[[set[int]], Any]
    """Turns the set of true literals of a model into a JSON-serializable solution."""
    info: dict = field(default_factory=dict)
    """Encoder metadata (bounds, sizes...) copied into the result."""

    @property
    def nvars(self) -> int:
        """Number of variables used by the encoding."""
        return self.pool.top

    def cost(self, true_lits: set[int]) -> int:
        """Objective value of a model given as its set of true literals."""
        return sum(1 for lit in self.objective if lit in true_lits)


def encode_vertex_cover(graph: Graph) -> Encoding:
    """Minimum vertex cover (x_v <-> v in the cover); kept as a simple test
    problem for the minimizer."""
    pool = IDPool()
    x = [pool.id(("x", v)) for v in range(graph.n)]
    clauses = [[x[u], x[v]] for u, v in graph.edges]

    def decode(true_lits: set[int]) -> dict:
        return {"cover": [v for v in range(graph.n) if x[v] in true_lits]}

    return Encoding(clauses=clauses, objective=x, pool=pool, decode=decode)


# --- pair cover (binary covering array on a graph) ---------------------------
#
# Find K stages (rows) of 0/1 values for every vertex such that each edge
# (i, j) sees all four patterns 00, 01, 10, 11 in some stage; minimize K.
#
# Structure used:
# * Flipping a vertex's column maps {00,01,10,11} to itself, so WLOG stage 0
#   is all zeros -- it covers 00 for every edge.
# * Upper bound: give each colour class of a proper colouring one column of an
#   optimal covering array for chi columns -> K <= can2(chi).
# * Lower bound: a clique of size w needs K >= can2(w); any edge needs K >= 4.


def can2(k: int) -> int:
    """Rows of an optimal binary strength-2 covering array with k >= 2 columns
    (Kleitman-Spencer / Katona): least N with C(N-1, ceil(N/2)) >= k."""
    n = 4
    while comb(n - 1, (n + 1) // 2) < k:
        n += 1
    return n


def greedy_coloring(graph: Graph) -> list[int]:
    """Largest-degree-first greedy colouring; returns the colour of each vertex."""
    adj = graph.adjacency()
    color = [-1] * graph.n
    for v in sorted(range(graph.n), key=lambda v: -len(adj[v])):
        used = {color[w] for w in adj[v]}
        color[v] = next(c for c in range(graph.n + 1) if c not in used)
    return color


def greedy_clique(graph: Graph) -> list[int]:
    """Largest of the greedy cliques grown from each vertex; returns its vertices.

    From each start vertex, repeatedly add the candidate with the most
    neighbours among the remaining candidates (the common neighbours of the
    clique so far).
    """
    adj = [set(a) for a in graph.adjacency()]
    order = sorted(range(graph.n), key=lambda v: -len(adj[v]))
    best: list[int] = []
    for start in order:
        # A clique containing `start` has at most deg+1 vertices; starts come
        # in decreasing degree, so once deg+1 <= |best| no later start can win.
        if len(adj[start]) + 1 <= len(best):
            break
        clique, cand = [start], set(adj[start])
        while cand:
            v = max(cand, key=lambda w: len(adj[w] & cand))
            clique.append(v)
            cand &= adj[v]
        if len(clique) > len(best):
            best = clique
    return best


def check_pair_cover(graph: Graph, rows: list[str]) -> list[tuple[tuple[int, int], str]]:
    """Return the (edge, pattern) pairs that ``rows`` fail to cover (empty = valid).

    ``rows`` is one string per stage, character ``v`` being vertex v's value.
    """
    missing = []
    for i, j in graph.edges:
        seen = {row[i] + row[j] for row in rows}
        missing += [((i, j), p) for p in ("00", "01", "10", "11") if p not in seen]
    return missing


def encode_pair_cover(graph: Graph, k_max: int | None = None, zero_stage: bool = True,
                      bounds: bool = True) -> Encoding:
    """Encode the pair cover problem with a stage budget of K = ``k_max``.

    Variables: ``x[v][s]`` = value of vertex v in stage s; ``u[s]`` = stage s
    is used. Used stages form a prefix and the objective is their count.

    ``k_max`` overrides the colouring upper bound, ``bounds=False`` replaces
    the clique lower bound by the trivial 4, ``zero_stage=False`` disables the
    all-zeros-stage symmetry breaking. Solutions decode to
    ``{"stages": K, "rows": [one 0/1 string per used stage]}``.
    """
    if any(i == j for i, j in graph.edges):
        raise ValueError("self-loops make pair cover infeasible (01 never occurs)")
    t0 = time.perf_counter()
    pool = IDPool()
    if graph.m == 0:
        log.info("pair_cover: no edges, 0 stages needed")
        return Encoding([], [], pool, lambda t: {"stages": 0, "rows": []}, {"lb": 0, "ub": 0})

    chi_greedy = len(set(greedy_coloring(graph)))
    omega_greedy = len(greedy_clique(graph))
    ub = can2(chi_greedy) if k_max is None else k_max
    lb = min(ub, can2(omega_greedy) if bounds else 4)
    log.debug("pair_cover: greedy chi=%d -> ub=%d, greedy omega=%d -> lb=%d",
              chi_greedy, ub, omega_greedy, lb)

    n, K = graph.n, ub
    u = [pool.id(("u", s)) for s in range(K)]
    x = [[pool.id(("x", v, s)) for s in range(K)] for v in range(n)]

    # u[s+1] -> u[s]: used stages are a prefix, which removes the symmetry of
    # choosing *which* stages are unused. The first lb stages are always used.
    clauses = [[-u[s + 1], u[s]] for s in range(K - 1)]
    clauses += [[u[s]] for s in range(lb)]

    if zero_stage:
        # Stage 0 (always used, lb >= 4) is all zeros and covers 00 for every
        # edge; the other patterns can only appear in stages 1..K-1.
        clauses += [[-x[v][0]] for v in range(n)]
        patterns, stages = [(0, 1), (1, 0), (1, 1)], range(1, K)
    else:
        patterns, stages = [(0, 0), (0, 1), (1, 0), (1, 1)], range(K)

    # Coverage. For each edge (i, j) and pattern (a, b), a witness variable
    # y_s per stage s implies "stage s is used and (x_i_s, x_j_s) = (a, b)",
    # and the clause OR_s y_s requires one witness. Only this direction is
    # needed: the solver sets a y true only when the stage really shows the
    # pattern. The u[s] link is skipped for s < lb, where u[s] is a unit fact.
    for i, j in graph.edges:
        for a, b in patterns:
            witnesses = []
            for s in stages:
                y = pool.id()  # anonymous: keeps IDPool's object map small
                witnesses.append(y)
                clauses.append([-y, x[i][s] if a else -x[i][s]])
                clauses.append([-y, x[j][s] if b else -x[j][s]])
                if s >= lb:
                    clauses.append([-y, u[s]])
            clauses.append(witnesses)

    def decode(t: set[int]) -> dict:
        used = [s for s in range(K) if u[s] in t]
        rows = ["".join("1" if x[v][s] in t else "0" for v in range(n)) for s in used]
        return {"stages": len(used), "rows": rows}

    info = {"lb": lb, "ub": ub, "chi_greedy": chi_greedy, "omega_greedy": omega_greedy}
    log.info("pair_cover: n=%d m=%d K in [%d, %d] vars=%d clauses=%d (%.3fs)",
             n, graph.m, lb, ub, pool.top, len(clauses), time.perf_counter() - t0)
    return Encoding(clauses=clauses, objective=u, pool=pool, decode=decode, info=info)


PROBLEMS: dict[str, Callable[..., Encoding]] = {
    "pair_cover": encode_pair_cover,
    "vertex_cover": encode_vertex_cover,
}


def encode(problem: str, graph: Graph, **params) -> Encoding:
    """Encode ``graph`` with the registered encoder ``problem``; ``params`` go to the encoder."""
    try:
        fn = PROBLEMS[problem]
    except KeyError:
        raise ValueError(f"unknown problem {problem!r}; known: {sorted(PROBLEMS)}") from None
    return fn(graph, **params)
