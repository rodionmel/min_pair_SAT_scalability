"""Problem encodings: (graph, K) -> CNF.

A problem is a family of decision problems indexed by a size K ("is there a
solution with K stages?"), assumed monotone: feasible for K implies feasible
for K+1. The minimizer (``solve.py``) searches for the least feasible K.

To add a problem, write ``def encode_x(graph, k, **params) -> Encoding`` and
register it in ``PROBLEMS``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from pysat.formula import IDPool

from .graph import Graph

log = logging.getLogger(__name__)


@dataclass
class Encoding:
    """CNF for one fixed K, independent of any solver.

    ``clauses`` use variables allocated from ``pool`` (``pool.top`` is the
    highest id in use).
    """

    clauses: list[list[int]]
    pool: IDPool
    decode: Callable[[set[int]], Any]
    """Turns the set of true literals of a model into a JSON-serializable solution."""

    @property
    def nvars(self) -> int:
        """Number of variables used by the encoding."""
        return self.pool.top


# --- pair cover --------------------------------------------------------------
#
# Assign every vertex a 0/1 value in each of K stages so that every edge
# (i, j) shows all four patterns 00, 01, 10, 11 in some stage:
#
#   for each edge (i, j), for each (a, b) in {00, 01, 10, 11}:
#       exists stage s: x[i][s] = a and x[j][s] = b

PATTERNS = ((0, 0), (0, 1), (1, 0), (1, 1))


def encode_pair_cover(graph: Graph, k: int) -> Encoding:
    """Straightforward CNF for "pair cover with ``k`` stages".

    Variables: ``x[v][s]`` = value of vertex v in stage s, plus one witness
    ``y`` per (edge, pattern, stage). The "exists s" is a disjunction of
    conjunctions, turned into CNF by Tseitin witnesses in the one direction
    needed (y -> conjunction), together with the clause OR_s y_s.

    Solutions decode to ``{"stages": k, "rows": [one 0/1 string per stage]}``,
    character v of a row being vertex v's value.
    """
    if any(i == j for i, j in graph.edges):
        raise ValueError("self-loops make pair cover infeasible (01 never occurs)")
    t0 = time.perf_counter()
    pool = IDPool()
    n = graph.n
    x = [[pool.id(("x", v, s)) for s in range(k)] for v in range(n)]

    clauses = []
    for i, j in graph.edges:
        for a, b in PATTERNS:
            witnesses = []
            for s in range(k):
                # y -> (x[i][s] == a) and (x[j][s] == b). The solver only sets
                # y true when stage s really shows the pattern, so the reverse
                # implication is not needed.
                y = pool.id()  # anonymous: keeps IDPool's object map small
                witnesses.append(y)
                clauses.append([-y, x[i][s] if a else -x[i][s]])
                clauses.append([-y, x[j][s] if b else -x[j][s]])
            clauses.append(witnesses)  # empty when k == 0: unsatisfiable, as it should be

    def decode(t: set[int]) -> dict:
        rows = ["".join("1" if x[v][s] in t else "0" for v in range(n)) for s in range(k)]
        return {"stages": k, "rows": rows}

    log.debug("pair_cover: n=%d m=%d k=%d vars=%d clauses=%d (%.3fs)",
              n, graph.m, k, pool.top, len(clauses), time.perf_counter() - t0)
    return Encoding(clauses=clauses, pool=pool, decode=decode)


def check_pair_cover(graph: Graph, rows: list[str]) -> list[tuple[tuple[int, int], str]]:
    """Return the (edge, pattern) pairs that ``rows`` fail to cover (empty = valid).

    ``rows`` is one string per stage, character ``v`` being vertex v's value.
    """
    missing = []
    for i, j in graph.edges:
        seen = {row[i] + row[j] for row in rows}
        missing += [((i, j), p) for p in ("00", "01", "10", "11") if p not in seen]
    return missing


PROBLEMS: dict[str, Callable[..., Encoding]] = {
    "pair_cover": encode_pair_cover,
}


def encode(problem: str, graph: Graph, k: int, **params) -> Encoding:
    """Encode ``graph`` for size ``k`` with the registered encoder ``problem``."""
    try:
        fn = PROBLEMS[problem]
    except KeyError:
        raise ValueError(f"unknown problem {problem!r}; known: {sorted(PROBLEMS)}") from None
    return fn(graph, k, **params)
