"""Problem encodings: (graph, K) -> CNF.

A problem is a family of decision problems indexed by a size K ("is there a
solution with K stages / colours?"), assumed monotone: feasible for K implies
feasible for K+1. The minimizer (``solve.py``) searches for the least feasible K.

To add a problem, write ``def encode_x(graph, k, **params) -> Encoding`` and
register it in ``PROBLEMS``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

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


SYMMETRY_MODES = ("none", "zero", "zero_lex")


def lex_leq(pool: IDPool, a: list[int], b: list[int], strict: bool = False) -> list[list[int]]:
    """Clauses enforcing a <=_lex b (a <_lex b if ``strict``) for equal-length
    literal vectors, most significant position first.

    Helper e_i means "a and b agree on positions < i" (e_0 = true). While the
    prefix agrees, position i must satisfy a_i <= b_i, and agreeing there
    forces e_{i+1}. If e_i is not forced, the solver may leave it false, which
    imposes nothing. Strict: full agreement (e_L) is forbidden.
    """
    clauses = []
    e = None  # None stands for the constant true e_0
    for ai, bi in zip(a, b):
        pre = [] if e is None else [-e]
        clauses.append(pre + [-ai, bi])  # prefix equal -> a_i <= b_i
        nxt = pool.id()
        clauses.append(pre + [-ai, -bi, nxt])  # prefix equal, a_i = b_i = 1 -> e_{i+1}
        clauses.append(pre + [ai, bi, nxt])  # prefix equal, a_i = b_i = 0 -> e_{i+1}
        e = nxt
    if strict:
        clauses.append([] if e is None else [-e])
    return clauses


def encode_pair_cover(graph: Graph, k: int, symmetry: str = "none", col_lex: bool = False) -> Encoding:
    """Straightforward CNF for "pair cover with ``k`` stages".

    Variables: ``x[v][s]`` = value of vertex v in stage s, plus one witness
    ``y`` per (edge, pattern, stage). The "exists s" is a disjunction of
    conjunctions, turned into CNF by Tseitin witnesses in the one direction
    needed (y -> conjunction), together with the clause OR_s y_s.

    Optional symmetry breaking (default ``"none"``: the basic encoding):

    * ``symmetry="zero"``: stage 0 is all zeros. Sound because flipping all
      values of one vertex keeps {00, 01, 10, 11} covered on its edges.
    * ``symmetry="zero_lex"``: additionally stages 1..k-1, read as n-bit
      strings, are in non-decreasing lexicographic order (stages are
      interchangeable).
    * ``col_lex=True``: vertex columns over stages 1..k-1 are strictly
      increasing. Only sound when every vertex permutation is an automorphism,
      i.e. for complete graphs (universal sets). Use with ``"zero_lex"``.

    Solutions decode to ``{"stages": k, "rows": [one 0/1 string per stage]}``,
    character v of a row being vertex v's value.
    """
    if symmetry not in SYMMETRY_MODES:
        raise ValueError(f"unknown symmetry {symmetry!r}; known: {SYMMETRY_MODES}")
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

    # --- optional symmetry breaking (adds clauses only) ---
    if symmetry != "none" and k > 0:
        clauses += [[-x[v][0]] for v in range(n)]
    if symmetry == "zero_lex":
        for s in range(1, k - 1):
            clauses += lex_leq(pool, [x[v][s] for v in range(n)], [x[v][s + 1] for v in range(n)])
    if col_lex:
        for v in range(n - 1):
            clauses += lex_leq(pool, x[v][1:], x[v + 1][1:], strict=True)

    def decode(t: set[int]) -> dict:
        rows = ["".join("1" if x[v][s] in t else "0" for v in range(n)) for s in range(k)]
        return {"stages": k, "rows": rows}

    log.debug("pair_cover: n=%d m=%d k=%d symmetry=%s col_lex=%s vars=%d clauses=%d (%.3fs)",
              n, graph.m, k, symmetry, col_lex, pool.top, len(clauses), time.perf_counter() - t0)
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


# --- colouring + universal set ------------------------------------------------
#
# Upper-bound route to pair cover: properly colour G with chi colours, then give
# colour j the j-th column of an optimal universal set of size chi. Adjacent
# vertices get different colours, hence different columns of the set, hence
# all four patterns. The number of stages is that of the universal set; it is
# an upper bound on the pair cover optimum, not always equal to it.


def greedy_clique(graph: Graph) -> list[int]:
    """Simplest greedy clique: scan vertices by decreasing degree (ties by id)
    and keep each vertex that is adjacent to all vertices kept so far."""
    adj = [set(a) for a in graph.adjacency()]
    clique: list[int] = []
    for v in sorted(range(graph.n), key=lambda v: (-len(adj[v]), v)):
        if all(u in adj[v] for u in clique):
            clique.append(v)
    return clique


def encode_coloring(graph: Graph, k: int, universal_sets: Optional[str] = None) -> Encoding:
    """CNF for "``graph`` is properly colourable with ``k`` colours", with
    clique fixing.

    Variables ``c[v][j]`` = vertex v has colour j. Clauses: every vertex has
    at least one colour; adjacent vertices never share a colour. (No
    at-most-one: any true colour of a vertex is a valid choice for it.)
    Symmetry breaking: the vertices of ``greedy_clique`` get colours 0, 1, ...
    in order (colours are interchangeable, and clique vertices need distinct
    colours anyway). For k < clique size this is unsatisfiable.

    Solutions decode to the colouring plus, if the universal set of size k is
    in the table ``universal_sets`` (default: the packaged table), the
    resulting stages: ``{"chi", "colors", "clique", "stages", "rows"}``;
    ``stages``/``rows`` are None when the set is missing.
    """
    if any(i == j for i, j in graph.edges):
        raise ValueError("self-loops make colouring infeasible")
    t0 = time.perf_counter()
    pool = IDPool()
    n = graph.n
    col = [[pool.id(("c", v, j)) for j in range(k)] for v in range(n)]
    clauses = [list(col[v]) for v in range(n)]  # empty for k == 0: unsatisfiable unless n == 0
    clauses += [[-col[u][j], -col[v][j]] for u, v in graph.edges for j in range(k)]
    clique = greedy_clique(graph)
    for j, v in enumerate(clique):
        clauses.append([col[v][j]] if j < k else [])  # clique needs more than k colours

    def decode(t: set[int]) -> dict:
        from .universal import DEFAULT_TABLE, stages_from_coloring  # lazy: universal imports us
        colors = [next(j for j in range(k) if col[v][j] in t) for v in range(n)]
        built = stages_from_coloring(colors, k, universal_sets or DEFAULT_TABLE)
        if built is None:
            log.warning("no precomputed universal set of size %d; reporting colouring only", k)
            built = {"stages": None, "rows": None}
        return {"chi": k, "colors": colors, "clique": clique, **built}

    log.debug("coloring: n=%d m=%d k=%d clique=%d vars=%d clauses=%d (%.3fs)",
              n, graph.m, k, len(clique), pool.top, len(clauses), time.perf_counter() - t0)
    return Encoding(clauses=clauses, pool=pool, decode=decode)


PROBLEMS: dict[str, Callable[..., Encoding]] = {
    "pair_cover": encode_pair_cover,
    "coloring": encode_coloring,
}

# Optional per-problem start of the K search: every smaller K must be infeasible.
K_MIN: dict[str, Callable[[Graph], int]] = {
    "coloring": lambda g: len(greedy_clique(g)),  # a clique of size q needs q colours
}


def encode(problem: str, graph: Graph, k: int, **params) -> Encoding:
    """Encode ``graph`` for size ``k`` with the registered encoder ``problem``."""
    try:
        fn = PROBLEMS[problem]
    except KeyError:
        raise ValueError(f"unknown problem {problem!r}; known: {sorted(PROBLEMS)}") from None
    return fn(graph, k, **params)


def k_min(problem: str, graph: Graph) -> int:
    """Where the K search for ``problem`` may start on ``graph`` (0 if no bound is known)."""
    fn = K_MIN.get(problem)
    return fn(graph) if fn else 0
