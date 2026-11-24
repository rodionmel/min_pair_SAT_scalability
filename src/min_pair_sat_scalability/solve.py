"""Search for the least feasible K of a monotone family of SAT instances.

* ``linear`` -- K = 0, 1, 2, ...; the first SAT answer is optimal, and the
  UNSAT answers before it are the optimality proof.
* ``binary`` -- K = 0, 1, 2, 4, 8, ... until SAT, then bisect the gap.

Each K is a fresh CNF solved by a fresh solver. Every call is recorded in
``Result.calls`` for scalability analysis.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from pysat.solvers import Solver

from .problems import Encoding

log = logging.getLogger(__name__)

STRATEGIES = ("linear", "binary")
DEFAULT_SOLVER = "glucose4"
# Name prefixes of PySAT bindings without interrupt() support: they cannot
# honour an in-process timeout.
NON_INTERRUPTIBLE = ("cadical", "lingeling")


@dataclass
class Result:
    """Outcome of one minimization run; serialized as-is into job results.

    ``status`` is ``"optimal"`` (``cost`` = least feasible K, proven),
    ``"unsat"`` (no K <= ``k_max`` is feasible), ``"timeout"`` (``cost`` is the
    best K found, if any; every K < ``lower_bound`` is proven infeasible) or
    ``"error"``. ``nvars``/``nclauses`` describe the instance of the best K.
    ``calls`` holds one entry per SAT call: ``k``, ``result``
    (True/False/None = SAT/UNSAT/interrupted), ``time`` (solve seconds),
    ``encode_time``, ``nvars``, ``nclauses``.
    """

    status: str
    cost: Optional[int] = None
    lower_bound: int = 0
    solution: Any = None
    wall_time: float = 0.0
    nvars: int = 0
    nclauses: int = 0
    calls: list[dict] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Plain-dict form for JSON output."""
        return asdict(self)


def minimize(build: Callable[[int], Encoding], strategy: str = "linear", solver: str = DEFAULT_SOLVER,
             timeout: Optional[float] = None, k_max: Optional[int] = None, k_min: int = 0) -> Result:
    """Find the least K for which ``build(K)`` is satisfiable.

    ``build`` must be monotone (SAT at K implies SAT at K+1). ``solver`` is
    any PySAT solver name; ``timeout`` (seconds) bounds the whole run;
    ``k_max`` caps the search (without it, a family that is never satisfiable
    would loop forever). ``k_min`` starts the search there; the caller vouches
    that every smaller K is infeasible (it becomes the initial lower bound).
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; known: {STRATEGIES}")
    if timeout is not None and solver.startswith(NON_INTERRUPTIBLE):
        raise ValueError(f"solver {solver!r} cannot be interrupted, so it cannot enforce a timeout; "
                         "use e.g. glucose4/minisat22/maplechrono, or drop the timeout and rely on "
                         "the job's wall-clock limit")
    log.info("minimize: strategy=%s solver=%s timeout=%s k_min=%d k_max=%s",
             strategy, solver, timeout, k_min, k_max)
    t0 = time.perf_counter()
    res = Result(status="timeout", lower_bound=k_min)
    search = _Search(build, solver, None if timeout is None else t0 + timeout, res)
    (_linear if strategy == "linear" else _binary)(search, k_max)
    res.wall_time = time.perf_counter() - t0
    log.info("minimize: %s K=%s lower_bound=%s calls=%d %.3fs",
             res.status, res.cost, res.lower_bound, len(res.calls), res.wall_time)
    return res


class _Search:
    """Solves single K values and keeps the best/infeasible bookkeeping in ``res``."""

    def __init__(self, build: Callable[[int], Encoding], solver: str, deadline: Optional[float], res: Result):
        self.build = build
        self.solver = solver
        self.deadline = deadline
        self.res = res

    def try_k(self, k: int) -> Optional[bool]:
        """Encode and solve K = ``k``: True/False, or None if the deadline hit.

        A SAT answer becomes the incumbent; an UNSAT answer raises the lower bound.
        """
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            log.info("K=%d: skipped, deadline passed", k)
            return None
        t = time.perf_counter()
        enc = self.build(k)
        enc_time = time.perf_counter() - t
        with Solver(name=self.solver, bootstrap_with=enc.clauses) as s:
            t = time.perf_counter()
            out = self._solve(s)
            dt = time.perf_counter() - t
            if out:
                self.res.cost = k
                self.res.solution = enc.decode({lit for lit in s.get_model() if lit > 0})
                self.res.nvars, self.res.nclauses = enc.nvars, len(enc.clauses)
            elif out is False:
                self.res.lower_bound = max(self.res.lower_bound, k + 1)
        self.res.calls.append({"k": k, "result": out, "time": dt, "encode_time": enc_time,
                               "nvars": enc.nvars, "nclauses": len(enc.clauses)})
        log.info("K=%d -> %s in %.3fs (encode %.3fs, vars=%d clauses=%d)", k,
                 {True: "SAT", False: "UNSAT", None: "INTERRUPTED"}[out], dt, enc_time,
                 enc.nvars, len(enc.clauses))
        return out

    def _solve(self, s: Solver) -> Optional[bool]:
        if self.deadline is None:
            return s.solve()
        # The solver blocks this thread, so a timer thread interrupts it when
        # the remaining budget runs out; solve_limited then returns None.
        timer = threading.Timer(max(0.0, self.deadline - time.perf_counter()), s.interrupt)
        timer.start()
        try:
            return s.solve_limited(expect_interrupt=True)
        finally:
            timer.cancel()


def _linear(search: _Search, k_max: Optional[int]) -> None:
    """K = k_min, k_min+1, ... until the first SAT (optimal) or UNSAT at ``k_max``."""
    k = search.res.lower_bound
    while k_max is None or k <= k_max:
        out = search.try_k(k)
        if out is None:
            return
        if out:
            search.res.status = "optimal"
            return
        k += 1
    search.res.status = "unsat"


def _binary(search: _Search, k_max: Optional[int]) -> None:
    """Doubling (k_min, 2*k_min, ...; from 0: 0, 1, 2, 4, ...) until SAT, then
    bisection between the last UNSAT and the first SAT K."""
    res = search.res
    k = res.lower_bound
    while True:  # 1. find some feasible K
        if k_max is not None and k > k_max:
            if k_max >= res.lower_bound:  # the cap itself was skipped by doubling
                k = k_max
            else:
                res.status = "unsat"
                return
        out = search.try_k(k)
        if out is None:
            return
        if out:
            break
        k = 1 if k == 0 else 2 * k
    # 2. invariant: every K < lower_bound is infeasible, K = cost is feasible
    while res.lower_bound < res.cost:
        mid = (res.lower_bound + res.cost) // 2
        log.debug("window [%d, %d], trying K=%d", res.lower_bound, res.cost, mid)
        if search.try_k(mid) is None:
            return
    res.status = "optimal"
