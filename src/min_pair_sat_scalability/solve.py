"""Minimization strategies over an ``Encoding``.

* ``linear``  -- SAT-UNSAT descent: find a model of cost c, require <= c-1, repeat.
* ``binary``  -- binary search on the cost bound.
* ``maxsat``  -- RC2 core-guided MaxSAT (objective literals as unit soft clauses).

``linear`` and ``binary`` share one incremental SAT solver; the bound is a
totalizer output passed as an assumption, so no clauses are ever removed.
Every SAT call is recorded in ``Result.calls`` for scalability analysis.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from pysat.card import ITotalizer
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
from pysat.solvers import Solver

from .problems import Encoding

log = logging.getLogger(__name__)

STRATEGIES = ("linear", "binary", "maxsat")
DEFAULT_SOLVER = "glucose4"
# Name prefixes of PySAT bindings without interrupt() support: they cannot
# honour an in-process timeout.
NON_INTERRUPTIBLE = ("cadical", "lingeling")


@dataclass
class Result:
    """Outcome of one minimization run; serialized as-is into job results.

    ``status`` is ``"optimal"`` (cost proven minimal), ``"unsat"`` (no
    solution at all), ``"timeout"`` (``cost`` is the best found, if any, and
    ``lower_bound`` the best proven bound) or ``"error"``. ``calls`` holds one
    ``{"bound", "result", "time"}`` entry per SAT call, where ``bound`` is the
    cost bound k asked for (``None`` = unbounded) and ``result`` is
    True/False/None (SAT/UNSAT/interrupted).
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


def minimize(enc: Encoding, strategy: str = "linear", solver: str = DEFAULT_SOLVER,
             timeout: Optional[float] = None) -> Result:
    """Find a model of ``enc`` with the fewest true objective literals.

    ``solver`` is any PySAT solver name. ``timeout`` (seconds, whole run)
    is enforced for ``linear``/``binary`` only; ``maxsat`` relies on the
    job's wall-clock limit. Raises ``ValueError`` for an unknown strategy or
    a timeout combined with a solver that cannot be interrupted.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; known: {STRATEGIES}")
    if timeout is not None and strategy != "maxsat" and solver.startswith(NON_INTERRUPTIBLE):
        raise ValueError(f"solver {solver!r} cannot be interrupted, so it cannot enforce a timeout; "
                         "use e.g. glucose4/minisat22/maplechrono, or drop the timeout and rely on "
                         "the job's wall-clock limit")
    log.info("minimize: strategy=%s solver=%s timeout=%s vars=%d clauses=%d objective=%d",
             strategy, solver, timeout, enc.nvars, len(enc.clauses), len(enc.objective))
    t0 = time.perf_counter()
    res = Result(status="timeout", nvars=enc.nvars, nclauses=len(enc.clauses))
    if strategy == "maxsat":
        _maxsat(enc, solver, res)
    else:
        deadline = None if timeout is None else t0 + timeout
        _sat_search(enc, strategy, solver, deadline, res)
    res.wall_time = time.perf_counter() - t0
    log.info("minimize: %s cost=%s lower_bound=%s calls=%d %.3fs",
             res.status, res.cost, res.lower_bound, len(res.calls), res.wall_time)
    return res


class _Oracle:
    """Incremental SAT solver with a run-wide deadline enforced by interrupt().

    Each call to ``solve`` is appended to ``res.calls`` and logged.
    """

    def __init__(self, enc: Encoding, solver: str, deadline: Optional[float], res: Result):
        t = time.perf_counter()
        self.s = Solver(name=solver, bootstrap_with=enc.clauses)
        log.debug("oracle: loaded %d clauses into %s in %.3fs", len(enc.clauses), solver,
                  time.perf_counter() - t)
        self.deadline = deadline
        self.res = res

    def solve(self, bound: Optional[int], assumptions=()) -> Optional[bool]:
        """Solve under ``assumptions``; True/False, or None if the deadline hit.

        ``bound`` is only recorded (it is what the assumptions encode).
        """
        t = time.perf_counter()
        if self.deadline is None:
            out = self.s.solve(assumptions=list(assumptions))
        else:
            left = self.deadline - t
            if left <= 0:
                log.info("call #%d bound<=%s: skipped, deadline passed", len(self.res.calls) + 1, bound)
                return None
            # The solver blocks this thread, so a timer thread interrupts it
            # when the remaining budget runs out; solve_limited then returns None.
            timer = threading.Timer(left, self.s.interrupt)
            timer.start()
            try:
                out = self.s.solve_limited(assumptions=list(assumptions), expect_interrupt=True)
            finally:
                timer.cancel()
                self.s.clear_interrupt()  # otherwise the next call aborts immediately
        dt = time.perf_counter() - t
        self.res.calls.append({"bound": bound, "result": out, "time": dt})
        log.info("call #%d bound<=%s -> %s in %.3fs", len(self.res.calls), bound,
                 {True: "SAT", False: "UNSAT", None: "INTERRUPTED"}[out], dt)
        return out

    def true_lits(self) -> set[int]:
        """Positive literals of the last model."""
        return {lit for lit in self.s.get_model() if lit > 0}


def _sat_search(enc: Encoding, strategy: str, solver: str, deadline, res: Result) -> None:
    """``linear`` / ``binary`` search; fills ``res`` in place."""
    oracle = _Oracle(enc, solver, deadline, res)
    try:
        # 1. Unbounded call: feasibility and a first upper bound.
        first = oracle.solve(bound=None)
        if first is None:
            return
        if first is False:
            res.status = "unsat"
            return

        def record(true_lits: set[int]) -> int:
            """Store a model as the incumbent and return its cost."""
            c = enc.cost(true_lits)
            res.cost, res.solution = c, enc.decode(true_lits)
            log.info("new best cost=%d", c)
            return c

        best = record(oracle.true_lits())
        if best == 0:
            res.status = "optimal"
            return

        # 2. Totalizer over the objective literals. Its outputs rhs[k] mean
        #    "at least k+1 objective literals are true", so assuming -rhs[k]
        #    enforces cost <= k. It is built once, only up to the first cost
        #    (ubound), and every bound is then a mere assumption: learnt
        #    clauses survive between calls.
        tot = ITotalizer(lits=enc.objective, ubound=best, top_id=enc.pool.top)
        oracle.s.append_formula(tot.cnf.clauses)
        res.nvars, res.nclauses = tot.top_id, len(enc.clauses) + len(tot.cnf.clauses)
        log.debug("totalizer: ubound=%d +%d vars +%d clauses", best, tot.top_id - enc.pool.top,
                  len(tot.cnf.clauses))

        # 3. Narrow [lo, best]: every cost < lo is proven infeasible, best is
        #    achieved. A SAT answer may jump below k, so best is re-read from
        #    the model rather than set to k.
        lo = 0
        while lo < best:
            k = best - 1 if strategy == "linear" else (lo + best - 1) // 2
            log.debug("search window [%d, %d], asking cost <= %d", lo, best, k)
            out = oracle.solve(bound=k, assumptions=[-tot.rhs[k]])
            if out is None:
                break
            if out:
                best = record(oracle.true_lits())
            else:
                lo = k + 1
        res.lower_bound = lo
        if lo >= best:
            res.status = "optimal"
    finally:
        oracle.s.delete()


def _maxsat(enc: Encoding, solver: str, res: Result) -> None:
    """RC2 MaxSAT: hard clauses + a weight-1 soft clause ``-lit`` per objective literal.

    RC2 has no clean interrupt, so the time limit is the job's wall-clock limit.
    """
    wcnf = WCNF()
    wcnf.extend(enc.clauses)
    for lit in enc.objective:
        wcnf.append([-lit], weight=1)
    t = time.perf_counter()
    with RC2(wcnf, solver=solver) as rc2:
        model = rc2.compute()
        dt = time.perf_counter() - t
        res.calls.append({"bound": None, "result": model is not None, "time": dt})
        log.info("rc2 -> %s in %.3fs", "SAT" if model is not None else "UNSAT", dt)
        if model is None:
            res.status = "unsat"
            return
        true_lits = {lit for lit in model if lit > 0}
        res.status = "optimal"
        res.cost = res.lower_bound = enc.cost(true_lits)
        res.solution = enc.decode(true_lits)
