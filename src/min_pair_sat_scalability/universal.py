"""Precomputed optimal universal sets.

A universal set of size c with K stages is a family of c columns (0/1 vectors
of length K) such that every two columns show all four patterns 00, 01, 10,
11 in some stage. That is exactly a pair cover of the complete graph K_c, so
the optimal (least K) set for each c is computed with the basic pair_cover
encoding and the same per-K search.

The table is a JSON file mapping ``str(c)`` to
``{"c": c, "k": K, "status": "optimal", "symmetry": ..., "columns": ["0110...", ...]}``
(one length-K string per column). It is rewritten after every new entry, so a
run can be killed at any time without losing finished entries.
"""

from __future__ import annotations

import functools
import itertools
import json
import logging
from pathlib import Path
from typing import Optional

from .graph import Graph
from .problems import check_pair_cover, encode_pair_cover
from .solve import DEFAULT_SOLVER, minimize

log = logging.getLogger(__name__)

DEFAULT_TABLE = Path(__file__).parent / "data" / "universal_sets.json"

# Symmetry-breaking settings for the pair_cover encoding of K_c; "full" adds
# strictly ordered columns, which is sound because all vertices of K_c are
# interchangeable.
SYMMETRY = {
    "none": {},
    "zero": {"symmetry": "zero"},
    "zero_lex": {"symmetry": "zero_lex"},
    "full": {"symmetry": "zero_lex", "col_lex": True},
}


def compute(c: int, solver: str = DEFAULT_SOLVER, timeout: Optional[float] = None,
            k_min: int = 0, symmetry: str = "none") -> Optional[dict]:
    """Optimal universal set of size ``c``, or None if the search timed out.

    ``k_min`` starts the K search there; pass a K known to be infeasible-below
    (e.g. the optimum for c-1: a universal set of size c contains one of size c-1).
    ``symmetry`` is a key of ``SYMMETRY``.
    """
    params = SYMMETRY[symmetry]
    clique = Graph.from_edges(c, itertools.combinations(range(c), 2), name=f"K{c}")
    res = minimize(lambda k: encode_pair_cover(clique, k, **params), solver=solver, timeout=timeout,
                   k_min=k_min)
    if res.status != "optimal":
        log.warning("universal set c=%d: %s (K >= %d)", c, res.status, res.lower_bound)
        return None
    rows = res.solution["rows"]
    assert not check_pair_cover(clique, rows)
    columns = ["".join(row[v] for row in rows) for v in range(c)]
    log.info("universal set c=%d: K=%d (%.2fs, symmetry=%s)", c, res.cost, res.wall_time, symmetry)
    return {"c": c, "k": res.cost, "status": "optimal", "symmetry": symmetry, "columns": columns}


def load_table(path: str | Path = DEFAULT_TABLE) -> dict[int, dict]:
    """Read a table file; a missing file is an empty table."""
    path = Path(path)
    if not path.exists():
        return {}
    return {int(c): entry for c, entry in json.loads(path.read_text()).items()}


def save_table(table: dict[int, dict], path: str | Path = DEFAULT_TABLE) -> None:
    """Write the table atomically (write-then-rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({str(c): table[c] for c in sorted(table)}, indent=1) + "\n")
    tmp.replace(path)
    _cached_table.cache_clear()


def precompute(start: int, stop: Optional[int] = None, path: str | Path = DEFAULT_TABLE,
               solver: str = DEFAULT_SOLVER, timeout: Optional[float] = None,
               symmetry: str = "none") -> dict[int, dict]:
    """Compute entries for c = start, start+1, ... up to ``stop`` (forever if None).

    Existing entries are kept and skipped. Stops at the first c that times out,
    since larger c would only be harder.
    """
    table = load_table(path)
    c = start
    while stop is None or c <= stop:
        if c not in table:
            k_min = max((e["k"] for cc, e in table.items() if cc < c), default=0)
            entry = compute(c, solver=solver, timeout=timeout, k_min=k_min, symmetry=symmetry)
            if entry is None:
                break
            table[c] = entry
            save_table(table, path)
            print(f"c={c}: K={entry['k']}", flush=True)
        c += 1
    return table


@functools.lru_cache(maxsize=None)
def _cached_table(path: str) -> dict[int, dict]:
    return load_table(path)


def stages_from_coloring(colors: list[int], chi: int, path: str | Path = DEFAULT_TABLE) -> Optional[dict]:
    """``{"stages", "rows"}`` for a colouring with ``chi`` colours, or None if
    the universal set of size ``chi`` is not in the table."""
    columns = columns_for(chi, path) if chi else []
    if columns is None:
        return None
    stages = len(columns[0]) if columns else 0
    rows = ["".join(columns[c][s] for c in colors) for s in range(stages)]
    return {"stages": stages, "rows": rows}


def columns_for(c: int, path: str | Path = DEFAULT_TABLE) -> Optional[list[str]]:
    """The precomputed columns for size ``c``, or None if not in the table."""
    entry = _cached_table(str(path)).get(c)
    return None if entry is None else entry["columns"]
