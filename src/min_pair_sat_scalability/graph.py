"""Graph model and file I/O.

Two input formats are supported:

* DIMACS graph format (``.dimacs``, ``.col``, ``.gr``) -- the primary format::

      c optional comment
      p edge <n_vertices> <n_edges>
      e <u> <v>            # 1-based vertex ids

* Plain edge list (any other extension): one ``u v`` pair per line, 0-based,
  ``#`` comments. The vertex count is max id + 1, so isolated trailing
  vertices are lost -- use DIMACS when that matters.

Internally vertices are always ``0 .. n-1``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)

DIMACS_SUFFIXES = {".dimacs", ".col", ".gr"}


@dataclass(frozen=True)
class Graph:
    """Immutable graph on vertices ``0 .. n-1``.

    Build with ``Graph.from_edges`` to get duplicate-free edges; the plain
    constructor only checks vertex ranges. ``name`` is informational and
    ignored by equality.
    """

    n: int
    edges: tuple[tuple[int, int], ...]
    directed: bool = False
    name: str = field(default="", compare=False)

    def __post_init__(self):
        for u, v in self.edges:
            if not (0 <= u < self.n and 0 <= v < self.n):
                raise ValueError(f"edge ({u}, {v}) out of range for n={self.n}")

    @classmethod
    def from_edges(cls, n: int, edges: Iterable[tuple[int, int]], directed: bool = False, name: str = "") -> "Graph":
        """Build a graph, dropping duplicate edges (and reversed duplicates if undirected).

        Undirected edges are normalized to ``(min, max)``; first-seen order is kept.
        """
        seen = set()
        out = []
        for u, v in edges:
            key = (u, v) if directed else (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                out.append(key)
        return cls(n=n, edges=tuple(out), directed=directed, name=name)

    @property
    def m(self) -> int:
        """Number of edges."""
        return len(self.edges)

    def adjacency(self) -> list[list[int]]:
        """Neighbour lists (out-neighbours if directed)."""
        adj: list[list[int]] = [[] for _ in range(self.n)]
        for u, v in self.edges:
            adj[u].append(v)
            if not self.directed and u != v:
                adj[v].append(u)
        return adj


def read_graph(path: str | Path, directed: bool = False) -> Graph:
    """Read a graph file; the format is chosen by extension (see module docstring)."""
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in DIMACS_SUFFIXES:
        g = parse_dimacs(text, directed=directed, name=path.stem)
    else:
        g = parse_edge_list(text, directed=directed, name=path.stem)
    log.debug("read %s: n=%d m=%d", path, g.n, g.m)
    return g


def parse_dimacs(text: str, directed: bool = False, name: str = "") -> Graph:
    """Parse DIMACS ``p edge`` text. Accepts ``e`` and ``a`` (arc) lines; ids are 1-based."""
    n = None
    edges = []
    for lineno, line in enumerate(text.splitlines(), 1):
        tok = line.split()
        if not tok or tok[0] == "c":
            continue
        if tok[0] == "p":
            n = int(tok[2])
        elif tok[0] in ("e", "a"):
            if n is None:
                raise ValueError(f"line {lineno}: edge before 'p' header")
            edges.append((int(tok[1]) - 1, int(tok[2]) - 1))
        else:
            raise ValueError(f"line {lineno}: unexpected line {line!r}")
    if n is None:
        raise ValueError("missing 'p edge <n> <m>' header")
    return Graph.from_edges(n, edges, directed=directed, name=name)


def parse_edge_list(text: str, directed: bool = False, name: str = "") -> Graph:
    """Parse ``u v`` lines (0-based, ``#`` comments); extra columns are ignored."""
    edges = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            u, v = line.split()[:2]
            edges.append((int(u), int(v)))
    n = 1 + max((max(e) for e in edges), default=-1)
    return Graph.from_edges(n, edges, directed=directed, name=name)


def to_dimacs(g: Graph) -> str:
    """Serialize to DIMACS text (``name`` becomes a ``c`` comment line)."""
    lines = [f"c {g.name}"] if g.name else []
    lines.append(f"p edge {g.n} {g.m}")
    lines += [f"e {u + 1} {v + 1}" for u, v in g.edges]
    return "\n".join(lines) + "\n"


def write_graph(g: Graph, path: str | Path) -> None:
    """Write ``g`` in DIMACS format."""
    Path(path).write_text(to_dimacs(g))
