# min_pair_SAT_scalability

A Python tool and testground for studying the scalability of SAT-based min-pair problems.

Pipeline: **graph file → CNF for a given K → search for the least satisfiable K → JSON result**,
plus a small job layer for sweeps run locally or as SLURM arrays.

## Setup

Use the system Python (`~/localPython` 3.13 lacks `ssl`/`bz2`, which pip and PySAT need):

```bash
/usr/bin/python3.9 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Layout

| module | role |
|---|---|
| `graph.py` | `Graph` model; read DIMACS (`p edge n m` / `e u v`, 1-based) or plain edge lists |
| `problems.py` | encoders `(graph, K) -> Encoding(clauses, decode)`; register new ones in `PROBLEMS` |
| `solve.py` | search over K: `linear` (K = 0, 1, 2, ...) or `binary` (doubling, then bisection) |
| `jobs.py` | sweep planning, local / SLURM execution, result collection |

### The problem: `pair_cover`

Assign every vertex a 0/1 value in each of K stages so that every edge (i, j) shows all four
patterns 00, 01, 10, 11 in some stage; minimize K.

The encoding is the straightforward one: variables `x[v][s]` for each vertex and stage, and for
each (edge, pattern, stage) a Tseitin witness `y -> (x[i][s] = a and x[j][s] = b)`, with one
clause `OR_s y` per (edge, pattern). Each K is a separate SAT instance; no symmetry breaking or
bounds are used. Each result lists every K tried with its answer, solve time and CNF size.

## Usage

Single run (`-v` logs progress, `-vv` debug detail, both on stderr):

```bash
min-pair-sat -v solve examples/graphs/gnp_30.dimacs --strategy linear --timeout 60
```

Sweep (`examples/sweep.json` expands graphs × strategies × solvers):

```bash
min-pair-sat jobs plan examples/sweep.json results/demo
min-pair-sat jobs run results/demo              # locally, or:
min-pair-sat jobs submit results/demo --sbatch-arg=--partition=short --max-parallel 50
min-pair-sat jobs status results/demo
min-pair-sat jobs collect results/demo          # -> results/demo/results.csv
```

Each job writes `runs/<id>.json` (status, cost, lower bound, solution, encoding size, per-call
SAT timings). Finished jobs are skipped, so `run`/`submit` can simply be repeated after failures.

### Solvers and timeouts

Any PySAT solver name works (`glucose4` default, `cadical195`, `minisat22`, `maplechrono`, ...).
`--timeout` bounds the whole search and works by interrupting the solver, which PySAT's CaDiCaL
and Lingeling bindings don't support: combining them with `--timeout` is an error, and the SLURM
time limit is then the only bound. `--k-max` caps the search (status `unsat` above it).

## Tests

```bash
pytest
```

## License

MIT, see [LICENSE](LICENSE).
