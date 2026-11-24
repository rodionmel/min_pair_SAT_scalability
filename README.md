# min_pair_SAT_scalability

A Python tool and testground for studying the scalability of SAT-based min-pair problems.

Pipeline: **graph file → encoding (CNF + objective) → minimizing SAT/MaxSAT search → JSON result**,
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
| `problems.py` | encoders `graph -> Encoding(clauses, objective, decode)`; register new ones in `PROBLEMS` |
| `solve.py` | minimization: `linear` (SAT-UNSAT descent), `binary`, `maxsat` (RC2) |
| `jobs.py` | sweep planning, local / SLURM execution, result collection |

### The problem: `pair_cover`

Assign every vertex a 0/1 value in each of K stages so that every edge (i, j) shows all four
patterns 00, 01, 10, 11 in some stage; minimize K. This is a binary covering array on the graph.
The encoding fixes stage 0 to all zeros (column flips are a symmetry) and brackets K between
`can2(greedy clique size)` and `can2(greedy colour count)`, where `can2(k)` is the optimal
covering-array size for k columns. `vertex_cover` remains as a simple test problem.

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
In-process timeouts use `interrupt()`, which PySAT's CaDiCaL and Lingeling bindings don't support:
combining them with `--timeout` is an error. For those and for `maxsat`, the SLURM time limit
is the only bound.

## Tests

```bash
pytest
```
