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
| `universal.py` | precomputed optimal universal sets (table in `src/.../data/universal_sets.json`) |
| `generators.py` | random graphs: uniform G(n, p) and thinned power of a triangular lattice |

### The problem: `pair_cover`

Assign every vertex a 0/1 value in each of K stages so that every edge (i, j) shows all four
patterns 00, 01, 10, 11 in some stage; minimize K.

The encoding is the straightforward one: variables `x[v][s]` for each vertex and stage, and for
each (edge, pattern, stage) a Tseitin witness `y -> (x[i][s] = a and x[j][s] = b)`, with one
clause `OR_s y` per (edge, pattern). Each K is a separate SAT instance; no symmetry breaking or
bounds are used. Each result lists every K tried with its answer, solve time and CNF size.

### Option: symmetry breaking (`"params": {"symmetry": ...}`)

Off by default. `"zero"` fixes stage 0 to all zeros (flipping a vertex's values keeps its edges
covered); `"zero_lex"` also orders stages 1..K-1 lexicographically (stages are interchangeable).
For complete graphs only, `col_lex` also orders the columns strictly (`universal --symmetry full`).

### Option: colouring route (`--problem coloring`)

An alternative to the direct encoding that gives an upper bound on the number of stages:
1. Find a clique with the simplest greedy algorithm (scan by decreasing degree, keep a vertex if it
   is adjacent to all kept ones).
2. Fix the clique vertices to colours 0, 1, ... (symmetry breaking) and SAT-search the least
   number of colours chi, starting at the clique size.
3. Give colour j the j-th column of the precomputed optimal universal set of size chi.

The result reports `chi` (the minimized value, `cost`) and the resulting `stages`/`rows`. This is
not always optimal for pair cover: a 10-vertex graph has chi = 5 (so 6 stages) but a 5-stage pair
cover. The universal sets are pair covers of complete graphs, computed with the basic encoding:

```bash
min-pair-sat universal --start 1 --stop 75 --symmetry full    # or omit --stop: run until killed
min-pair-sat universal --show
min-pair-sat solve graph.dimacs --problem coloring
```

If chi exceeds the table, the colouring is still reported but `stages` is null; after extending
the table, `min-pair-sat jobs fill-stages <run dir>` fills them in from the saved colourings.
The packaged table covers c = 1 … 75 (c ≤ 10 basic encoding, c ≥ 11 with `--symmetry full`).

## Usage

Single run (`-v` logs progress, `-vv` debug detail, both on stderr):

```bash
min-pair-sat -v solve examples/graphs/gnp_30.dimacs --strategy linear --timeout 60
```

Generate graphs (every combination of the listed values, `--count` seeds each; DIMACS files
named after their parameters, e.g. `tri_n8_p0.1_e2_s0.dimacs`):

```bash
# uniform random: n vertices, each edge included with probability p
min-pair-sat gen gnp --n 20 40 80 --p 0.1 0.3 --count 5 --out-dir graphs/gnp
# triangular lattice with n vertices per side (n(n+1)/2 vertices), raised to power e
# (e=2: also join vertices at distance 2, ...), then each edge EXcluded with probability p
min-pair-sat gen triangular --n 8 16 --p 0.1 --e 1 2 3 --count 5 --out-dir graphs/tri
```

The power is built on the full lattice; exclusion is applied to the power graph.

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

### Experiments

`experiments/` holds sweep specs whose `generator` block makes `jobs plan` generate the graphs
into `<run dir>/graphs`. Sizes grow by a factor of sqrt(2); each job records its generator
parameters, which become columns (`model`, `n`, `p`, `e`, `seed`) of `results.csv`.

| spec | grid | jobs |
|---|---|---|
| `gnp_grid.json` | p ∈ {0.1, 0.25, 0.5, 0.75, 0.9}, n = 8 … 128, 5 seeds | 225 |
| `tri_grid.json` | p ∈ {0, 0.1, 0.25, 0.5, 0.75}, e ∈ 1…5, n = 4 … 32, 5 seeds | 875 |
| `gnp_small.json`, `tri_small.json` | reduced versions for trial runs (30 s timeout) | 24, 36 |

```bash
min-pair-sat jobs plan experiments/tri_small.json results/tri_small
min-pair-sat -v jobs submit results/tri_small --mem 2G
min-pair-sat jobs collect results/tri_small
```

### Plots

`pip install -e ".[plot]"`, collect the sweeps, then `python analysis/plot_results.py` writes to
`results/figures/`: `overview_<model>.png` (time heatmap, both methods), `scaling_<model>.png`
(time vs vertices per p; lines break at unsolved sizes, timeouts as hollow markers),
`methods.png` (pair_cover vs coloring per graph) and `anatomy.png` (time per K tried).

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
