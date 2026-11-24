import csv
import itertools
import json
import random

import pytest

from min_pair_sat_scalability import jobs
from min_pair_sat_scalability.cli import main
from min_pair_sat_scalability.graph import Graph, parse_dimacs, parse_edge_list, read_graph, write_graph


def random_graph(n, p, seed):
    rng = random.Random(seed)
    return Graph.from_edges(n, [(u, v) for u, v in itertools.combinations(range(n), 2) if rng.random() < p])


# --- graph I/O --------------------------------------------------------------

def test_dimacs_roundtrip(tmp_path):
    g = Graph.from_edges(5, [(0, 1), (1, 2), (3, 4)], name="t")
    path = tmp_path / "t.dimacs"
    write_graph(g, path)
    assert read_graph(path) == g


def test_dimacs_keeps_isolated_vertices():
    g = parse_dimacs("c x\np edge 4 1\ne 1 2\n")
    assert (g.n, g.edges) == (4, ((0, 1),))


def test_edge_list_dedupes_undirected():
    g = parse_edge_list("# comment\n0 1\n1 0\n1 2  # trailing\n")
    assert (g.n, g.edges) == (3, ((0, 1), (1, 2)))


def test_out_of_range_edge_rejected():
    with pytest.raises(ValueError):
        parse_dimacs("p edge 2 1\ne 1 3\n")


# --- jobs -------------------------------------------------------------------

def make_sweep(tmp_path):
    gdir = tmp_path / "graphs"
    gdir.mkdir()
    for s in range(2):
        write_graph(random_graph(8, 0.4, s), gdir / f"g{s}.dimacs")
    spec = tmp_path / "sweep.json"
    spec.write_text(json.dumps({"problem": "pair_cover", "graphs": ["graphs/*.dimacs"],
                                "strategies": ["linear", "binary"], "timeout": 30}))
    return spec


def test_sweep_plan_run_collect(tmp_path):
    run_dir = tmp_path / "run"
    planned = jobs.plan(make_sweep(tmp_path), run_dir)
    assert len(planned) == 4
    assert jobs.run_local(run_dir) == 4
    assert jobs.pending(run_dir) == []
    assert jobs.run_local(run_dir) == 0  # resumable: nothing left to do
    rows = list(csv.DictReader(open(jobs.collect(run_dir))))
    assert {r["status"] for r in rows} == {"optimal"}
    by_graph = {}
    for r in rows:
        by_graph.setdefault(r["graph"], set()).add(r["cost"])
    assert all(len(costs) == 1 for costs in by_graph.values())  # strategies agree


def test_bad_job_is_recorded_not_raised(tmp_path):
    out = jobs.solve_one({"id": 0, "problem": "nope", "graph": str(tmp_path / "missing.dimacs"),
                          "strategy": "linear", "solver": "glucose4"})
    assert out["status"] == "error" and out["error"]


def test_submit_dry_run_only_pending(tmp_path):
    run_dir = tmp_path / "run"
    jobs.plan(make_sweep(tmp_path), run_dir)
    jobs.run_job(run_dir, 1)
    script = jobs.submit(run_dir, max_parallel=2, sbatch_args=("--partition=x",), dry_run=True)
    text = script.read_text()
    assert "#SBATCH --array=0,2-3%2" in text
    assert "#SBATCH --partition=x" in text
    assert "jobs run" in text and "$SLURM_ARRAY_TASK_ID" in text


def test_submit_propagates_verbosity(tmp_path):
    run_dir = tmp_path / "run"
    jobs.plan(make_sweep(tmp_path), run_dir)
    text = jobs.submit(run_dir, dry_run=True, verbosity=2).read_text()
    assert "cli -v -v jobs run" in text


# --- CLI --------------------------------------------------------------------

def test_cli_solve(tmp_path, capsys):
    path = tmp_path / "g.txt"
    path.write_text("0 1\n1 2\n2 0\n")
    assert main(["solve", str(path), "--strategy", "binary"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert (out["status"], out["cost"]) == ("optimal", 4)  # triangle needs 4 stages


def test_cli_k_max(tmp_path, capsys):
    path = tmp_path / "g.txt"
    path.write_text("0 1\n1 2\n2 0\n")
    assert main(["solve", str(path), "--k-max", "3"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "unsat"


def test_verbose_logs_go_to_stderr(tmp_path, capsys):
    path = tmp_path / "g.txt"
    path.write_text("0 1\n1 2\n2 0\n")
    assert main(["-vv", "solve", str(path)]) == 0
    cap = capsys.readouterr()
    json.loads(cap.out)  # stdout is still pure JSON
    assert "K=4 -> SAT" in cap.err and "pair_cover:" in cap.err
    main([])  # reset logging to quiet for later tests


def test_plan_with_generator(tmp_path):
    spec = tmp_path / "grid.json"
    spec.write_text(json.dumps({"problem": "pair_cover", "timeout": 30, "generator": {
        "model": "triangular", "n": {"start": 3, "stop": 4}, "p": [0, 0.5], "e": [1, 2], "seeds": 2}}))
    run_dir = tmp_path / "run"
    planned = jobs.plan(spec, run_dir)
    assert len(planned) == 2 * 2 * 2 * 2  # n in {3, 4} x p x e x seeds
    assert len(list((run_dir / "graphs").iterdir())) == 16
    assert planned[-1]["gen"] == {"model": "triangular", "n": 4, "p": 0.5, "seed": 1, "e": 2}
    jobs.run_local(run_dir)
    rows = list(csv.DictReader(open(jobs.collect(run_dir))))
    assert rows[-1]["model"] == "triangular" and rows[-1]["e"] == "2" and rows[-1]["seed"] == "1"


def test_array_spec_compression():
    assert jobs._array_spec([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"
    assert jobs._array_spec([3]) == "3"
