"""Minimal job management: plan a sweep, run locally or as a SLURM array, collect.

Layout of a run directory::

    <dir>/jobs.jsonl        one job per line (written once by ``plan``)
    <dir>/spec.json         copy of the sweep spec
    <dir>/runs/<i>.json     result of job i (written by ``run_job``)
    <dir>/logs/             SLURM stdout/stderr
    <dir>/submit.sbatch     generated array script
    <dir>/results.csv       written by ``collect``

A job whose ``runs/<i>.json`` exists is done; re-running or re-submitting
only touches pending jobs.

Sweep spec (JSON) -- the list fields are expanded as a cartesian product::

    {
      "problem": "pair_cover",
      "graphs": ["graphs/*.dimacs"],      # globs, relative to the spec file
      "strategies": ["linear", "binary"],
      "solvers": ["glucose4"],
      "timeout": 600,
      "k_max": null,                      # optional cap on the K search
      "params": {}                        # passed to the encoder
    }

Instead of (or besides) ``graphs``, a ``generator`` block generates the
graphs into ``<dir>/graphs`` at plan time; each job then also records its
generator parameters under ``gen``::

    "generator": {
      "model": "triangular",              # or "gnp"
      "n": {"start": 4, "stop": 32},      # sqrt(2) progression (optional "factor"), or a list
      "p": [0, 0.1, 0.25],
      "e": [1, 2, 3],                     # triangular only
      "seeds": 5                          # seeds 0..4 per combination
    }
"""

from __future__ import annotations

import csv
import glob
import itertools
import json
import logging
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from .generators import generate, geometric_sizes
from .graph import read_graph, write_graph
from .problems import encode
from .solve import DEFAULT_SOLVER, Result, minimize

log = logging.getLogger(__name__)


def plan(spec_path: str | Path, out_dir: str | Path) -> list[dict]:
    """Expand the sweep spec into ``out_dir/jobs.jsonl`` and return the jobs.

    Graph paths are stored absolute so jobs run from any working directory.
    Refuses to overwrite an existing plan: job ids must stay stable once
    results reference them.
    """
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text())
    out = Path(out_dir).resolve()
    jobs_file = out / "jobs.jsonl"
    if jobs_file.exists():
        raise FileExistsError(f"{jobs_file} exists; use a fresh directory per sweep")
    (out / "runs").mkdir(parents=True, exist_ok=True)

    # (graph path, generator params or None)
    graphs: list[tuple[str, Optional[dict]]] = []
    for pattern in spec.get("graphs", []):
        matches = sorted(glob.glob(str(spec_path.parent / pattern)))
        if not matches:
            raise FileNotFoundError(f"no graphs match {pattern!r}")
        log.debug("plan: %r matched %d graphs", pattern, len(matches))
        graphs += [(m, None) for m in matches]
    if "generator" in spec:
        graphs += _generate_graphs(spec["generator"], out / "graphs")
    if not graphs:
        raise ValueError("spec has neither 'graphs' nor 'generator'")

    jobs = []
    for (g, gen), st, so in itertools.product(graphs, spec.get("strategies", ["linear"]),
                                              spec.get("solvers", [DEFAULT_SOLVER])):
        job = {"id": len(jobs), "problem": spec["problem"], "graph": g, "strategy": st, "solver": so,
               "timeout": spec.get("timeout"), "k_max": spec.get("k_max"), "params": spec.get("params", {})}
        if gen is not None:
            job["gen"] = gen
        jobs.append(job)
    (out / "spec.json").write_text(json.dumps(spec, indent=2))
    jobs_file.write_text("".join(json.dumps(j) + "\n" for j in jobs))
    return jobs


def _generate_graphs(gspec: dict, gdir: Path) -> list[tuple[str, dict]]:
    """Generate every (n, p, e, seed) combination of a ``generator`` block into ``gdir``."""
    model = gspec["model"]
    n_spec = gspec["n"]
    ns = (geometric_sizes(n_spec["start"], n_spec["stop"], n_spec.get("factor", 2 ** 0.5))
          if isinstance(n_spec, dict) else list(n_spec))
    es = gspec.get("e", [1]) if model == "triangular" else [None]
    gdir.mkdir(parents=True, exist_ok=True)
    out = []
    for n, p, e, seed in itertools.product(ns, gspec["p"], es, range(gspec.get("seeds", 1))):
        g, fname = generate(model, n, p, e, seed)
        path = gdir / fname
        write_graph(g, path)
        params = {"model": model, "n": n, "p": p, "seed": seed}
        if e is not None:
            params["e"] = e
        out.append((str(path), params))
    log.info("plan: generated %d %s graphs (n in %s) into %s", len(out), model, ns, gdir)
    return out


def load_jobs(run_dir: str | Path) -> list[dict]:
    """All jobs of a run directory, in id order."""
    return [json.loads(line) for line in (Path(run_dir) / "jobs.jsonl").read_text().splitlines() if line]


def result_path(run_dir: str | Path, job_id: int) -> Path:
    """Where job ``job_id`` writes its result."""
    return Path(run_dir) / "runs" / f"{job_id}.json"


def pending(run_dir: str | Path) -> list[int]:
    """Ids of jobs without a result file."""
    return [j["id"] for j in load_jobs(run_dir) if not result_path(run_dir, j["id"]).exists()]


def solve_one(job: dict) -> dict:
    """Run a single job description and return its result dict; never raises.

    The result is ``Result.to_dict()`` plus graph size, the job itself, host
    and SLURM job id. Failures become ``status="error"`` with the traceback
    attached.
    """
    log.info("job %s: %s %s strategy=%s solver=%s", job.get("id"), job["problem"], job["graph"],
             job["strategy"], job["solver"])
    t0 = time.perf_counter()
    try:
        g = read_graph(job["graph"])
        params = job.get("params", {})
        res = minimize(lambda k: encode(job["problem"], g, k, **params), strategy=job["strategy"],
                       solver=job["solver"], timeout=job.get("timeout"), k_max=job.get("k_max"))
        out = res.to_dict()
        out.update(graph_n=g.n, graph_m=g.m)
    except Exception as e:  # recorded, so one bad job does not kill a sweep
        log.exception("job %s failed", job.get("id"))
        out = Result(status="error", error=f"{type(e).__name__}: {e}").to_dict()
        out["traceback"] = traceback.format_exc()
    out.update(job=job, host=socket.gethostname(), total_time=time.perf_counter() - t0,
               slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    return out


def run_job(run_dir: str | Path, job_id: int, force: bool = False) -> Optional[dict]:
    """Run job ``job_id`` of ``run_dir`` and store its result.

    Returns the result, or None if the job was already done (and not ``force``).
    """
    path = result_path(run_dir, job_id)
    if path.exists() and not force:
        log.info("job %d: already done, skipping", job_id)
        return None
    job = next(j for j in load_jobs(run_dir) if j["id"] == job_id)
    out = solve_one(job)
    # Write-then-rename is atomic, so a job killed mid-write (e.g. by SLURM's
    # time limit) leaves no result file and stays pending.
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(path)
    log.debug("job %d: wrote %s", job_id, path)
    return out


def run_local(run_dir: str | Path, force: bool = False) -> int:
    """Run all pending jobs (all jobs if ``force``) sequentially; returns how many ran."""
    ids = [j["id"] for j in load_jobs(run_dir)] if force else pending(run_dir)
    for i in ids:
        out = run_job(run_dir, i, force=force)
        print(f"job {i}: {out['status']} K={out['cost']} {out['total_time']:.2f}s", flush=True)
    return len(ids)


SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --array={array}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task=1
#SBATCH --output={logs}/%A_%a.out
#SBATCH --error={logs}/%A_%a.err
{extra}
exec {python} -m min_pair_sat_scalability.cli {verbose}jobs run {run_dir} --index "$SLURM_ARRAY_TASK_ID"
"""


def _slurm_time(seconds: float) -> str:
    """Seconds -> SLURM ``D-HH:MM:SS``."""
    s = int(seconds)
    return f"{s // 86400}-{s % 86400 // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _array_spec(ids: list[int]) -> str:
    """Compress sorted ids into a SLURM array spec, e.g. [0,1,2,5,7,8] -> "0-2,5,7-8"."""
    parts = []
    start = prev = ids[0]
    for i in ids[1:] + [None]:
        if i is not None and i == prev + 1:
            prev = i
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        if i is not None:
            start = prev = i
    return ",".join(parts)


def submit(run_dir: str | Path, mem: str = "4G", time_limit: Optional[str] = None,
           max_parallel: Optional[int] = None, sbatch_args: tuple[str, ...] = (),
           dry_run: bool = False, verbosity: int = 0) -> Optional[Path]:
    """Write ``submit.sbatch`` covering all pending jobs and (unless ``dry_run``) sbatch it.

    Each array task runs ``jobs run --index $SLURM_ARRAY_TASK_ID`` with this
    interpreter (the venv), so the run directory must be on a filesystem the
    nodes share. ``verbosity`` adds that many ``-v`` flags to the task command.
    Returns the script path, or None if nothing is pending.
    """
    run_dir = Path(run_dir).resolve()
    ids = pending(run_dir)
    if not ids:
        print("nothing pending")
        return None
    if time_limit is None:  # solver timeout + margin for encoding and I/O
        timeouts = [j.get("timeout") or 3600 for j in load_jobs(run_dir)]
        time_limit = _slurm_time(max(timeouts) * 1.2 + 300)
    array = _array_spec(ids)
    if max_parallel:
        array += f"%{max_parallel}"
    (run_dir / "logs").mkdir(exist_ok=True)
    script = run_dir / "submit.sbatch"
    script.write_text(SBATCH_TEMPLATE.format(
        name=run_dir.name, array=array, time=time_limit, mem=mem, logs=run_dir / "logs",
        extra="\n".join(f"#SBATCH {a}" for a in sbatch_args), python=sys.executable,
        verbose="-v " * verbosity, run_dir=run_dir))
    log.debug("submit: array=%s time=%s mem=%s", array, time_limit, mem)
    if dry_run:
        print(f"wrote {script} ({len(ids)} jobs); not submitted")
    else:
        out = subprocess.run(["sbatch", str(script)], check=True, capture_output=True, text=True)
        print(out.stdout.strip())
    return script


CSV_FIELDS = ["id", "graph", "model", "n", "p", "e", "seed",
              "problem", "strategy", "solver", "timeout", "graph_n", "graph_m",
              "k_max", "status", "cost", "lower_bound", "nvars", "nclauses", "n_calls",
              "solve_time", "encode_time", "wall_time", "total_time", "host", "error"]


def collect(run_dir: str | Path) -> Path:
    """Summarize finished jobs into ``results.csv`` (one row per job) and print status counts."""
    run_dir = Path(run_dir)
    rows, counts = [], {}
    for job in load_jobs(run_dir):
        path = result_path(run_dir, job["id"])
        if not path.exists():
            counts["pending"] = counts.get("pending", 0) + 1
            continue
        r = json.loads(path.read_text())
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        # Flatten: job fields, result fields, and per-call times summed over all K tried.
        calls = r.get("calls", [])
        row = {**job, **job.get("gen", {}), **{k: r.get(k) for k in CSV_FIELDS if k in r},
               "n_calls": len(calls),
               "solve_time": sum(c["time"] for c in calls),
               "encode_time": sum(c.get("encode_time", 0) for c in calls)}
        rows.append({k: row.get(k) for k in CSV_FIELDS})
    out = run_dir / "results.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(" ".join(f"{k}={v}" for k, v in sorted(counts.items())), f"-> {out}")
    return out
