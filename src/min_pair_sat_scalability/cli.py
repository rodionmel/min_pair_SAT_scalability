"""Command-line entry point.

Debug output: ``-v`` logs progress (every K tried: answer, time, CNF size),
``-vv`` adds debug detail. Logs go to stderr, so JSON on stdout stays
clean. The environment variable ``MIN_PAIR_SAT_LOG`` (e.g. ``DEBUG``) sets
the level when no ``-v`` is given.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from . import __version__, jobs
from .problems import PROBLEMS
from .solve import DEFAULT_SOLVER, STRATEGIES


def setup_logging(verbosity: int) -> None:
    """Configure the package logger: 0 -> WARNING (or $MIN_PAIR_SAT_LOG), 1 -> INFO, 2+ -> DEBUG."""
    if verbosity:
        level = logging.INFO if verbosity == 1 else logging.DEBUG
    else:
        level = getattr(logging, os.environ.get("MIN_PAIR_SAT_LOG", "WARNING").upper(), logging.WARNING)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(relativeCreated)8.0fms %(levelname)-5s %(name)s: %(message)s"))
    logger = logging.getLogger("min_pair_sat_scalability")
    logger.handlers[:] = [handler]  # idempotent when main() is called repeatedly (tests)
    logger.setLevel(level)
    logger.propagate = False


def _cmd_solve(a) -> int:
    """``solve``: run one graph through ``jobs.solve_one`` and print/write the JSON result."""
    job = {"id": None, "problem": a.problem, "graph": a.graph, "strategy": a.strategy,
           "solver": a.solver, "timeout": a.timeout, "k_max": a.k_max, "params": json.loads(a.params)}
    out = jobs.solve_one(job)
    text = json.dumps(out, indent=2)
    if a.output:
        with open(a.output, "w") as f:
            f.write(text)
        print(f"{out['status']} K={out['cost']} -> {a.output}")
    else:
        print(text)
    return 0 if out["status"] != "error" else 1


def _cmd_jobs(a) -> int:
    """``jobs <plan|run|submit|status|collect>``: dispatch to ``jobs.py``."""
    if a.jobs_cmd == "plan":
        print(f"planned {len(jobs.plan(a.spec, a.dir))} jobs in {a.dir}")
    elif a.jobs_cmd == "run":
        if a.index is not None:
            jobs.run_job(a.dir, a.index, force=a.force)
        else:
            jobs.run_local(a.dir, force=a.force)
    elif a.jobs_cmd == "submit":
        jobs.submit(a.dir, mem=a.mem, time_limit=a.time, max_parallel=a.max_parallel,
                    sbatch_args=tuple(a.sbatch_arg), dry_run=a.dry_run, verbosity=a.verbose)
    elif a.jobs_cmd == "status":
        all_jobs, todo = jobs.load_jobs(a.dir), jobs.pending(a.dir)
        print(f"{len(all_jobs) - len(todo)}/{len(all_jobs)} done, {len(todo)} pending")
    elif a.jobs_cmd == "collect":
        jobs.collect(a.dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for ``min-pair-sat`` and its subcommands."""
    p = argparse.ArgumentParser(prog="min-pair-sat", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="-v: progress log, -vv: debug log (stderr); goes before the subcommand")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("solve", help="solve one graph and print the result as JSON")
    s.add_argument("graph", help="graph file (.dimacs/.col/.gr = DIMACS, otherwise edge list)")
    s.add_argument("--problem", default="pair_cover", choices=sorted(PROBLEMS))
    s.add_argument("--strategy", default="linear", choices=STRATEGIES)
    s.add_argument("--solver", default=DEFAULT_SOLVER, help="any PySAT solver name")
    s.add_argument("--timeout", type=float, default=None, help="seconds, whole search")
    s.add_argument("--k-max", type=int, default=None, help="give up (status unsat) above this K")
    s.add_argument("--params", default="{}", help="JSON dict of encoder parameters")
    s.add_argument("-o", "--output", help="write JSON here instead of stdout")
    s.set_defaults(func=_cmd_solve)

    j = sub.add_parser("jobs", help="plan / run / submit / collect a sweep")
    jsub = j.add_subparsers(dest="jobs_cmd", required=True)
    jp = jsub.add_parser("plan", help="expand a sweep spec into DIR/jobs.jsonl")
    jp.add_argument("spec")
    jp.add_argument("dir")
    jr = jsub.add_parser("run", help="run pending jobs locally (or one --index)")
    jr.add_argument("dir")
    jr.add_argument("--index", type=int)
    jr.add_argument("--force", action="store_true", help="re-run finished jobs too")
    js = jsub.add_parser("submit", help="submit pending jobs as a SLURM array")
    js.add_argument("dir")
    js.add_argument("--mem", default="4G")
    js.add_argument("--time", help="SLURM time limit (default: max timeout * 1.2 + 5 min)")
    js.add_argument("--max-parallel", type=int, help="array throttle (%%N)")
    js.add_argument("--sbatch-arg", action="append", default=[],
                    help="extra #SBATCH line, e.g. --sbatch-arg=--partition=long")
    js.add_argument("--dry-run", action="store_true", help="write the script, do not submit")
    for name, help_ in (("status", "count done / pending"), ("collect", "write DIR/results.csv")):
        jsub.add_parser(name, help=help_).add_argument("dir")
    j.set_defaults(func=_cmd_jobs)
    return p


def main(argv=None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    if args.cmd is None:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
