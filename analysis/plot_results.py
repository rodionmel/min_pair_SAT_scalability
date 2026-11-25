"""Plots for the pair_cover vs coloring grid runs.

Reads ``results/<sweep>/results.csv`` (run ``min-pair-sat jobs collect`` first)
and writes PNG figures:

1. overview_<model>.png   heatmap of total time per (parameter point, n), both methods
2. scaling_<model>.png    total time vs number of vertices, one line per p
3. methods.png            pair_cover time vs coloring time, one point per graph
4. anatomy.png            solve time of each K tried, for the slowest solved graphs

Timeouts are censored data ("at least the timeout"), so they are drawn as hollow
markers on the timeout line, never as ordinary points. Jobs still running are
shown as such in the heatmaps and left out elsewhere.

Usage:  python analysis/plot_results.py [--results results] [--out results/figures]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Colours from the reference data-viz palette (validated slots / ramp steps).
SERIES = {"gnp": "#2a78d6", "triangular": "#eb6834"}  # categorical slots 1, 2
SAT_UNSAT = {True: "#2a78d6", False: "#eb6834"}
P_RAMP = ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # blue 250..700, ordinal
P_MARKERS = ["o", "s", "D", "^", "v"]
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TIMEOUT_FILL, RUNNING_FILL = "#383835", "#f0efec"
METHODS = [("pair_cover", ""), ("coloring", "_coloring")]
MODEL_DIR = {"gnp": "gnp_grid_1seed", "triangular": "tri_grid_1seed"}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.dpi": 160, "savefig.bbox": "tight",
})


# --- data -------------------------------------------------------------------

def load(results: Path, model: str, method_suffix: str) -> tuple[dict, list[dict]]:
    """(rows keyed by (n, p, e), all planned jobs) for one sweep."""
    d = results / (MODEL_DIR[model] + method_suffix)
    rows = {}
    if (d / "results.csv").exists():
        for r in csv.DictReader(open(d / "results.csv")):
            rows[(int(r["n"]), float(r["p"]), int(r["e"]) if r["e"] else 1)] = r
    jobs = [json.loads(line) for line in open(d / "jobs.jsonl")]
    return rows, jobs


def points(jobs: list[dict]) -> list[tuple[int, float, int]]:
    return sorted({(j["gen"]["n"], j["gen"]["p"], j["gen"].get("e", 1)) for j in jobs})


def time_of(r: dict) -> float:
    return float(r["total_time"])


def is_timeout(r: dict) -> bool:
    return r["status"] != "optimal"


# --- 1. overview heatmap ------------------------------------------------------

def overview(results: Path, out: Path, model: str) -> None:
    data = [load(results, model, sfx) for _, sfx in METHODS]
    pts = points(data[0][1])
    ns = sorted({n for n, _, _ in pts})
    params = sorted({(e, p) for _, p, e in pts})  # rows
    tmax = max([1800.0] + [time_of(r) for rows, _ in data for r in rows.values()])
    bounds = [0.01, 0.1, 1, 10, 60, 300, 1800, tmax + 1]
    cmap = ListedColormap(SEQ_RAMP)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(9, 1.0 + 0.28 * len(params)), sharey=True)
    for ax, (method, _), (rows, _) in zip(axes, METHODS, data):
        ax.grid(False)
        for yi, (e, p) in enumerate(params):
            for xi, n in enumerate(ns):
                r = rows.get((n, p, e))
                if r is None:
                    face, hatch = RUNNING_FILL, "...."
                elif is_timeout(r):
                    face, hatch = TIMEOUT_FILL, None
                else:
                    face, hatch = cmap(norm(max(time_of(r), 0.011))), None
                ax.add_patch(plt.Rectangle((xi, yi), 0.92, 0.88, facecolor=face, hatch=hatch,
                                           edgecolor="#b0afa8" if hatch else "none", linewidth=0))
        ax.set_xlim(-0.1, len(ns))
        ax.set_ylim(len(params), -0.15)
        ax.set_xticks([i + 0.46 for i in range(len(ns))], [str(n) for n in ns])
        ax.set_xlabel("n" + (" (vertices per side)" if model == "triangular" else " (vertices)"))
        ax.set_title(method, color=TEXT, loc="left", fontweight="bold")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    labels = [f"e={e}  p={p}" if model == "triangular" else f"p={p}" for e, p in params]
    axes[0].set_yticks([i + 0.44 for i in range(len(params))], labels)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02, ticks=bounds[:-1])
    cb.ax.set_yticklabels(["0.01 s", "0.1 s", "1 s", "10 s", "1 min", "5 min", "30 min"])
    cb.set_label("total time", color=TEXT)
    fig.legend(handles=[Patch(facecolor=TIMEOUT_FILL, label="timeout (≥ 30 min)"),
                        Patch(facecolor=RUNNING_FILL, hatch="....", edgecolor="#b0afa8",
                              label="still running")],
               loc="lower left", bbox_to_anchor=(0.0, -0.06 if len(params) > 6 else -0.2), ncol=2)
    height = fig.get_figheight()
    fig.suptitle(f"{model}: time to optimal answer", x=0.0, y=1.0 + 0.45 / height, ha="left",
                 color=TEXT, fontweight="bold")
    fig.savefig(out / f"overview_{model}.png")
    plt.close(fig)


# --- 2. scaling curves -------------------------------------------------------

def scaling(results: Path, out: Path, model: str) -> None:
    data = [load(results, model, sfx) for _, sfx in METHODS]
    pts = points(data[0][1])
    ps = sorted({p for _, p, _ in pts})
    es = sorted({e for _, _, e in pts})
    timeout = float(data[0][1][0]["timeout"])
    fig, axes = plt.subplots(len(es), 2, figsize=(8.5, 2.6 * len(es)), sharex=True, sharey=True,
                             squeeze=False)
    for row, e in enumerate(es):
        for col, ((method, _), (rows, _)) in enumerate(zip(METHODS, data)):
            ax = axes[row][col]
            for pi, p in enumerate(ps):
                line, cens = [], []  # line: (vertices, time or NaN); NaN breaks the line
                for n in sorted({nn for nn, pp, ee in pts if pp == p and ee == e}):
                    r = rows.get((n, p, e))
                    v = n * (n + 1) // 2 if model == "triangular" else n
                    if r is None:  # still running: break the line, draw nothing
                        line.append((v, float("nan")))
                    elif is_timeout(r):
                        cens.append(v)
                        line.append((v, float("nan")))  # never bridge an unsolved size
                    else:
                        line.append((v, time_of(r)))
                color, marker = P_RAMP[pi], P_MARKERS[pi]
                if line:
                    ax.plot(*zip(*line), color=color, marker=marker, markersize=4.5, linewidth=1.6,
                            label=f"p={p}")
                if cens:
                    ax.plot(cens, [timeout] * len(cens), linestyle="none",
                            marker=marker, markersize=6, markerfacecolor="white", markeredgecolor=color,
                            markeredgewidth=1.4)
            ax.axhline(timeout, color=MUTED, linestyle="--", linewidth=0.8)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            if row == 0:
                ax.set_title(method, color=TEXT, loc="left", fontweight="bold")
            if col == 0:
                ax.set_ylabel(("" if model == "gnp" else f"e={e}\n") + "total time [s]")
            if row == len(es) - 1:
                ax.set_xlabel("vertices")
    handles = [plt.Line2D([], [], color=P_RAMP[i], marker=P_MARKERS[i], markersize=4.5, linewidth=1.6)
               for i in range(len(ps))]
    labels = [f"p={p}" for p in ps]
    handles.append(plt.Line2D([], [], linestyle="none", marker="o", markersize=6, markerfacecolor="white",
                              markeredgecolor=MUTED, markeredgewidth=1.4))
    labels.append("timeout (≥ 30 min, on dashed line)")
    h = fig.get_figheight()
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / h))
    fig.legend(handles, labels, loc="upper left", ncol=len(labels), fontsize=8,
               bbox_to_anchor=(0.0, 1.0 - 0.3 / h))  # own line below the title
    fig.suptitle(f"{model}: scaling with size", x=0.0, y=1.0, ha="left", va="top", color=TEXT,
                 fontweight="bold")
    fig.savefig(out / f"scaling_{model}.png")
    plt.close(fig)


# --- 3. method comparison ---------------------------------------------------

def methods(results: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    timeout = 1800.0
    for model, marker in (("gnp", "o"), ("triangular", "^")):
        (pc, jobs), (co, _) = (load(results, model, sfx) for _, sfx in METHODS)
        timeout = float(jobs[0]["timeout"])
        solved, cens = [], []
        for key, a in pc.items():
            b = co.get(key)
            if b is None:  # coloring still running
                continue
            x = timeout if is_timeout(a) else time_of(a)
            y = timeout if is_timeout(b) else time_of(b)
            (cens if (is_timeout(a) or is_timeout(b)) else solved).append((x, y))
        c = SERIES[model]
        if solved:
            ax.scatter(*zip(*solved), s=22, color=c, marker=marker, edgecolor="white", linewidth=0.6,
                       label=f"{model} (both solved)", zorder=3)
        if cens:
            ax.scatter(*zip(*cens), s=30, facecolor="white", edgecolor=c, marker=marker, linewidth=1.2,
                       label=f"{model} (timeout on a side)", zorder=3)
    lo, hi = 0.01, timeout * 1.6
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=0.8)
    ax.text(0.015, 0.03, "y = x", color=MUTED, fontsize=8, rotation=45)
    ax.axhline(timeout, color=MUTED, linestyle="--", linewidth=0.8)
    ax.axvline(timeout, color=MUTED, linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("pair_cover total time [s]")
    ax.set_ylabel("coloring total time [s]")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Per graph: below the diagonal = coloring faster", loc="left", color=TEXT,
                 fontweight="bold")
    fig.savefig(out / "methods.png")
    plt.close(fig)


# --- 4. anatomy of a solve ---------------------------------------------------

def anatomy(results: Path, out: Path, count: int = 4) -> None:
    runs = []
    for model in ("gnp", "triangular"):
        d = results / MODEL_DIR[model]
        for f in (d / "runs").glob("*.json"):
            r = json.loads(f.read_text())
            if r["status"] == "optimal":
                runs.append((r["wall_time"], model, r))
    runs.sort(key=lambda x: -x[0])
    chosen = runs[:count]
    fig, axes = plt.subplots(1, len(chosen), figsize=(2.4 * len(chosen), 2.6), sharey=True)
    for ax, (_, model, r) in zip(axes, chosen):
        g = r["job"]["gen"]
        ks = [c["k"] for c in r["calls"]]
        ts = [max(c["time"], 1e-3) for c in r["calls"]]
        ax.bar(ks, ts, width=0.7, color=[SAT_UNSAT[c["result"]] for c in r["calls"]])
        ax.set_yscale("log")
        ax.set_xticks(ks)
        ax.set_xlabel("K tried")
        name = f"gnp n={g['n']} p={g['p']}" if model == "gnp" else f"tri n={g['n']} p={g['p']} e={g['e']}"
        ax.set_title(name, fontsize=8, color=TEXT, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("solve time [s]")
    fig.legend(handles=[Patch(color=SAT_UNSAT[False], label="UNSAT (K too small)"),
                        Patch(color=SAT_UNSAT[True], label="SAT (optimal K)")],
               loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.08))
    fig.suptitle("pair_cover: where the time goes (slowest solved graphs)", x=0.0, ha="left",
                 color=TEXT, fontweight="bold", y=1.08)
    fig.tight_layout()
    fig.savefig(out / "anatomy.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results", type=Path)
    ap.add_argument("--out", default="results/figures", type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for model in ("gnp", "triangular"):
        overview(a.results, a.out, model)
        scaling(a.results, a.out, model)
    methods(a.results, a.out)
    anatomy(a.results, a.out)
    print(f"figures written to {a.out}")


if __name__ == "__main__":
    main()
