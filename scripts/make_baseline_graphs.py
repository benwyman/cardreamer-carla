#!/usr/bin/env python3
# make_baseline_graphs.py
#
# Baseline curves for:
#   - episode/reward_rate
#   - episode/score
#   - episode/length
#
# Excludes: eval_runs, combined_runs_*
# Figures -> ./figures/training

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# ========= config =========
METRICS = [
    "episode/reward_rate",
    "episode/score",
    "episode/length",
]
SMOOTH_WINDOW = 131           # moving-average window (odd)
MIN_POINTS_PER_SERIES = 20
FIGSIZE = (10, 6)
DPI = 150

# Friendly axis labels
YLABELS = {
    "episode/reward_rate": "Reward Rate",
    "episode/score": "Score",
    "episode/length": "Episode Length",
}
XLABEL = "Step"

# Fixed agent order so colors match eval
AGENT_ORDER = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]

# Colors aligned with eval (tab10-like)
AGENT_COLORS = {
    "A1": "#1f77b4",  # blue
    "A2": "#ff7f0e",  # orange
    "A3": "#2ca02c",  # green
    "A4": "#d62728",  # red
    "A5": "#9467bd",  # purple
    "A6": "#8c564b",  # brown
    "A7": "#17becf",  # cyan
}

# Your shorthand
SENSOR_SHORTHANDS = {
    "A1": "F",
    "A2": "S",
    "A3": "LiDAR",
    "A4": "F+S",
    "A5": "F+L",
    "A6": "S+L",
    "A7": "F+S+L",
}

def canonical_agent_id(seed: str) -> str:
    """Extract A1..A7 from any seed-like string."""
    m = re.search(r"\bA[1-7]\b", seed.upper())
    return m.group(0) if m else seed.upper()

def map_seed_label(seed: str) -> tuple[str, str | None]:
    """Return display label and color using canonical agent ID."""
    aid = canonical_agent_id(seed)
    shorthand = SENSOR_SHORTHANDS.get(aid, "")
    label = f"{aid} — {shorthand}" if shorthand else aid
    color = AGENT_COLORS.get(aid)
    return label, color

# ========= path helpers =========
def _any_part_contains(parts, needle):
    needle = needle.lower()
    return any(needle in p.lower() for p in parts)

def is_baseline_path(p: Path) -> bool:
    parts = list(p.parts)
    if _any_part_contains(parts, "eval_runs"):
        return False
    if any(re.search(r"^combined_runs", part.lower()) for part in parts):
        return False
    return _any_part_contains(parts, "left_turn") or _any_part_contains(parts, "lane_merge")

def task_from_path(p: Path) -> str | None:
    parts = [s.lower() for s in p.parts]
    if any("left_turn" in s for s in parts):
        return "left_turn"
    if any("lane_merge" in s for s in parts):
        return "lane_merge"
    return None

def seed_from_path(p: Path) -> str:
    for part in p.parts:
        m = re.search(r"\b([Aa]\d+)\b", part)
        if m:
            return m.group(1).upper()
    for part in p.parts:
        m = re.search(r"\b([Aa]\d+)", part)
        if m:
            return m.group(1).upper()
    return p.parent.name

# ========= data helpers =========
def smooth_mavg(y: np.ndarray, window: int) -> np.ndarray:
    """Moving average with edge padding (no zero-padding artifacts)."""
    n = len(y)
    if n < 3 or window <= 1:
        return y
    max_odd = n if (n % 2 == 1) else (n - 1)
    win = min(window, max_odd)
    if win < 3:
        return y
    pad = win // 2
    kernel = np.ones(win, dtype=float) / win
    ypad = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(ypad, kernel, mode="valid")  # length == n

STEP_KEYS = ("env_step", "step", "global_step", "trainer/steps")

def extract_step(obj) -> float | None:
    for k in STEP_KEYS:
        if k in obj:
            try:
                return float(obj[k])
            except Exception:
                pass
    return None  # STRICT: skip rows without a defined step

def dedup_by_step(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Collapse duplicate x by averaging y."""
    if len(xs) == 0:
        return xs, ys, 0
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    uniq, idx_start = np.unique(xs, return_index=True)
    means = []
    dropped = 0
    for i, start in enumerate(idx_start):
        end = idx_start[i + 1] if i + 1 < len(idx_start) else len(xs)
        if end - start == 1:
            means.append(ys[start])
        else:
            means.append(float(np.mean(ys[start:end])))
            dropped += (end - start - 1)
    return uniq.astype(float), np.array(means, dtype=float), dropped

def load_series(jsonl_path: Path, wanted_keys: set[str]) -> dict[str, list[tuple[float, float]]]:
    out = {k: [] for k in wanted_keys}
    bad_lines = 0
    lines_with_step = 0
    lines_without_step = 0

    with jsonl_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                bad_lines += 1
                continue

            step = extract_step(obj)
            if step is None:
                lines_without_step += 1
                continue  # STRICT: skip anything without a defined step
            else:
                lines_with_step += 1

            for k in wanted_keys:
                if k in obj:
                    try:
                        val = float(obj[k])
                    except Exception:
                        continue
                    out[k].append((step, val))

    if bad_lines:
        print(f"[warn] {jsonl_path} had {bad_lines} malformed lines (ignored).")
    print(f"[debug] {jsonl_path} -> with_step={lines_with_step}, no_step={lines_without_step}")
    for k in wanted_keys:
        print(f"        {k}: {len(out[k])} points")
    return out

def plot_task_metric(task: str, metric: str, runs: dict[str, list[tuple[float, float]]], figdir: Path):
    plt.figure(figsize=FIGSIZE)
    any_plotted = False
    plotted = 0
    skipped_short = 0

    # Plot in fixed agent order to force consistent colors.
    ordered_keys = [a for a in AGENT_ORDER if a in runs] + [k for k in runs.keys() if k not in AGENT_ORDER]

    for seed in ordered_keys:
        series = runs[seed]
        if not series:
            print(f"[skip] {task} {metric} — {seed}: no data.")
            continue

        xs = np.array([t[0] for t in series], dtype=float)
        ys = np.array([t[1] for t in series], dtype=float)
        mask = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[mask], ys[mask]

        if len(xs) < MIN_POINTS_PER_SERIES:
            skipped_short += 1
            continue

        xs, ys, dropped = dedup_by_step(xs, ys)
        if dropped:
            print(f"[note] {task} {metric} — {seed}: collapsed {dropped} duplicates by step.")

        ys_s = smooth_mavg(ys, SMOOTH_WINDOW)
        label, color = map_seed_label(seed)
        plt.plot(xs, ys_s, linewidth=1.6, alpha=0.95, label=label, color=color)
        plotted += 1
        any_plotted = True

    if not any_plotted:
        print(f"[warn] {task} {metric}: nothing to plot (series_skipped_short={skipped_short}).")
        plt.close()
        return

    plt.title(metric)  # keep title as raw metric name
    plt.xlabel(XLABEL)
    plt.ylabel(YLABELS.get(metric, metric))
    plt.grid(True, alpha=0.3)
    plt.legend(title="Legend", ncols=2, fontsize=9)

    out_path = figdir / f"{task}__{metric.replace('/', '_')}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    plt.close()
    print(f"[ok] Saved: {out_path} (series_plotted={plotted}, series_skipped_short={skipped_short})")

# ========= main =========
def main():
    if len(sys.argv) != 2:
        print('Usage: python make_baseline_graphs.py ".\\CARLA LOGS"')
        sys.exit(1)

    logs_root = Path(sys.argv[1]).resolve()
    if not logs_root.exists():
        print(f"[error] Logs root not found: {logs_root}")
        sys.exit(1)

    # Save into figures/training
    figdir = Path.cwd() / "figures" / "training"
    figdir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Figures -> {figdir}")

    all_jsonl = sorted(logs_root.rglob("metrics.jsonl"))
    baseline_files = [p for p in all_jsonl if is_baseline_path(p)]
    if not baseline_files:
        print(f"[error] No baseline metrics.jsonl under {logs_root}")
        sys.exit(1)

    print("[info] Found baseline runs:")
    for p in baseline_files:
        print("  -", p)

    wanted = set(METRICS)
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # data[task][metric][seed] -> list[(x,y)]

    for jf in baseline_files:
        task = task_from_path(jf)
        if task is None:
            print(f"[skip] cannot infer task from: {jf}")
            continue
        seed = seed_from_path(jf)
        series = load_series(jf, wanted)

        missing = [k for k in METRICS if not series[k]]
        if missing:
            print(f"[info] {task} {seed}: missing -> {', '.join(missing)}")

        for k, seq in series.items():
            if seq:
                # Store under canonical agent ID to align with colors/labels
                aid = canonical_agent_id(seed)
                data[task][k][aid].extend(seq)

    for task in sorted(data.keys()):
        for metric in METRICS:
            runs = data[task].get(metric, {})
            if not runs:
                print(f"[warn] {task} {metric}: no runs had this metric.")
                continue
            plot_task_metric(task, metric, runs, figdir)

    print("[done] Baseline graph generation complete.")

if __name__ == "__main__":
    main()
