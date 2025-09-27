# make_eval_graphs.py
# Eval graphs organized by task with all weathers on the same figure.
# X-axis shows agent IDs; a second horizontal line shows compact sensor tags.
# For each agent, 3 grouped bars (default, clearnoon, wetnoon) share the agent color
# and are differentiated by hatch patterns and slight spacing. No mean/SEM overlays.

from __future__ import annotations
import json
import math
from pathlib import Path
from collections import defaultdict, OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# -----------------------------
# Styling
# -----------------------------
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "font.size": 11,
})

# -----------------------------
# Fixed agent order, colors, and sensor labels
# -----------------------------
AGENT_ORDER = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]

# Distinct, readable colors (tab10 palette)
AGENT_COLORS = {
    "A1": "#1f77b4",  # blue
    "A2": "#ff7f0e",  # orange
    "A3": "#2ca02c",  # green
    "A4": "#d62728",  # red
    "A5": "#9467bd",  # purple
    "A6": "#8c564b",  # brown
    "A7": "#17becf",  # cyan
}

# Full sensor descriptions (for reference if needed later)
AGENT_SENSORS = OrderedDict([
    ("A1", "Front only (F)"),
    ("A2", "Sides only (L+R)"),
    ("A3", "LiDAR only"),
    ("A4", "Front + Sides"),
    ("A5", "Front + LiDAR"),
    ("A6", "Sides + LiDAR"),
    ("A7", "Front + Sides + LiDAR"),
])

# Compact sensor tags for x-axis second line
AGENT_SENSORS_SHORT = {
    "A1": "F",
    "A2": "S",
    "A3": "LiDAR",
    "A4": "F+S",
    "A5": "F+L",
    "A6": "S+L",
    "A7": "F+S+L",
}

# Weather ordering and hatch patterns (bars stay agent-colored; patterns distinguish weather)
# Canonical keys use "clearnoon" (not "cloudynoon") to match directory names like '*_clearnoon'.
WEATHER_ORDER = ["default", "clearnoon", "wetnoon"]
WEATHER_LABEL = {"default": "Default", "clearnoon": "ClearNoon", "wetnoon": "WetNoon"}
WEATHER_HATCH = {"default": "", "clearnoon": "//", "wetnoon": "xx"}

def canonical_weather(w: str) -> str:
    """Normalize various spellings to canonical keys in WEATHER_ORDER."""
    s = w.lower().replace(" ", "").replace("-", "").replace("_", "")
    if s in ("default", ""):
        return "default"
    if s in ("clearnoon", "cloudynoon", "cloudynoone", "clearnoon"):
        return "clearnoon"
    if s in ("wetnoon", "wet_noon"):
        return "wetnoon"
    return s  # fall back to whatever was found

# -----------------------------
# File discovery
# -----------------------------
def find_eval_runs(root: Path) -> list[tuple[str, str, str, Path]]:
    """
    Return list of tuples: (task, weather, seed, metrics_path)
    Looks for: <root>/eval_runs/A{seed}/{scenario}/metrics.jsonl
    where scenario like 'left_turn_default' or 'lane_merge_clearnoon'
    """
    runs = []
    eval_root = root / "eval_runs"
    if not eval_root.exists():
        print(f"[warn] No eval_runs folder at: {eval_root}")
        return runs

    for seed_dir in sorted(eval_root.glob("A*")):
        seed = seed_dir.name  # e.g. A1
        for scen_dir in sorted(seed_dir.iterdir()):
            if not scen_dir.is_dir():
                continue
            name = scen_dir.name.lower()
            # Parse task + weather
            if name.startswith("left_turn"):
                task = "left_turn"
                weather = name.replace("left_turn_", "") if name != "left_turn" else "default"
            elif name.startswith("lane_merge"):
                task = "lane_merge"
                weather = name.replace("lane_merge_", "") if name != "lane_merge" else "default"
            else:
                continue

            metrics_path = scen_dir / "metrics.jsonl"
            if metrics_path.exists():
                runs.append((task, weather, seed, metrics_path))
    return runs

# -----------------------------
# Parsing one eval run file
# -----------------------------
def parse_eval_file(path: Path) -> dict:
    """
    Read metrics.jsonl for one eval run and compute episode-level aggregates.

    We compute:
      - episodes: count of 'episode/length'
      - success_rate: fraction of episodes with stats/sum_destination_reached > 0
      - collision_rate: fraction with stats/sum_is_collision > 0
      - out_of_lane_rate: fraction with stats/sum_out_of_lane > 0
      - time_exceeded_rate: fraction with stats/sum_time_exceeded > 0
      - mean_travel_distance: avg of stats/sum_travel_distance
      - mean_ttc: avg of stats/mean_ttc
      - mean_speed_norm: avg of stats/mean_speed_norm
      - mean_wpt_dis: avg of stats/mean_wpt_dis
      - mean_episode_length: avg of episode/length
      - mean_score: avg of episode/score
    """
    succ_ep, coll_ep, lane_ep, time_ep = [], [], [], []
    travel_ep, ttc_ep, spd_ep, wpt_ep = [], [], [], []
    length_ep, score_ep = [], []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue

            if "episode/length" in row:
                length_ep.append(row["episode/length"])
            if "episode/score" in row:
                score_ep.append(float(row["episode/score"]))

            if "stats/sum_destination_reached" in row:
                succ_ep.append(1.0 if row["stats/sum_destination_reached"] > 0 else 0.0)
            if "stats/sum_is_collision" in row:
                coll_ep.append(1.0 if row["stats/sum_is_collision"] > 0 else 0.0)
            if "stats/sum_out_of_lane" in row:
                lane_ep.append(1.0 if row["stats/sum_out_of_lane"] > 0 else 0.0)
            if "stats/sum_time_exceeded" in row:
                time_ep.append(1.0 if row["stats/sum_time_exceeded"] > 0 else 0.0)
            if "stats/sum_travel_distance" in row:
                travel_ep.append(float(row["stats/sum_travel_distance"]))
            if "stats/mean_ttc" in row:
                ttc_ep.append(float(row["stats/mean_ttc"]))
            if "stats/mean_speed_norm" in row:
                spd_ep.append(float(row["stats/mean_speed_norm"]))
            if "stats/mean_wpt_dis" in row:
                wpt_ep.append(float(row["stats/mean_wpt_dis"]))

    # Align counts defensively by trimming to the shortest list among episode vectors
    episode_vectors = [succ_ep, coll_ep, lane_ep, time_ep, travel_ep, ttc_ep, spd_ep, wpt_ep, length_ep, score_ep]
    valid_len = min((len(v) for v in episode_vectors if len(v) > 0), default=0)

    def _mean(x):
        return float(np.mean(x)) if len(x) else float("nan")

    if valid_len > 0:
        succ_ep   = succ_ep[:valid_len]
        coll_ep   = coll_ep[:valid_len]
        lane_ep   = lane_ep[:valid_len]
        time_ep   = time_ep[:valid_len]
        travel_ep = travel_ep[:valid_len]
        ttc_ep    = ttc_ep[:valid_len]
        spd_ep    = spd_ep[:valid_len]
        wpt_ep    = wpt_ep[:valid_len]
        length_ep = length_ep[:valid_len]
        score_ep  = score_ep[:valid_len]

    out = {
        "episodes": len(length_ep),
        "success_rate": _mean(succ_ep),
        "collision_rate": _mean(coll_ep),
        "out_of_lane_rate": _mean(lane_ep),
        "time_exceeded_rate": _mean(time_ep),
        "mean_travel_distance": _mean(travel_ep),
        "mean_ttc": _mean(ttc_ep),
        "mean_speed_norm": _mean(spd_ep),
        "mean_wpt_dis": _mean(wpt_ep),
        "mean_episode_length": _mean(length_ep),
        "mean_score": _mean(score_ep),
    }
    return out

# -----------------------------
# Plotting helpers
# -----------------------------
def bar_grouped_by_weather(values_by_weather: dict[str, dict[str, float]],
                           title: str, ylabel: str, outpath: Path):
    """
    Grouped bar chart with up to 3 weathers per agent.
    values_by_weather: { weather: { 'A1': value, ... }, ... }
    Bars share the agent color; weather is indicated by hatch patterns.
    (No mean/SEM overlays.)
    """
    weathers = [w for w in WEATHER_ORDER if w in values_by_weather]
    if not weathers:
        print(f"[skip] No weather data for figure: {outpath.name}")
        return

    # Build value matrix aligned to AGENT_ORDER × weathers
    agents = [a for a in AGENT_ORDER if any(a in values_by_weather[w] for w in weathers)]
    if not agents:
        print(f"[skip] No agent data for figure: {outpath.name}")
        return

    X = np.arange(len(agents))
    width = 0.24  # bar width per weather
    # Center the group around the tick: offsets for up to 3 weathers
    offsets = {
        1: [0.0],
        2: [-0.5*width, +0.5*width],
        3: [-width, 0.0, +width],
    }.get(len(weathers), np.linspace(-width*(len(weathers)-1)/2, width*(len(weathers)-1)/2, len(weathers)))

    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    # Plot bars for each weather
    for wi, weather in enumerate(weathers):
        vals = []
        for a in agents:
            v = values_by_weather[weather].get(a, np.nan)
            vals.append(v)
        vals = np.array(vals, float)

        # Each agent keeps its color; distinguish weather by hatch
        bar_colors = [AGENT_COLORS.get(a, "#999999") for a in agents]
        ax.bar(
            X + offsets[wi],
            vals,
            width=width * 0.9,  # slight gap between bars
            color=bar_colors,
            edgecolor="black",
            linewidth=0.5,
            hatch=WEATHER_HATCH.get(weather, ""),
            label=WEATHER_LABEL.get(weather, weather.title()),
            alpha=0.95,
        )

    # Two-line tick labels: top = agent ID, bottom = compact sensor tag
    tick_labels = [f"{a}\n{AGENT_SENSORS_SHORT.get(a, '')}" for a in agents]
    ax.set_xticks(X)
    ax.set_xticklabels(tick_labels)
    ax.tick_params(axis='x', pad=6)
    ax.margins(x=0.02)

    # Legend for weather (pattern-only, so bar colors remain agent-specific)
    legend_handles = [Patch(facecolor="white", edgecolor="black",
                            hatch=WEATHER_HATCH.get(w, ""), label=WEATHER_LABEL.get(w, w.title()))
                      for w in weathers]
    ax.legend(handles=legend_handles, title="Weather", frameon=False, ncol=len(weathers))

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Agent")
    # Expand limits to include outer bars
    ax.set_xlim(X[0] - 0.5 - width, X[-1] + 0.5 + width)

    fig.subplots_adjust(bottom=0.22)  # room for two-line ticks
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)

# -----------------------------
# Main
# -----------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str, help="Path to 'CARLA LOGS' parent folder")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    figroot = (root.parent / "figures" / "eval").resolve()
    print(f"[info] Figures -> {figroot}")

    runs = find_eval_runs(root)
    if not runs:
        print("[warn] No eval runs found.")
        return

    # Group by task → weather(canonical) → list[(seed, path)]
    by_task_weather = defaultdict(lambda: defaultdict(list))
    for task, weather, seed, path in runs:
        wcanon = canonical_weather(weather)
        by_task_weather[task][wcanon].append((seed, path))

    # What we will plot (title, ylabel, key in parsed dict)
    METRICS = [
        ("Success Rate", "Success Rate", "success_rate"),
        ("Collision Rate", "Collision Rate", "collision_rate"),
        ("Out of Lane Rate", "Out of Lane Rate", "out_of_lane_rate"),   # <-- added
        ("Score (avg)", "Score (avg)", "mean_score"),
        ("Speed Norm (avg)", "Speed Norm (avg)", "mean_speed_norm"),
        ("Travel Distance (avg)", "Distance (m, avg)", "mean_travel_distance"),
        ("Time Exceeded Rate", "Time Exceeded Rate", "time_exceeded_rate"),
    ]

    # Iterate tasks; for each, combine weathers on one figure per metric
    for task in sorted(by_task_weather.keys()):
        # Aggregate per weather → per agent values (episode-averaged metrics)
        agg_by_weather = {}
        for weather in WEATHER_ORDER:
            if weather not in by_task_weather[task]:
                continue
            items = sorted(by_task_weather[task][weather], key=lambda x: x[0])  # sort by seed name
            per_agent = {}
            for seed, mpath in items:
                parsed = parse_eval_file(mpath)
                per_agent[seed] = parsed
            agg_by_weather[weather] = per_agent

        # Report episodes per agent per weather
        print(f"[info] Task: {task}")
        for weather in WEATHER_ORDER:
            if weather not in agg_by_weather:
                print(f"  - {weather}: MISSING")
                continue
            print(f"  - {weather}:")
            for seed in AGENT_ORDER:
                if seed in agg_by_weather[weather]:
                    print(f"      • {seed}: episodes={agg_by_weather[weather][seed]['episodes']}")
                else:
                    print(f"      • {seed}: MISSING")

        # Make figures (one per metric) with grouped bars by weather
        outdir = figroot / task  # no per-weather subdir now
        pretty_task = task.replace('_', ' ').title()
        for title, ylabel, key in METRICS:
            # Build weather → {agent: value}
            values_by_weather = {}
            for weather, per_agent in agg_by_weather.items():
                vals = {a: per_agent[a][key] for a in per_agent if key in per_agent[a] and np.isfinite(per_agent[a][key])}
                if vals:
                    values_by_weather[weather] = vals

            if not values_by_weather:
                print(f"[skip] No data for {task} :: {title}")
                continue

            fname = f"{task}__allweathers__{key}.png"
            outpath = outdir / fname
            figure_title = f"{pretty_task} — {title} (default/clearnoon/wetnoon)"
            bar_grouped_by_weather(values_by_weather, title=figure_title, ylabel=ylabel, outpath=outpath)
            print(f"[ok] Saved: {outpath}")

    print("[done] Eval graph generation complete.")

if __name__ == "__main__":
    main()
