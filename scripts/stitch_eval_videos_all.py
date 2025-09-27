#!/usr/bin/env python3
"""
Scan all agents and eval runs under an input root and write stitched videos to a clean output root.
- Reads TensorBoard image summaries via EventAccumulator
- Decodes (GIF/PNG/JPEG) with Pillow
- Writes MP4 (H.264) via imageio (auto-picks ffmpeg from imageio-ffmpeg). If MP4 open fails, falls back to a single GIF.
- No intermediate snippet files.
- Streaming write to avoid huge memory use.
"""

import io
import os
import statistics
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import imageio

# ---- defaults tailored to your machine ----
INPUT_ROOT = Path("/mnt/c/Users/wyman/Desktop/dissertation/CARLA LOGS/eval_runs")
OUTPUT_ROOT = Path("/mnt/c/Users/wyman/Desktop/dissertation/evalvideos")

TAGS = [
    "stats/policy_camera",
    "stats/policy_camera_left",
    "stats/policy_camera_right",
    "stats/policy_lidar",
    "stats/policy_birdeye_wpt",
]

def find_event_file(run_dir: Path) -> Path | None:
    c = sorted(run_dir.glob("events.out.tfevents.*.v2"))
    if not c:
        c = sorted(run_dir.glob("events.out.tfevents.*"))
    return c[0] if c else None

def iter_agents_and_runs(input_root: Path):
    # agents like A1..A7 under input_root
    for agent_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in agent_dir.iterdir() if p.is_dir()):
            evt = find_event_file(run_dir)
            if evt:
                yield agent_dir.name, run_dir, evt

def pil_frame_iter(img: Image.Image):
    """Yield (PIL.Image RGB, duration_ms:int)."""
    if getattr(img, "is_animated", False) and getattr(img, "n_frames", 1) > 1:
        for frame in ImageSequence.Iterator(img):
            rgb = frame.convert("RGB")
            dur = int(frame.info.get("duration", img.info.get("duration", 0)) or 0)
            yield rgb, dur
    else:
        yield img.convert("RGB"), int(img.info.get("duration", 0)) or 0

def compute_fps(durations_ms: list[int]) -> int:
    nz = [d for d in durations_ms if d > 0]
    if nz:
        med = statistics.median(nz)
        fps = int(round(1000.0 / med))
        return max(1, min(60, fps))
    return 8

def even_pad_needed(size_wh: tuple[int, int]) -> tuple[int, int]:
    w, h = size_wh
    return (w % 2, h % 2)

def ensure_even_frame(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    ph, pw = h % 2, w % 2
    if ph == 0 and pw == 0:
        return arr
    return np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode="edge")

def process_one_tag(agent: str, run_dir: Path, acc: EventAccumulator, tag: str, out_dir: Path):
    events = sorted(
        acc.Images(tag),
        key=lambda e: (int(getattr(e, "step", 0)), float(getattr(e, "wall_time", 0.0))),
    )
    if not events:
        print(f"[{agent}/{run_dir.name}] {tag} -> skip (no events)")
        return

    # Pass 1: determine target size (first frame) and collect durations to compute FPS
    durations_ms = []
    target_size = None  # (W,H)

    for e in events:
        with Image.open(io.BytesIO(e.encoded_image_string)) as img:
            for rgb, dur in pil_frame_iter(img):
                if target_size is None:
                    target_size = rgb.size  # (W,H)
                durations_ms.append(dur)

    fps = compute_fps(durations_ms)
    base = tag.replace("/", "_")
    mp4_path = out_dir / f"{base}.mp4"
    gif_path = out_dir / f"{base}.gif"

    # Writer open (MP4 first, else GIF)
    writer = None
    used_fmt = None
    try:
        writer = imageio.get_writer(
            mp4_path, fps=fps, codec="libx264"  # ffmpeg is auto-resolved by imageio-ffmpeg
        )
        used_fmt = "mp4"
    except Exception as ex:
        print(f"[{agent}/{run_dir.name}] {tag} MP4 open failed ({ex}); using GIF fallback.")
        writer = imageio.get_writer(gif_path, duration=1.0/max(fps,1), loop=0)
        used_fmt = "gif"

    # Second pass: decode and stream frames to writer
    count = 0
    w_even_pad, h_even_pad = even_pad_needed(target_size)
    padded = (w_even_pad or h_even_pad)

    for e in events:
        with Image.open(io.BytesIO(e.encoded_image_string)) as img:
            for rgb, _ in pil_frame_iter(img):
                if rgb.size != target_size:
                    rgb = rgb.resize(target_size, Image.BILINEAR)
                arr = np.array(rgb, dtype=np.uint8)
                if used_fmt == "mp4" and padded:
                    arr = ensure_even_frame(arr)
                writer.append_data(arr)
                count += 1

    writer.close()
    out_name = mp4_path.name if used_fmt == "mp4" else gif_path.name
    print(f"[{agent}/{run_dir.name}] {tag} -> {out_name} ({count} frames @ {fps} fps)")

def process_run(agent: str, run_dir: Path, evt_path: Path, out_root: Path):
    out_dir = out_root / agent / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{agent}/{run_dir.name}] loading: {evt_path.name}")
    acc = EventAccumulator(str(evt_path), size_guidance={"images": 0})
    acc.Reload()

    avail = set(acc.Tags().get("images", []))
    for tag in TAGS:
        if tag in avail:
            process_one_tag(agent, run_dir, acc, tag, out_dir)
        else:
            print(f"[{agent}/{run_dir.name}] {tag} -> skip (tag missing)")

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    any_found = False
    for agent, run_dir, evt in iter_agents_and_runs(INPUT_ROOT):
        any_found = True
        process_run(agent, run_dir, evt, OUTPUT_ROOT)
    if not any_found:
        print(f"No runs found under: {INPUT_ROOT}")

if __name__ == "__main__":
    main()
