"""Volleyball example — full-video inference + longest-rally detection.

Uses the distilled YOLO student (runs/volleyball_rally/best.pt) on every frame,
then applies very simple tracking:
  1. per-frame: keep highest-confidence ball box (or None)
  2. gap fill: linearly interpolate boxes across detection gaps <= max_gap
     (default 25 frames = 1s at 25fps) — handles single missed frames
  3. smooth: moving-average of center+size (window 5) for stable overlay
  4. rallies: contiguous ball-present segments; a gap > max_gap ends the rally.
     Longest rally = longest such segment.

Outputs (runs/volleyball_rally/rally/):
  detections.csv, rallies.json, longest_rally.mp4,
  presence_timeline.png, trajectory_map.png

Usage:
    python scripts/track_rally.py --config volleyball.yaml [--conf 0.2] [--max-gap 25]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import get_logger, load_config, print_summary, resolve_config, working_paths  # noqa: E402


def interpolate_gaps(boxes, max_gap):
    """Fill None gaps of length <= max_gap by linear interpolation. Returns filled list + mask."""
    n = len(boxes)
    filled = [b if b is not None else None for b in boxes]
    is_interp = [False] * n
    i = 0
    while i < n:
        if filled[i] is not None:
            i += 1
            continue
        j = i
        while j < n and filled[j] is None:
            j += 1
        gap = j - i
        left = filled[i - 1] if i > 0 else None
        right = filled[j] if j < n else None
        if left is not None and right is not None and gap <= max_gap:
            for k in range(gap):
                t = (k + 1) / (gap + 1)
                filled[i + k] = [float(l + (r - l) * t) for l, r in zip(left, right)]
                is_interp[i + k] = True
        i = j
    return filled, is_interp


def smooth_track(boxes, window=5):
    """Moving average over present boxes; None stays None."""
    n = len(boxes)
    out = [None] * n
    half = window // 2
    for i in range(n):
        if boxes[i] is None:
            continue
        pts = [boxes[k] for k in range(max(0, i - half), min(n, i + half + 1)) if boxes[k] is not None]
        out[i] = [float(sum(v) / len(pts)) for v in zip(*pts)]
    return out


def find_rallies(present, max_gap, fps):
    """Segments of present==True split by gaps > max_gap. Returns list of (start, end) inclusive."""
    n = len(present)
    rallies = []
    i = 0
    while i < n:
        if not present[i]:
            i += 1
            continue
        start = i
        while i < n:
            if present[i]:
                i += 1
                continue
            j = i
            while j < n and not present[j]:
                j += 1
            if j - i > max_gap:
                break
            i = j
        rallies.append((start, i - 1))
    return rallies


def main() -> None:
    ap = argparse.ArgumentParser(description="Volleyball longest-rally tracker")
    ap.add_argument("--config", required=True)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--max-gap", type=int, default=25, help="frames of no-ball that end a rally (25 = 1s @25fps)")
    ap.add_argument("--video", default="data/volleyball.mp4")
    args = ap.parse_args()

    config = resolve_config(load_config(args.config))
    paths = working_paths(config["working_dir"])
    log = get_logger("track_rally", config["working_dir"])
    outdir = paths["root"] / "rally"
    outdir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(str(paths["root"] / "best.pt"))

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info("Video %s: %d frames @ %.1ffps %dx%d", args.video, total, fps, W, H)

    raw, confs = [], []
    from tqdm import tqdm
    for _ in tqdm(range(total), desc="Detecting ball", unit="frame"):
        ok, frame = cap.read()
        if not ok:
            raw.append(None)
            confs.append(0.0)
            continue
        res = model.predict(frame, imgsz=640, conf=args.conf, verbose=False)[0]
        if len(res.boxes) == 0:
            raw.append(None)
            confs.append(0.0)
        else:
            b = max(res.boxes, key=lambda x: float(x.conf))
            raw.append([float(v) for v in b.xyxy[0].tolist()])
            confs.append(float(b.conf))
    cap.release()

    filled, is_interp = interpolate_gaps(raw, args.max_gap)
    smooth = smooth_track(filled)
    present = [b is not None for b in filled]
    rallies = find_rallies(present, args.max_gap, fps)
    log.info("Progress: detection done — %d/%d frames with ball (%.1f%%)",
             sum(1 for b in raw if b), total, 100 * sum(1 for b in raw if b) / total)

    rally_infos = []
    for s, e in rallies:
        rally_infos.append({"start_frame": s, "end_frame": e,
                            "start_s": round(s / fps, 2), "end_s": round(e / fps, 2),
                            "duration_s": round((e - s + 1) / fps, 2),
                            "num_frames": e - s + 1})
    rally_infos.sort(key=lambda r: r["num_frames"], reverse=True)
    longest = rally_infos[0] if rally_infos else None
    log.info("Found %d rallies; longest = %s", len(rally_infos), longest)

    with open(outdir / "detections.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_s", "x1", "y1", "x2", "y2", "conf", "interpolated"])
        for i, (b, c) in enumerate(zip(smooth, confs)):
            if b is None:
                w.writerow([i, round(i / fps, 3), "", "", "", "", "", ""])
            else:
                w.writerow([i, round(i / fps, 3), *[round(v, 1) for v in b],
                            round(c, 3), int(is_interp[i])])

    import json as _json
    (outdir / "rallies.json").write_text(_json.dumps(
        {"fps": fps, "total_frames": total, "num_rallies": len(rally_infos),
         "rallies": rally_infos[:20], "longest": longest}, indent=2))

    if longest:
        s, e = longest["start_frame"], longest["end_frame"]
        cap = cv2.VideoCapture(args.video)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(outdir / "longest_rally.mp4"), fourcc, fps, (W, H))
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        for i in tqdm(range(s, e + 1), desc="Rendering rally clip"):
            ok, frame = cap.read()
            if not ok:
                break
            b = smooth[i]
            if b is not None:
                x1, y1, x2, y2 = [int(v) for v in b]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 210, 60), 3)
                cv2.putText(frame, "ball", (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 210, 60), 2)
            cv2.putText(frame, f"RALLY t={i / fps - s / fps:.1f}s / {longest['duration_s']}s",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            vw.write(frame)
        vw.release()
        cap.release()
        log.info("Clip written: %s (%d frames)", outdir / "longest_rally.mp4", e - s + 1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(total) / fps
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.fill_between(t, 0, 1, where=np.array(present, dtype=bool), color="green", alpha=0.7, label="ball visible")
    if longest:
        ax.axvspan(longest["start_s"], longest["end_s"], color="gold", alpha=0.5, label="longest rally")
    ax.set_xlabel("time (s)")
    ax.set_yticks([])
    ax.set_title(f"Ball presence over full match — longest rally {longest['duration_s']}s "
                 f"({longest['start_s']}s–{longest['end_s']}s)" if longest else "Ball presence")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(outdir / "presence_timeline.png", dpi=120)
    plt.close(fig)

    if longest:
        s, e = longest["start_frame"], longest["end_frame"]
        fig, ax = plt.subplots(figsize=(10, 6))
        xs = [(b[0] + b[2]) / 2 / W for b in smooth[s:e + 1] if b is not None]
        ys = [1 - (b[1] + b[3]) / 2 / H for b in smooth[s:e + 1] if b is not None]
        ax.scatter(xs, ys, c=np.linspace(0, 1, len(xs)), cmap="viridis", s=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Ball trajectory — longest rally ({longest['duration_s']}s)")
        ax.set_xlabel("court x (normalized)")
        ax.set_ylabel("court y (normalized)")
        fig.tight_layout()
        fig.savefig(outdir / "trajectory_map.png", dpi=120)
        plt.close(fig)

    print_summary({"stage": "track_rally", "num_rallies": len(rally_infos), "longest": longest,
                   "ball_visible_pct": round(100 * sum(present) / len(present), 1) if present else 0,
                   "outdir": str(outdir)})


if __name__ == "__main__":
    main()
