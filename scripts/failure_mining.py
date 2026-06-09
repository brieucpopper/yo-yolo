"""Phase 8 — Failure mining.

Surfaces the most interesting teacher-vs-student disagreements on the validation
set. Each validation image is scored by combining:

* count mismatch (teacher boxes vs YOLO boxes), and
* localization error (1 - mean IoU of greedily matched boxes).

The top-K images are rendered side by side (teacher = warm/red palette,
YOLO = green palette) into `failures/` together with an `index.json`.

Usage:
    python scripts/failure_mining.py --config config.yaml [--top-k 50]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (  # noqa: E402
    draw_boxes,
    ensure_dirs,
    get_logger,
    get_model_class_color,
    iou_xyxy,
    load_config,
    print_summary,
    read_json,
    resolve_config,
    working_paths,
    write_json,
)


def greedy_mean_iou(teacher: list, yolo: list) -> float:
    """Mean IoU of greedily matched boxes (0 if either side empty)."""
    if not teacher or not yolo:
        return 0.0
    teacher_xyxy = [b["xyxy"] if isinstance(b, dict) else b for b in teacher]
    yolo_xyxy = [b["xyxy"] if isinstance(b, dict) else b for b in yolo]
    used = set()
    ious = []
    for t in teacher_xyxy:
        best, best_j = 0.0, -1
        for j, y in enumerate(yolo_xyxy):
            if j in used:
                continue
            v = iou_xyxy(t, y)
            if v > best:
                best, best_j = v, j
        if best_j >= 0:
            used.add(best_j)
            ious.append(best)
        else:
            ious.append(0.0)
    return sum(ious) / len(ious) if ious else 0.0


def score(teacher: list, yolo: list, iou_thr: float) -> dict:
    n_t, n_y = len(teacher), len(yolo)
    count_diff = abs(n_t - n_y)
    mean_iou = greedy_mean_iou(teacher, yolo)
    loc_error = 1.0 - mean_iou if (n_t and n_y) else (1.0 if (n_t or n_y) else 0.0)
    # Disagreement score: count mismatches dominate, localization breaks ties.
    disagreement = count_diff + loc_error
    reason = []
    if n_t and not n_y:
        reason.append(f"teacher found {n_t}, YOLO found 0")
    elif n_y and not n_t:
        reason.append(f"YOLO found {n_y}, teacher found 0")
    elif count_diff:
        reason.append(f"count mismatch ({n_t} vs {n_y})")
    if n_t and n_y and mean_iou < iou_thr:
        reason.append(f"low IoU ({mean_iou:.2f})")
    return {
        "num_teacher": n_t,
        "num_yolo": n_y,
        "count_diff": count_diff,
        "mean_iou": round(mean_iou, 3),
        "disagreement": round(disagreement, 3),
        "reason": "; ".join(reason) or "agreement",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="YO-YOLO: mine teacher/student failures")
    ap.add_argument("--config", required=True)
    ap.add_argument("--top-k", type=int)
    args = ap.parse_args()

    config = resolve_config(load_config(args.config))
    paths = working_paths(config["working_dir"])
    log = get_logger("failure_mining", config["working_dir"])
    ensure_dirs(paths["failures"])

    fcfg = config.get("failure_mining", {}) or {}
    top_k = args.top_k or fcfg.get("top_k", 50)
    iou_thr = fcfg.get("iou_threshold", 0.5)

    pred_files = sorted(paths["predictions"].glob("*.json")) if paths["predictions"].exists() else []
    if not pred_files:
        log.error("No predictions found. Run evaluate.py first.")
        print_summary({"stage": "failure_mining", "mined": False, "reason": "no predictions"})
        sys.exit(1)

    scored = []
    for pf in pred_files:
        pred = read_json(pf)
        s = score(pred.get("teacher_boxes", []), pred.get("yolo_boxes", []), iou_thr)
        s.update({"image": pred["image"], "stem": pf.stem, "width": pred["width"], "height": pred["height"]})
        scored.append((s, pred))

    scored.sort(key=lambda x: x[0]["disagreement"], reverse=True)
    top = [s for s in scored if s[0]["disagreement"] > 0][:top_k]

    index = []
    for rank, (s, pred) in enumerate(tqdm(top, desc="Rendering failures", unit="img"), 1):
        try:
            image = Image.open(pred["image"]).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        teacher_raw = pred.get("teacher_boxes", [])
        yolo_raw = pred.get("yolo_boxes", [])
        teacher_boxes = [b["xyxy"] if isinstance(b, dict) else b for b in teacher_raw]
        yolo_boxes    = [b["xyxy"] if isinstance(b, dict) else b for b in yolo_raw]
        teacher_colors = [get_model_class_color("teacher", b.get("class_id", 0) if isinstance(b, dict) else 0)
                          for b in teacher_raw]
        yolo_colors    = [get_model_class_color("yolo", b.get("class_id", 0) if isinstance(b, dict) else 0)
                          for b in yolo_raw]
        vis = draw_boxes(image, teacher_boxes, color=teacher_colors)
        vis = draw_boxes(
            vis, yolo_boxes, color=yolo_colors,
            labels=[f"{b['confidence']:.2f}" if isinstance(b, dict) else "" for b in yolo_raw],
        )
        out = paths["failures"] / f"{rank:03d}_{Path(pred['image']).stem}.png"
        vis.save(out)
        index.append({"rank": rank, "render": str(out), **s})

    write_json(paths["failures"] / "index.json", index)
    log.info("Mined %d failure cases -> %s", len(index), paths["failures"])

    print_summary({"stage": "failure_mining", "mined": True, "num_failures": len(index),
                   "failures_dir": str(paths["failures"])})


if __name__ == "__main__":
    main()
