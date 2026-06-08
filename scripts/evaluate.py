"""Phase 7 — Evaluation.

Validates the trained YOLO model (mAP50, mAP50-95, precision, recall) and runs
inference over the validation images, storing teacher (ground-truth) and YOLO
boxes per image for the failure-mining and dashboard stages.

Outputs:
    eval/metrics.json
    eval/predictions/<image>.json   # {teacher_boxes, yolo_boxes, width, height}

Usage:
    python scripts/evaluate.py --config config.yaml [--weights path/to/best.pt]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (  # noqa: E402
    ensure_dirs,
    get_logger,
    load_config,
    print_summary,
    read_json,
    resolve_config,
    working_paths,
    write_json,
)


def _teacher_boxes_for(stem: str, paths: dict) -> tuple[list, int, int]:
    """Load teacher boxes (xyxy pixels) from annotation metadata."""
    meta_path = paths["metadata"] / f"{stem}.json"
    if meta_path.exists():
        meta = read_json(meta_path)
        boxes = [[b["x1"], b["y1"], b["x2"], b["y2"]] for b in meta.get("boxes_xyxy", [])]
        return boxes, meta.get("width", 0), meta.get("height", 0)
    return [], 0, 0


def main() -> None:
    ap = argparse.ArgumentParser(description="YO-YOLO: evaluate the trained detector")
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    config = resolve_config(load_config(args.config))
    paths = working_paths(config["working_dir"])
    log = get_logger("evaluate", config["working_dir"])
    ensure_dirs(paths["eval"], paths["predictions"])

    weights = Path(args.weights) if args.weights else paths["root"] / "best.pt"
    if not weights.exists():
        log.error("Weights not found: %s (train first)", weights)
        print_summary({"stage": "evaluate", "evaluated": False, "reason": "weights not found"})
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics is not installed.")
        print_summary({"stage": "evaluate", "evaluated": False, "reason": "ultralytics not installed"})
        sys.exit(1)

    model = YOLO(str(weights))

    log.info("Computing validation metrics...")
    val = model.val(data=str(paths["data_yaml"]), device=args.device, verbose=False)
    box = val.box
    metrics = {
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "weights": str(weights),
    }

    # Per-image predictions + YOLO latency.
    val_images = sorted(p for p in paths["images_val"].iterdir() if p.is_file()
                        ) if paths["images_val"].exists() else []
    yolo_latencies = []
    for img_path in tqdm(val_images, desc="Inferring (val)", unit="img"):
        with Image.open(img_path) as im:
            iw, ih = im.size
        start = time.perf_counter()
        res = model.predict(str(img_path), verbose=False, device=args.device)[0]
        yolo_latencies.append((time.perf_counter() - start) * 1000.0)

        yolo_boxes = []
        for b in res.boxes:
            xyxy = b.xyxy[0].tolist()
            yolo_boxes.append({"xyxy": xyxy, "confidence": float(b.conf[0])})

        teacher_boxes, tw, th = _teacher_boxes_for(img_path.stem, paths)
        write_json(
            paths["predictions"] / f"{img_path.stem}.json",
            {
                "image": str(img_path),
                "width": iw,
                "height": ih,
                "teacher_boxes": teacher_boxes,
                "yolo_boxes": yolo_boxes,
            },
        )

    metrics["yolo_avg_latency_ms"] = round(sum(yolo_latencies) / len(yolo_latencies), 2) if yolo_latencies else 0.0
    metrics["num_val_images"] = len(val_images)
    write_json(paths["metrics"], metrics)
    log.info(
        "mAP50=%.3f mAP50-95=%.3f P=%.3f R=%.3f", metrics["mAP50"], metrics["mAP50_95"],
        metrics["precision"], metrics["recall"],
    )

    print_summary({"stage": "evaluate", "evaluated": True, **{k: round(metrics[k], 4) for k in (
        "mAP50", "mAP50_95", "precision", "recall")}, "yolo_avg_latency_ms": metrics["yolo_avg_latency_ms"]})


if __name__ == "__main__":
    main()
