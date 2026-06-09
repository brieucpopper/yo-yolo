"""Phase 2 — Preview annotation.

Samples a few images, runs the Locate Anything teacher on them, draws the
predicted boxes, and assembles a single `preview_grid.png` so the user can
sanity-check the prompt BEFORE annotating the whole dataset.

Usage:
    python scripts/preview_annotation.py --config config.yaml [--num 4]
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (  # noqa: E402
    draw_boxes,
    get_logger,
    get_model_class_color,
    load_config,
    print_summary,
    read_json,
    resolve_config,
    working_paths,
)
from locate_anything import LocateAnythingWorker, worker_from_config  # noqa: E402


def make_grid(images: list[Image.Image], cols: int = 2) -> Image.Image:
    if not images:
        raise ValueError("no images to grid")
    cols = min(cols, len(images))
    rows = math.ceil(len(images) / cols)
    cell_w = max(im.width for im in images)
    cell_h = max(im.height for im in images)
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), (30, 30, 30))
    for idx, im in enumerate(images):
        r, c = divmod(idx, cols)
        grid.paste(im, (c * cell_w, r * cell_h))
    return grid


def main() -> None:
    ap = argparse.ArgumentParser(description="YO-YOLO: preview teacher annotations")
    ap.add_argument("--config", required=True)
    ap.add_argument("--num", type=int, default=4)
    args = ap.parse_args()

    config = resolve_config(load_config(args.config))
    paths = working_paths(config["working_dir"])
    log = get_logger("preview_annotation", config["working_dir"])

    manifest = read_json(paths["manifest"])
    image_paths = [r["path"] for r in manifest["images"]]
    rng = random.Random(config.get("seed", 42))
    sample = rng.sample(image_paths, min(args.num, len(image_paths)))

    worker: LocateAnythingWorker = worker_from_config(config)
    class_descriptions = config.get("class_descriptions", [config.get("class_description", "")])
    class_names = config.get("class_names", [config.get("class_name", "object")])
    gen_mode = config.get("teacher", {}).get("generation_mode", "fast")

    annotated, total_boxes, latencies = [], 0, []
    for i, p in enumerate(sample, 1):
        log.info("Previewing %d/%d: %s", i, len(sample), Path(p).name)
        image = Image.open(p).convert("RGB")
        kwargs = {"generation_mode": gen_mode} if worker.backend == "local" else {}
        result = worker.detect(image, class_descriptions, **kwargs)
        boxes = LocateAnythingWorker.parse_boxes(result["answer"], image.width, image.height)
        total_boxes += len(boxes)
        latencies.append(result["latency_ms"])
        labels = []
        colors = []
        for b in boxes:
            cid = b.get("class_id", 0)
            labels.append(class_names[cid] if cid < len(class_names) else f"class_{cid}")
            colors.append(get_model_class_color("teacher", cid))
        annotated.append(draw_boxes(image, boxes, color=colors, labels=labels))

    grid = make_grid(annotated, cols=2)
    paths["preview_grid"].parent.mkdir(parents=True, exist_ok=True)
    grid.save(paths["preview_grid"])
    abs_path = paths["preview_grid"].resolve()
    log.info("Saved preview grid -> %s", abs_path)
    print(f"Preview grid: {abs_path}")

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    print_summary(
        {
            "stage": "preview_annotation",
            "num_previewed": len(sample),
            "total_boxes": total_boxes,
            "avg_latency_ms": round(avg_latency, 1),
            "preview_grid": str(abs_path),
        }
    )


if __name__ == "__main__":
    main()
