"""Phase 3 — Full annotation.

Runs the Locate Anything teacher over every image in the manifest, converts the
predicted boxes to YOLO format, and lays out an Ultralytics-style dataset:

    dataset/
    ├── images/{train,val}/
    ├── labels/{train,val}/
    ├── metadata/<image>.json   # raw answer, boxes, latency
    └── data.yaml

Per-image teacher latency is stored for the later speed comparison.

Usage:
    python scripts/annotate_dataset.py --config config.yaml
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml
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
    xyxy_to_yolo,
)
from locate_anything import LocateAnythingWorker, worker_from_config  # noqa: E402


def write_data_yaml(paths: dict, class_name: str) -> None:
    data = {
        "path": str(paths["dataset"].resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: class_name},
        "nc": 1,
    }
    with open(paths["data_yaml"], "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="YO-YOLO: annotate the dataset with the teacher")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    config = resolve_config(load_config(args.config))
    paths = working_paths(config["working_dir"])
    log = get_logger("annotate_dataset", config["working_dir"])

    ensure_dirs(
        paths["images_train"], paths["images_val"],
        paths["labels_train"], paths["labels_val"],
        paths["metadata"],
    )

    manifest = read_json(paths["manifest"])
    records = manifest["images"]
    class_name = manifest["class_name"]
    phrase = config["class_description"]
    gen_mode = config.get("teacher", {}).get("generation_mode", "fast")

    log.info("Loading teacher (%s backend)...", config.get("teacher", {}).get("backend", "auto"))
    worker: LocateAnythingWorker = worker_from_config(config)

    total_boxes, empty_images, latencies = 0, 0, []

    for rec in tqdm(records, desc="Annotating", unit="img"):
        src = Path(rec["path"])
        split = rec["split"]
        img_dir = paths["images_train"] if split == "train" else paths["images_val"]
        lbl_dir = paths["labels_train"] if split == "train" else paths["labels_val"]

        stem = src.stem
        dst_img = img_dir / src.name
        try:
            image = Image.open(src).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            log.warning("Skipping unreadable image %s (%s)", src, exc)
            continue

        kwargs = {"generation_mode": gen_mode} if worker.backend == "local" else {}
        result = worker.ground_multi(image, phrase, **kwargs)
        boxes = LocateAnythingWorker.parse_boxes(result["answer"], image.width, image.height)
        latencies.append(result["latency_ms"])

        shutil.copy2(src, dst_img)

        lines = []
        for b in boxes:
            xc, yc, w, h = xyxy_to_yolo(b, image.width, image.height)
            if w <= 0 or h <= 0:
                continue
            lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

        total_boxes += len(lines)
        if not lines:
            empty_images += 1

        write_json(
            paths["metadata"] / f"{stem}.json",
            {
                "image": str(dst_img),
                "source": str(src),
                "split": split,
                "width": image.width,
                "height": image.height,
                "num_boxes": len(lines),
                "boxes_xyxy": boxes,
                "latency_ms": result["latency_ms"],
                "raw_answer": result["answer"],
            },
        )

    write_data_yaml(paths, class_name)

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    annotation_stats = {
        "num_images": len(records),
        "total_boxes": total_boxes,
        "empty_images": empty_images,
        "avg_latency_ms": round(avg_latency, 2),
        "total_teacher_seconds": round(sum(latencies) / 1000.0, 2),
        "class_name": class_name,
    }
    write_json(paths["dataset"] / "annotation_stats.json", annotation_stats)
    log.info("Annotation done: %d boxes across %d images", total_boxes, len(records))

    print_summary({"stage": "annotate_dataset", **annotation_stats, "data_yaml": str(paths["data_yaml"])})


if __name__ == "__main__":
    main()
