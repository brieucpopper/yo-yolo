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


def write_data_yaml(paths: dict, class_names: list[str]) -> None:
    data = {
        "path": str(paths["dataset"].resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)},
        "nc": len(class_names),
    }
    with open(paths["data_yaml"], "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def _annotate_one(rec, paths, class_names, class_descriptions, config, kwargs):
    """Annotate a single image. Creates its own worker (thread-safe)."""
    from common import read_json as _read_json  # noqa: F401 (kept local for threads)
    from locate_anything import LocateAnythingWorker as _Worker
    from locate_anything import worker_from_config as _worker_from_config

    src = Path(rec["path"])
    split = rec["split"]
    img_dir = paths["images_train"] if split == "train" else paths["images_val"]
    lbl_dir = paths["labels_train"] if split == "train" else paths["labels_val"]
    stem = src.stem
    dst_img = img_dir / src.name
    try:
        image = Image.open(src).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        return {"skipped": True, "error": str(exc), "stem": stem}

    import time as _time
    worker = _worker_from_config(config)
    result = None
    for attempt in range(3):
        try:
            result = worker.detect(image, class_descriptions, **kwargs)
            break
        except Exception:
            if attempt == 2:
                raise
            _time.sleep(2 * (attempt + 1))
    all_boxes = _Worker.parse_boxes(result["answer"], image.width, image.height)

    MAX_DIM = 1800
    if max(image.width, image.height) > MAX_DIM:
        ratio = MAX_DIM / max(image.width, image.height)
        image_resized = image.resize(
            (int(image.width * ratio), int(image.height * ratio)),
            Image.LANCZOS,
        )
    else:
        image_resized = image
    image_resized.save(dst_img)

    lines = []
    per_class = {}
    for b in all_boxes:
        cid = b.get("class_id", 0)
        if cid >= len(class_names):
            continue
        xc, yc, w, h = xyxy_to_yolo(b, image.width, image.height)
        if w <= 0 or h <= 0:
            continue
        lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        per_class[class_names[cid]] = per_class.get(class_names[cid], 0) + 1

    (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    write_json(
        paths["metadata"] / f"{stem}.json",
        {
            "image": str(dst_img),
            "source": str(src),
            "split": split,
            "width": image.width,
            "height": image.height,
            "num_boxes": len(lines),
            "boxes_xyxy": all_boxes,
            "latency_ms": result["latency_ms"],
            "raw_answer": result["answer"],
        },
    )
    return {"skipped": False, "num_boxes": len(lines), "latency_ms": result["latency_ms"],
            "per_class": per_class, "stem": stem}


def main() -> None:
    ap = argparse.ArgumentParser(description="YO-YOLO: annotate the dataset with the teacher")
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=None,
                    help="Parallel teacher requests (endpoint backend only). Matches lserver -np.")
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
    class_names = manifest.get("class_names", [manifest["class_name"]])
    class_descriptions = manifest.get("class_descriptions", [manifest.get("class_description", "")])

    workers = args.workers or config.get("teacher", {}).get("workers", 1)
    log.info("Loading teacher (%s backend, workers=%d)...",
             config.get("teacher", {}).get("backend", "auto"), workers)
    probe: LocateAnythingWorker = worker_from_config(config)
    if workers > 1 and probe.backend == "local":
        log.warning("workers>1 is only supported with the endpoint backend; falling back to 1.")
        workers = 1

    gen_mode = config.get("teacher", {}).get("generation_mode", "fast")
    kwargs = {"generation_mode": gen_mode} if probe.backend == "local" else {}

    total_boxes, empty_images, latencies = 0, 0, []
    class_box_counts = {name: 0 for name in class_names}

    def _accumulate(res):
        nonlocal total_boxes, empty_images
        if res.get("skipped"):
            empty_images += 1
            return
        latencies.append(res["latency_ms"])
        total_boxes += res["num_boxes"]
        if not res["num_boxes"]:
            empty_images += 1
        for k, v in res.get("per_class", {}).items():
            class_box_counts[k] = class_box_counts.get(k, 0) + v

    if workers <= 1:
        for i, rec in enumerate(tqdm(records, desc="Annotating", unit="img"), 1):
            res = _annotate_one(rec, paths, class_names, class_descriptions, config, kwargs)
            if res.get("skipped") and "error" in res:
                log.warning("Skipping unreadable image %s (%s)", rec["path"], res["error"])
            _accumulate(res)
            if i % 100 == 0:
                log.info("Progress %d/%d images — %d boxes so far", i, len(records), total_boxes)
    else:
        import concurrent.futures
        log.info("Annotating with %d parallel streams...", workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_annotate_one, rec, paths, class_names,
                              class_descriptions, config, kwargs) for rec in records]
            for i, fut in enumerate(tqdm(concurrent.futures.as_completed(futs),
                                         total=len(futs), desc=f"Annotating x{workers}", unit="img"), 1):
                try:
                    res = fut.result()
                except Exception as exc:  # noqa: BLE001
                    log.warning("Annotation failed (%s)", exc)
                    empty_images += 1
                    continue
                _accumulate(res)
                if i % 100 == 0:
                    log.info("Progress %d/%d images — %d boxes so far", i, len(records), total_boxes)

    write_data_yaml(paths, class_names)

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    annotation_stats = {
        "num_images": len(records),
        "total_boxes": total_boxes,
        "empty_images": empty_images,
        "avg_latency_ms": round(avg_latency, 2),
        "total_teacher_seconds": round(sum(latencies) / 1000.0, 2),
        "class_names": class_names,
        "boxes_per_class": class_box_counts,
    }
    write_json(paths["dataset"] / "annotation_stats.json", annotation_stats)
    log.info("Annotation done: %d boxes across %d images", total_boxes, len(records))

    print_summary({"stage": "annotate_dataset", **annotation_stats, "data_yaml": str(paths["data_yaml"])})


if __name__ == "__main__":
    main()
