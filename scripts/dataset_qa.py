"""Phase 4 — Dataset QA.

Computes lightweight-but-useful dataset statistics, bounding-box size buckets,
and validates every label file. Writes `qa_report.md` and `qa_stats.json`.

Usage:
    python scripts/dataset_qa.py --config config.yaml
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
    get_logger,
    load_config,
    print_summary,
    read_json,
    resolve_config,
    working_paths,
    write_json,
)

# Bounding-box area buckets, defined as fraction of image area (COCO-like).
SMALL_MAX = 0.01   # < 1% of image area
MEDIUM_MAX = 0.09  # < 9% of image area


def iter_label_files(paths: dict):
    for split, lbl_dir, img_dir in (
        ("train", paths["labels_train"], paths["images_train"]),
        ("val", paths["labels_val"], paths["images_val"]),
    ):
        if not lbl_dir.exists():
            continue
        for lbl in sorted(lbl_dir.glob("*.txt")):
            yield split, lbl, img_dir


def find_image(img_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"):
        for cand in (img_dir / f"{stem}{ext}", img_dir / f"{stem}{ext.upper()}"):
            if cand.exists():
                return cand
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="YO-YOLO: dataset QA")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    config = resolve_config(load_config(args.config))
    paths = working_paths(config["working_dir"])
    log = get_logger("dataset_qa", config["working_dir"])

    num_images = 0
    num_objects = 0
    empty_images = 0
    per_image_counts = []
    sizes = {"small": 0, "medium": 0, "large": 0}
    issues = {
        "invalid_class_id": 0,
        "out_of_range_coords": 0,
        "non_positive_area": 0,
        "out_of_bounds": 0,
        "missing_image": 0,
        "malformed_line": 0,
    }

    for split, lbl, img_dir in tqdm(list(iter_label_files(paths)), desc="QA", unit="img"):
        num_images += 1
        img_path = find_image(img_dir, lbl.stem)
        if img_path is None:
            issues["missing_image"] += 1
            continue
        with Image.open(img_path) as im:
            iw, ih = im.size

        lines = [ln for ln in lbl.read_text().splitlines() if ln.strip()]
        if not lines:
            empty_images += 1
        per_image_counts.append(len(lines))

        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                issues["malformed_line"] += 1
                continue
            try:
                cid = int(float(parts[0]))
                xc, yc, w, h = (float(v) for v in parts[1:])
            except ValueError:
                issues["malformed_line"] += 1
                continue

            num_objects += 1
            if cid != 0:
                issues["invalid_class_id"] += 1
            if not all(0.0 <= v <= 1.0 for v in (xc, yc, w, h)):
                issues["out_of_range_coords"] += 1
            if w <= 0 or h <= 0:
                issues["non_positive_area"] += 1
            if (xc - w / 2 < -1e-6) or (yc - h / 2 < -1e-6) or (xc + w / 2 > 1 + 1e-6) or (yc + h / 2 > 1 + 1e-6):
                issues["out_of_bounds"] += 1

            area = w * h
            if area < SMALL_MAX:
                sizes["small"] += 1
            elif area < MEDIUM_MAX:
                sizes["medium"] += 1
            else:
                sizes["large"] += 1

    objs_per_image = (num_objects / num_images) if num_images else 0.0
    empty_pct = (empty_images / num_images * 100.0) if num_images else 0.0
    total_issues = sum(issues.values())

    stats = {
        "num_images": num_images,
        "num_objects": num_objects,
        "objects_per_image": round(objs_per_image, 3),
        "empty_images": empty_images,
        "empty_image_pct": round(empty_pct, 2),
        "bbox_sizes": sizes,
        "validation_issues": issues,
        "total_issues": total_issues,
        "passed": total_issues == 0,
    }
    write_json(paths["qa_stats"], stats)
    _write_report(paths["qa_report"], config, stats)
    log.info("QA complete: %d objects, %d issues", num_objects, total_issues)

    print_summary({"stage": "dataset_qa", **{k: stats[k] for k in (
        "num_images", "num_objects", "objects_per_image", "empty_image_pct", "total_issues", "passed")},
        "qa_report": str(paths["qa_report"])})


def _write_report(path: Path, config: dict, stats: dict) -> None:
    s = stats
    lines = [
        f"# Dataset QA — {config.get('class_name', 'object')}",
        "",
        "## Dataset Statistics",
        f"- Image count: **{s['num_images']}**",
        f"- Object count: **{s['num_objects']}**",
        f"- Objects per image: **{s['objects_per_image']}**",
        f"- Empty images: **{s['empty_images']}** ({s['empty_image_pct']}%)",
        "",
        "## Bounding Box Sizes",
        f"- Small (<1% area): **{s['bbox_sizes']['small']}**",
        f"- Medium (1–9% area): **{s['bbox_sizes']['medium']}**",
        f"- Large (>9% area): **{s['bbox_sizes']['large']}**",
        "",
        "## Label Validation",
    ]
    for key, val in s["validation_issues"].items():
        flag = "✅" if val == 0 else "⚠️"
        lines.append(f"- {flag} {key.replace('_', ' ')}: **{val}**")
    lines += ["", f"**Overall: {'PASSED ✅' if s['passed'] else 'ISSUES FOUND ⚠️'}**", ""]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
