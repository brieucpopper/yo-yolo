"""Phase 9 — Markdown report.

Assembles `report.md` from all the artifacts produced by earlier stages:
dataset summary, annotation statistics, training summary, metrics, sample
predictions, failure cases, and the speed comparison.

Sample prediction montages are written to `dashboard_assets/`.

Usage:
    python scripts/generate_report.py --config config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (  # noqa: E402
    draw_boxes,
    ensure_dirs,
    get_logger,
    get_model_class_color,
    load_config,
    print_summary,
    read_json,
    resolve_config,
    working_paths,
)


def _safe_read(path: Path):
    return read_json(path) if path.exists() else {}


def _rel(target: Path, base: Path) -> str:
    try:
        return str(target.relative_to(base))
    except ValueError:
        return str(target)


def _sample_montage(paths: dict, n: int = 6) -> list[str]:
    """Render up to n validation predictions to dashboard_assets/, return paths."""
    out_paths = []
    pred_files = sorted(paths["predictions"].glob("*.json")) if paths["predictions"].exists() else []
    for pf in pred_files[:n]:
        pred = read_json(pf)
        try:
            img = Image.open(pred["image"]).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        teacher_raw = pred.get("teacher_boxes", [])
        yolo_raw    = pred.get("yolo_boxes", [])
        teacher_boxes  = [b["xyxy"] if isinstance(b, dict) else b for b in teacher_raw]
        yolo_boxes     = [b["xyxy"] if isinstance(b, dict) else b for b in yolo_raw]
        teacher_colors = [get_model_class_color("teacher", b.get("class_id", 0) if isinstance(b, dict) else 0)
                          for b in teacher_raw]
        yolo_colors    = [get_model_class_color("yolo",    b.get("class_id", 0) if isinstance(b, dict) else 0)
                          for b in yolo_raw]
        vis = draw_boxes(img, teacher_boxes, color=teacher_colors)
        vis = draw_boxes(vis, yolo_boxes, color=yolo_colors)
        out = paths["dashboard_assets"] / f"sample_{pf.stem}.png"
        vis.save(out)
        out_paths.append(out)
    return out_paths


def main() -> None:
    ap = argparse.ArgumentParser(description="YO-YOLO: generate the markdown report")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    config = resolve_config(load_config(args.config))
    paths = working_paths(config["working_dir"])
    log = get_logger("generate_report", config["working_dir"])
    ensure_dirs(paths["dashboard_assets"])

    manifest = _safe_read(paths["manifest"])
    ann = _safe_read(paths["dataset"] / "annotation_stats.json")
    qa = _safe_read(paths["qa_stats"])
    train = _safe_read(paths["root"] / "train_summary.json")
    metrics = _safe_read(paths["metrics"])
    failures = _safe_read(paths["failures"] / "index.json") if (paths["failures"] / "index.json").exists() else []

    base = paths["root"]
    samples = _sample_montage(paths)

    teacher_ms = ann.get("avg_latency_ms", 0)
    yolo_ms = metrics.get("yolo_avg_latency_ms", 0)

    L = []
    L.append(f"# YO-YOLO Report — `{manifest.get('class_name', 'object')}`\n")
    L.append(f"> {manifest.get('class_description', '')}\n")

    L.append("## Dataset Summary\n")
    L.append(f"- Images: **{manifest.get('num_images', 0)}** "
             f"(train {manifest.get('num_train', 0)} / val {manifest.get('num_val', 0)})")
    L.append(f"- Objects: **{qa.get('num_objects', ann.get('total_boxes', 0))}**")
    L.append(f"- Objects per image: **{qa.get('objects_per_image', '—')}**")
    L.append(f"- Empty images: **{qa.get('empty_image_pct', '—')}%**\n")

    L.append("## Annotation Statistics\n")
    L.append(f"- Teacher avg latency: **{teacher_ms} ms/img**")
    L.append(f"- Total teacher time: **{ann.get('total_teacher_seconds', '—')} s**")
    L.append(f"- Total boxes: **{ann.get('total_boxes', 0)}**")
    if qa:
        b = qa.get("bbox_sizes", {})
        L.append(f"- Box sizes — small {b.get('small', 0)}, medium {b.get('medium', 0)}, large {b.get('large', 0)}")
        L.append(
            f"- QA validation: **{'PASSED' if qa.get('passed') else str(qa.get('total_issues', 0)) + ' issues'}**")
    L.append("")

    L.append("## Training Summary\n")
    if train:
        L.append(f"- Model: **{train.get('model', '—')}**")
        L.append(
            f"- Epochs: **{train.get('epochs', '—')}**, imgsz {train.get('imgsz', '—')}, batch {train.get('batch', '—')}")
        L.append(f"- Duration: **{train.get('duration_seconds', '—')} s**")
        aug = train.get("augment", {})
        if aug:
            L.append(f"- Augmentations: `{aug}`")
    else:
        L.append("- _Not trained yet._")
    L.append("")

    L.append("## Metrics\n")
    if metrics:
        L.append("| Metric | Value |")
        L.append("| --- | --- |")
        L.append(f"| mAP50 | {metrics.get('mAP50', 0):.3f} |")
        L.append(f"| mAP50-95 | {metrics.get('mAP50_95', 0):.3f} |")
        L.append(f"| Precision | {metrics.get('precision', 0):.3f} |")
        L.append(f"| Recall | {metrics.get('recall', 0):.3f} |")
    else:
        L.append("- _Not evaluated yet._")
    L.append("")

    L.append("## Sample Predictions\n")
    L.append("_Teacher = warm colors (red family), YOLO = cool colors (green family)._\n")
    for s in samples:
        L.append(f"![sample]({_rel(s, base)})")
    L.append("")

    L.append("## Failure Analysis\n")
    if failures:
        L.append(f"Top {len(failures)} teacher/student disagreements.\n")
        for f in failures[:6]:
            L.append(
                f"- **#{f['rank']}** — {f['reason']} (teacher {f['num_teacher']}, YOLO {f['num_yolo']}, IoU {f['mean_iou']})")
            L.append(f"  ![failure]({_rel(Path(f['render']), base)})")
    else:
        L.append("- _No failure cases mined._")
    L.append("")

    L.append("## Speed Comparison\n")
    L.append("| System | Latency |")
    L.append("| --- | --- |")
    L.append(f"| Locate Anything (teacher) | {teacher_ms} ms |")
    L.append(f"| YOLO (student) | {yolo_ms} ms |")
    if teacher_ms and yolo_ms:
        L.append(f"\n**YOLO is ~{teacher_ms / yolo_ms:.0f}× faster than the teacher.**")
    L.append("")

    paths["report"].write_text("\n".join(L))
    log.info("Report written -> %s", paths["report"])

    print_summary({"stage": "generate_report", "report": str(paths["report"]),
                   "num_samples": len(samples), "num_failures": len(failures)})


if __name__ == "__main__":
    main()
