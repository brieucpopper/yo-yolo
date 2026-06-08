"""Shared helpers for the YO-YOLO pipeline scripts.

Centralizes config loading, working-dir layout, logging, the YOLO-safe class
name derivation, and a few small geometry helpers so the individual stage
scripts stay focused on their job.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# A small stop-word list used when deriving a shorthand class name from a free
# text description (e.g. "A ripe yellow banana fruit" -> "banana").
_STOPWORDS = {
    "a", "an", "the", "of", "with", "and", "or", "this", "that", "these",
    "those", "is", "are", "in", "on", "at", "to", "for", "include", "including",
    "ripe", "partially", "occluded", "some", "any", "all", "image", "images",
    "photo", "photos", "picture", "pictures", "object", "objects", "fruit",
    "item", "items", "thing", "things",
}

_COLOR_WORDS = {
    "red", "green", "blue", "yellow", "orange", "purple", "pink", "black",
    "white", "gray", "grey", "brown", "dark", "light", "bright", "pale",
}


# --------------------------------------------------------------------- config


def load_config(path: str | Path) -> dict:
    """Load a YAML config file and return a dict."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def resolve_config(config: dict) -> dict:
    """Fill in derived/default fields on a raw config dict."""
    cfg = dict(config)
    cfg.setdefault("max_images", None)
    cfg.setdefault("review_after_annotation", True)
    cfg.setdefault("val_split", 0.2)
    cfg.setdefault("seed", 42)
    if not cfg.get("class_name"):
        cfg["class_name"] = derive_class_name(cfg.get("class_description", "object"))
    return cfg


def save_resolved_config(config: dict, working_dir: str | Path) -> Path:
    out = Path(working_dir) / "config.resolved.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)
    return out


def derive_class_name(description: str) -> str:
    """Derive a YOLO-safe shorthand class name from a free-text description.

    "A ripe yellow banana fruit. Include partially occluded bananas." -> "banana"
    The heuristic is intentionally simple; the orchestrator/user can override
    by setting ``class_name`` in the config.
    """
    text = description.lower().strip()
    # First sentence only, strip punctuation.
    text = re.split(r"[.\n;]", text)[0]
    tokens = re.findall(r"[a-z][a-z\-]*", text)
    candidates = [
        t for t in tokens
        if t not in _STOPWORDS and t not in _COLOR_WORDS and len(t) > 1
    ]
    chosen = candidates[-1] if candidates else (tokens[-1] if tokens else "object")
    chosen = chosen.rstrip("s") if len(chosen) > 3 and chosen.endswith("s") else chosen
    return slugify(chosen)


def slugify(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return value or "object"


# --------------------------------------------------------------- working dir


def working_paths(working_dir: str | Path) -> dict[str, Path]:
    """Return the canonical sub-paths used across the pipeline."""
    wd = Path(working_dir)
    return {
        "root": wd,
        "dataset": wd / "dataset",
        "images_train": wd / "dataset" / "images" / "train",
        "images_val": wd / "dataset" / "images" / "val",
        "labels_train": wd / "dataset" / "labels" / "train",
        "labels_val": wd / "dataset" / "labels" / "val",
        "metadata": wd / "dataset" / "metadata",
        "data_yaml": wd / "dataset" / "data.yaml",
        "manifest": wd / "manifest.json",
        "preview_grid": wd / "preview_grid.png",
        "qa_report": wd / "qa_report.md",
        "qa_stats": wd / "qa_stats.json",
        "runs": wd / "runs",
        "eval": wd / "eval",
        "metrics": wd / "eval" / "metrics.json",
        "predictions": wd / "eval" / "predictions",
        "failures": wd / "failures",
        "report": wd / "report.md",
        "dashboard_assets": wd / "dashboard_assets",
        "logs": wd / "logs",
    }


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------- logging


def get_logger(name: str, working_dir: Optional[str | Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if working_dir is not None:
        logs = Path(working_dir) / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logs / f"{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ----------------------------------------------------------------- IO helpers


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str | Path, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def print_summary(summary: dict) -> None:
    """Print a machine-parseable one-line summary the orchestrator can read."""
    print("YOYOLO_SUMMARY " + json.dumps(summary))


# ---------------------------------------------------------------- geometry


def iou_xyxy(a: list[float], b: list[float]) -> float:
    """IoU of two boxes given as [x1, y1, x2, y2]."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def xyxy_to_yolo(box: dict, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert a pixel xyxy box dict to normalized YOLO (xc, yc, w, h)."""
    x1 = max(0.0, min(box["x1"], img_w))
    y1 = max(0.0, min(box["y1"], img_h))
    x2 = max(0.0, min(box["x2"], img_w))
    y2 = max(0.0, min(box["y2"], img_h))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    xc = (x1 + x2) / 2.0 / img_w
    yc = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return xc, yc, w, h


def yolo_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> list[float]:
    """Convert a normalized YOLO box back to pixel xyxy."""
    bw, bh = w * img_w, h * img_h
    cx, cy = xc * img_w, yc * img_h
    return [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]


# ---------------------------------------------------------------- drawing


def draw_boxes(
    image,
    boxes: list,
    color: tuple[int, int, int] = (255, 64, 64),
    width: int = 3,
    labels: Optional[list[str]] = None,
    dashed: bool = False,
):
    """Draw xyxy boxes on a copy of a PIL image and return it.

    ``boxes`` may be a list of ``[x1, y1, x2, y2]`` or dicts with x1..y2 keys.
    """
    from PIL import ImageDraw, ImageFont

    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001
        font = None

    for idx, box in enumerate(boxes):
        if isinstance(box, dict):
            xy = [box["x1"], box["y1"], box["x2"], box["y2"]]
        else:
            xy = list(box[:4])
        if dashed:
            _draw_dashed_rect(draw, xy, color, width)
        else:
            draw.rectangle(xy, outline=color, width=width)
        if labels and idx < len(labels) and labels[idx]:
            txt = labels[idx]
            ty = max(0, xy[1] - 12)
            draw.rectangle([xy[0], ty, xy[0] + 7 * len(txt) + 4, ty + 12], fill=color)
            draw.text((xy[0] + 2, ty), txt, fill=(255, 255, 255), font=font)
    return img


def _draw_dashed_rect(draw, xy, color, width, dash=8):
    x1, y1, x2, y2 = xy
    for (xa, ya, xb, yb) in (
        (x1, y1, x2, y1),
        (x2, y1, x2, y2),
        (x2, y2, x1, y2),
        (x1, y2, x1, y1),
    ):
        length = max(abs(xb - xa), abs(yb - ya))
        if length == 0:
            continue
        steps = int(length // dash)
        for s in range(0, steps + 1, 2):
            t0 = s / max(steps, 1)
            t1 = min((s + 1) / max(steps, 1), 1.0)
            draw.line(
                [xa + (xb - xa) * t0, ya + (yb - ya) * t0,
                 xa + (xb - xa) * t1, ya + (yb - ya) * t1],
                fill=color,
                width=width,
            )
