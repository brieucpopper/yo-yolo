"""Phase 1 — Dataset parsing.

Scans the image directory, applies the optional image cap, derives a YOLO-safe
class name, and writes a `manifest.json` describing the train/val split.

This is the script to modify if the user wants something fancier than "use all
images in a directory" (e.g. recursive globbing, filtering by filename,
stratified sampling, ...). Keep the output `manifest.json` contract stable.

Usage:
    python scripts/parse_dataset.py --config config.yaml
    python scripts/parse_dataset.py --images-dir ./imgs --working-dir ./runs/x \
        --class-description "a banana" [--max-images 200]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (  # noqa: E402
    IMAGE_EXTENSIONS,
    ensure_dirs,
    get_logger,
    load_config,
    print_summary,
    resolve_config,
    save_resolved_config,
    working_paths,
    write_json,
)


def scan_images(images_dir: Path, recursive: bool = False) -> list[Path]:
    it = images_dir.rglob("*") if recursive else images_dir.iterdir()
    files = sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    return files


def build_manifest(config: dict) -> dict:
    log = get_logger("parse_dataset", config["working_dir"])
    images_dir = Path(config["images_dir"]).expanduser()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images_dir does not exist: {images_dir}")

    recursive = bool(config.get("recursive", False))
    images = scan_images(images_dir, recursive=recursive)
    log.info("Found %d images in %s", len(images), images_dir)
    if not images:
        raise RuntimeError(f"No supported images found in {images_dir}")

    max_images = config.get("max_images")
    if max_images:
        images = images[: int(max_images)]
        log.info("Capped to %d images (max_images)", len(images))

    rng = random.Random(config.get("seed", 42))
    shuffled = images[:]
    rng.shuffle(shuffled)
    val_split = float(config.get("val_split", 0.2))
    n_val = max(1, int(round(len(shuffled) * val_split))) if len(shuffled) > 1 else 0
    val_set = set(shuffled[:n_val])

    records = []
    for p in images:
        records.append({"path": str(p.resolve()), "split": "val" if p in val_set else "train"})

    n_train = sum(1 for r in records if r["split"] == "train")
    exts = sorted({Path(r["path"]).suffix.lower() for r in records})

    manifest = {
        "images_dir": str(images_dir),
        "class_name": config["class_name"],
        "class_names": config.get("class_names", [config["class_name"]]),
        "class_description": config.get("class_description", ""),
        "class_descriptions": config.get("class_descriptions", [config.get("class_description", "")]),
        "num_images": len(records),
        "num_train": n_train,
        "num_val": len(records) - n_train,
        "extensions": exts,
        "val_split": val_split,
        "seed": config.get("seed", 42),
        "images": records,
    }
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(
        description="YO-YOLO: scan images and build a manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="When --config is given all other flags become optional overrides.",
    )
    ap.add_argument("--config", help="Path to a YAML config file (all other args become optional overrides)")
    ap.add_argument("--images-dir", help="Folder containing raw images (overrides config)")
    ap.add_argument("--working-dir", help="Where pipeline outputs go (overrides config)")
    ap.add_argument("--class-description", help="Natural-language description of what to detect (overrides config)")
    ap.add_argument("--class-name", help="Short YOLO-safe class name (auto-derived if omitted)")
    ap.add_argument("--max-images", type=int, help="Cap number of images used")
    ap.add_argument("--val-split", type=float, help="Fraction held out for validation")
    ap.add_argument("--recursive", action="store_true", help="Scan images_dir recursively")
    args = ap.parse_args()

    config = load_config(args.config) if args.config else {}
    for key, val in {
        "images_dir": args.images_dir,
        "working_dir": args.working_dir,
        "class_description": args.class_description,
        "class_name": args.class_name,
        "max_images": args.max_images,
        "val_split": args.val_split,
    }.items():
        if val is not None:
            config[key] = val
    if args.recursive:
        config["recursive"] = True

    # Validate required fields (may come from config or CLI)
    missing = [f for f in ("images_dir", "working_dir", "class_description") if not config.get(f)]
    if missing:
        ap.error(
            f"Missing required field(s): {missing}. "
            "Provide via --config <file> or explicit CLI flags."
        )

    config = resolve_config(config)
    paths = working_paths(config["working_dir"])
    ensure_dirs(paths["root"], paths["logs"])
    save_resolved_config(config, config["working_dir"])

    manifest = build_manifest(config)
    write_json(paths["manifest"], manifest)

    print_summary(
        {
            "stage": "parse_dataset",
            "num_images": manifest["num_images"],
            "num_train": manifest["num_train"],
            "num_val": manifest["num_val"],
            "class_name": manifest["class_name"],
            "extensions": manifest["extensions"],
            "manifest": str(paths["manifest"]),
        }
    )


if __name__ == "__main__":
    main()
