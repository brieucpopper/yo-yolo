"""Phase 5 — Optional FiftyOne review.

Loads the dataset into FiftyOne for human inspection before (and optionally
after) training. Teacher labels are shown as "ground_truth"; if YOLO
predictions exist (after evaluation), they are added as "yolo".

By default only the validation split is launched.

Usage:
    python scripts/launch_fiftyone.py --config config.yaml [--split val|train|all]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (  # noqa: E402
    get_logger,
    load_config,
    print_summary,
    read_json,
    resolve_config,
    working_paths,
)


def _load_yolo_detections(label_path: Path, class_name: str):
    import fiftyone as fo

    dets = []
    if not label_path.exists():
        return fo.Detections(detections=dets)
    for ln in label_path.read_text().splitlines():
        parts = ln.split()
        if len(parts) != 5:
            continue
        _, xc, yc, w, h = (float(v) for v in parts)
        # FiftyOne expects [top-left-x, top-left-y, width, height] normalized.
        dets.append(fo.Detection(label=class_name, bounding_box=[xc - w / 2, yc - h / 2, w, h]))
    return fo.Detections(detections=dets)


def main() -> None:
    ap = argparse.ArgumentParser(description="YO-YOLO: launch FiftyOne review")
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="val", choices=["val", "train", "all"])
    ap.add_argument("--no-wait", action="store_true", help="Do not block on the session")
    ap.add_argument("--port", type=int, default=5151, help="Port for FiftyOne app (default 5151)")
    ap.add_argument("--background", action="store_true",
                    help="Daemonize: spawn a detached child that keeps session.wait() running, then return immediately")
    args = ap.parse_args()

    config = resolve_config(load_config(args.config))
    paths = working_paths(config["working_dir"])
    log = get_logger("launch_fiftyone", config["working_dir"])

    # --background: spawn a detached child that blocks on session.wait(), then return.
    if args.background:
        import subprocess
        logs_dir = paths["logs"]
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "fiftyone.log"
        pid_file = logs_dir / "fiftyone.pid"
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--config", args.config,
            "--split", args.split,
            "--port", str(args.port),
            # no --background → child will block on session.wait()
        ]
        with open(log_file, "w") as f:
            proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        pid_file.write_text(str(proc.pid))
        log.info("FiftyOne started in background (PID %d) on port %d.", proc.pid, args.port)
        log.info("Log:  %s", log_file)
        log.info("Stop: kill $(cat %s)", pid_file)
        print_summary({
            "stage": "launch_fiftyone",
            "launched": True,
            "background": True,
            "pid": proc.pid,
            "port": args.port,
            "url": f"http://localhost:{args.port}",
        })
        return

    try:
        import fiftyone as fo
    except ImportError:
        log.error("fiftyone is not installed. `pip install fiftyone` to use this phase.")
        print_summary({"stage": "launch_fiftyone", "launched": False, "reason": "fiftyone not installed"})
        return

    # Ensure FiftyOne can connect to MongoDB
    if not fo.config.database_uri:
        import os
        mongo_uri = os.environ.get("FIFTYONE_DATABASE_URI", "mongodb://localhost:27017")
        fo.config.database_uri = mongo_uri
        log.info("Set database_uri to %s", mongo_uri)

    manifest = read_json(paths["manifest"])
    class_name = manifest["class_name"]

    splits = ["train", "val"] if args.split == "all" else [args.split]
    dataset = fo.Dataset(name=f"yoyolo_{class_name}", overwrite=True)

    preds_dir = paths["predictions"]
    n_samples = 0
    for split in splits:
        img_dir = paths["images_train"] if split == "train" else paths["images_val"]
        lbl_dir = paths["labels_train"] if split == "train" else paths["labels_val"]
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.iterdir()):
            if not img.is_file():
                continue
            sample = fo.Sample(filepath=str(img))
            sample["split"] = split
            sample["ground_truth"] = _load_yolo_detections(lbl_dir / f"{img.stem}.txt", class_name)
            pred_json = preds_dir / f"{img.stem}.json"
            if pred_json.exists():
                sample["yolo"] = _yolo_pred_field(read_json(pred_json), class_name)
            dataset.add_sample(sample)
            n_samples += 1

    log.info("Loaded %d samples into FiftyOne dataset '%s'", n_samples, dataset.name)
    session = fo.launch_app(dataset, port=args.port)
    print_summary({
        "stage": "launch_fiftyone",
        "launched": True,
        "num_samples": n_samples,
        "port": args.port,
        "url": f"http://localhost:{args.port}",
    })
    if not args.no_wait:
        log.info("FiftyOne running at http://localhost:%d — Press Ctrl+C to stop.", args.port)
        session.wait()


def _yolo_pred_field(pred: dict, class_name: str):
    import fiftyone as fo

    w, h = pred.get("width", 1), pred.get("height", 1)
    dets = []
    for box in pred.get("yolo_boxes", []):
        x1, y1, x2, y2 = box["xyxy"] if "xyxy" in box else [box["x1"], box["y1"], box["x2"], box["y2"]]
        dets.append(
            fo.Detection(
                label=class_name,
                bounding_box=[x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h],
                confidence=box.get("confidence"),
            )
        )
    return fo.Detections(detections=dets)


if __name__ == "__main__":
    main()
