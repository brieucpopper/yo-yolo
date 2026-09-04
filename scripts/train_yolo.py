"""Phase 6 — YOLO training.

Trains a small YOLO11n (or YOLOv8n) detector on the teacher-annotated dataset
using Ultralytics, with the default augmentation recipe from the plan.

Outputs `best.pt`, `last.pt`, and `results.csv` under the working dir, and
records where they landed in `train_summary.json`.

Usage:
    python scripts/train_yolo.py --config config.yaml [--epochs N] [--device 0]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (  # noqa: E402
    get_logger,
    load_config,
    print_summary,
    resolve_config,
    working_paths,
    write_json,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="YO-YOLO: train the student detector")
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--device", default=None, help="e.g. '0', '0,1', or 'cpu'")
    args = ap.parse_args()

    config = resolve_config(load_config(args.config))
    paths = working_paths(config["working_dir"])
    log = get_logger("train_yolo", config["working_dir"])

    tcfg = config.get("train", {}) or {}
    augment = tcfg.get("augment", {}) or {}

    epochs = args.epochs or tcfg.get("epochs", 100)
    imgsz = args.imgsz or tcfg.get("imgsz", 640)
    batch = args.batch or tcfg.get("batch", 16)
    model_name = tcfg.get("model", "yolo11n.pt")

    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics is not installed. `pip install ultralytics`.")
        print_summary({"stage": "train_yolo", "trained": False, "reason": "ultralytics not installed"})
        sys.exit(1)

    log.info("Training %s for %d epochs (imgsz=%d, batch=%d)", model_name, epochs, imgsz, batch)
    model = YOLO(model_name)

    patience = tcfg.get("patience", 30)

    class LossPlateauStopper:
        def __init__(self, model, patience, log):
            self.model = model
            self.patience = patience
            self.log = log
            self.best_loss = float("inf")
            self.plateau_epochs = 0

        def on_train_epoch_end(self, trainer):
            loss = trainer.loss_items
            if isinstance(loss, dict):
                current = float(loss.get("box", next(iter(loss.values()))))
            elif hasattr(loss, "__iter__"):
                try:
                    current = float(loss[0])  # box_loss
                except (KeyError, IndexError, TypeError):
                    current = float(list(loss)[0])
            else:
                current = float(loss)
            if current < self.best_loss:
                self.best_loss = current
                self.plateau_epochs = 0
            else:
                self.plateau_epochs += 1
                if self.plateau_epochs >= self.patience:
                    self.log.info("Loss plateau for %d epochs (best=%.4f). Stopping early.", self.patience, self.best_loss)
                    trainer.stop_train = True
                    trainer.epoch = trainer.epoch - 1  # don't count the plateau epochs

    stopper = LossPlateauStopper(model, patience, log)
    model.add_callback("on_train_epoch_end", stopper.on_train_epoch_end)

    start = time.perf_counter()
    results = model.train(
        data=str(paths["data_yaml"]),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        project=str(paths["runs"]),
        name="train",
        exist_ok=True,
        device=args.device,
        verbose=False,
        workers=2,
        **augment,
    )
    duration = time.perf_counter() - start

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else paths["runs"] / "train"
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    results_csv = save_dir / "results.csv"

    # Mirror the key artifacts to the top level of the working dir for easy access.
    for src, name in ((best, "best.pt"), (last, "last.pt"), (results_csv, "results.csv")):
        if src.exists():
            shutil.copy2(src, paths["root"] / name)

    summary = {
        "model": model_name,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "augment": augment,
        "duration_seconds": round(duration, 1),
        "save_dir": str(save_dir),
        "best_pt": str(paths["root"] / "best.pt"),
        "last_pt": str(paths["root"] / "last.pt"),
        "results_csv": str(paths["root"] / "results.csv"),
    }
    write_json(paths["root"] / "train_summary.json", summary)
    log.info("Training finished in %.1fs -> %s", duration, best)

    print_summary({"stage": "train_yolo", "trained": True, **{k: summary[k] for k in (
        "model", "epochs", "duration_seconds", "best_pt")}})


if __name__ == "__main__":
    main()
