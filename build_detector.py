"""YO-YOLO — optional one-shot orchestrator.

Runs the full pipeline end to end by calling the stage scripts in sequence.
The interactive, agent-driven path lives in SKILL.md; this script is the
"just build it" convenience entry point.

Usage:
    python build_detector.py --config config.yaml
    python build_detector.py --config config.yaml --yes        # skip preview prompt
    python build_detector.py --config config.yaml --no-review  # skip FiftyOne
    python build_detector.py --config config.yaml --no-dashboard
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"


def run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(SCRIPTS / script), *args]
    print(f"\n=== running {script} {' '.join(args)} ===", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"Stage failed: {script} (exit {result.returncode})")


def main() -> None:
    ap = argparse.ArgumentParser(description="YO-YOLO one-shot pipeline")
    ap.add_argument("--config", required=True)
    ap.add_argument("--yes", action="store_true", help="Skip the preview approval prompt")
    ap.add_argument("--no-review", action="store_true", help="Skip the FiftyOne review phase")
    ap.add_argument("--no-dashboard", action="store_true", help="Do not launch the dashboard at the end")
    args = ap.parse_args()

    cfg = ["--config", args.config]

    run("parse_dataset.py", *cfg)
    run("preview_annotation.py", *cfg)

    if not args.yes:
        print("\nReview preview_grid.png in the working dir.")
        if input("Continue with full annotation? [y/N] ").strip().lower() not in ("y", "yes"):
            raise SystemExit("Aborted by user after preview.")

    run("annotate_dataset.py", *cfg)
    run("dataset_qa.py", *cfg)

    if not args.no_review:
        try:
            run("launch_fiftyone.py", *cfg, "--no-wait")
        except SystemExit as exc:
            print(f"(FiftyOne review skipped: {exc})")

    run("train_yolo.py", *cfg)
    run("evaluate.py", *cfg)
    run("failure_mining.py", *cfg)
    run("generate_report.py", *cfg)

    print("\nPipeline complete. Outputs are in the working dir.")
    if not args.no_dashboard:
        run("launch_dashboard.py", *cfg)


if __name__ == "__main__":
    main()
