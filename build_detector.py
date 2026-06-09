"""YO-YOLO — optional one-shot orchestrator.

Runs the full pipeline end to end by calling the stage scripts in sequence.
The interactive, agent-driven path lives in SKILL.md; this script is the
"just build it" convenience entry point.

Usage:
    python build_detector.py --config config.yaml
    python build_detector.py --config config.yaml --yes        # skip preview prompt
    python build_detector.py --config config.yaml --no-review  # skip FiftyOne
    python build_detector.py --config config.yaml --no-dashboard
    python build_detector.py --config config.yaml --venv-path /path/to/venv
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"


def get_python(venv_path: str | None) -> str:
    """Get python executable, using uv if available."""
    if venv_path:
        venv_python = Path(venv_path) / "bin" / "python"
        if venv_python.exists():
            return str(venv_python)
    # Try uv
    try:
        result = subprocess.run(["uv", "python", "find", "3.13"], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return sys.executable


def run(script: str, python: str, *args: str) -> None:
    cmd = [python, str(SCRIPTS / script), *args]
    print(f"\n=== running {script} {' '.join(args)} ===", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"Stage failed: {script} (exit {result.returncode})")


def main() -> None:
    ap = argparse.ArgumentParser(description="YO-YOLO one-shot pipeline")
    ap.add_argument("--config", required=True)
    ap.add_argument("--yes", action="store_true", help="Skip the preview approval prompt")
    ap.add_argument("--no-review", action="store_true", help="Skip the Voxel51 review phase")
    ap.add_argument("--no-dashboard", action="store_true", help="Do not launch the dashboard at the end")
    ap.add_argument("--venv-path", help="Path to existing uv virtual environment to reuse")
    args = ap.parse_args()

    python = get_python(args.venv_path)
    if args.venv_path:
        print(f"Using existing venv: {args.venv_path}")
    elif python != sys.executable:
        print(f"Using uv python: {python}")
    else:
        print("Using system python (consider using --venv-path or installing uv)")

    cfg = ["--config", args.config]

    run("parse_dataset.py", python, *cfg)
    run("preview_annotation.py", python, *cfg)

    if not args.yes:
        print("\nReview preview_grid.png in the working dir.")
        if input("Continue with full annotation? [y/N] ").strip().lower() not in ("y", "yes"):
            raise SystemExit("Aborted by user after preview.")

    run("annotate_dataset.py", python, *cfg)
    run("dataset_qa.py", python, *cfg)

    if not args.no_review:
        try:
            run("launch_fiftyone.py", python, *cfg, "--no-wait")
        except SystemExit as exc:
            print(f"(FiftyOne review skipped: {exc})")

    run("train_yolo.py", python, *cfg)
    run("evaluate.py", python, *cfg)
    run("failure_mining.py", python, *cfg)
    run("generate_report.py", python, *cfg)

    print("\n" + "=" * 60)
    print("Pipeline complete. Outputs are in the working dir.")
    print("=" * 60)

    # Print report path
    config_data = json.loads(Path(args.config).read_text()) if Path(args.config).exists() else {}
    wd = config_data.get("working_dir", ".")
    report_path = Path(wd) / "report.md"
    print(f"\n  Report:     file://{report_path.resolve()}")

    # Launch web interface in background
    if not args.no_dashboard:
        try:
            run("launch_dashboard.py", python, *cfg, "--background")
            print(f"  Web interface:  http://localhost:7860")
        except SystemExit as exc:
            print(f"(Dashboard skipped: {exc})")

    # Launch Voxel51 review in background (unless --no-review)
    if not args.no_review:
        try:
            run("launch_fiftyone.py", python, *cfg, "--split", "val")
            print(f"  Voxel51:   http://localhost:5151")
        except SystemExit as exc:
            print(f"(FiftyOne skipped: {exc})")

    print()


if __name__ == "__main__":
    main()
