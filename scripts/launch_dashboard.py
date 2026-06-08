"""Phase 10 — Interactive dashboard (Gradio).

Four tabs:

1. Live Inference — upload an image, run Locate Anything / YOLO / an optional
   "Orchestrator" LLM baseline, with a always-visible speed panel.
2. Validation Browser — page through validation images with toggleable teacher
   and YOLO overlays.
3. Failure Gallery — the top disagreement cases mined in phase 8.
4. Training Report — renders report.md inline.

Usage:
    python scripts/launch_dashboard.py --config config.yaml [--share]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (  # noqa: E402
    draw_boxes,
    get_logger,
    load_config,
    read_json,
    resolve_config,
    working_paths,
)


class DashboardState:
    """Lazily-loaded models + cached artifacts for the dashboard."""

    def __init__(self, config: dict, paths: dict, log):
        self.config = config
        self.paths = paths
        self.log = log
        self.class_names = config.get("class_names", [config.get("class_name", "object")])
        self.class_name = self.class_names[0]
        self.class_descriptions = config.get("class_descriptions", [config.get("class_description", self.class_name)])
        self.phrase = self.class_descriptions[0]
        self._teacher = None
        self._yolo = None
        self.val_images = sorted(p for p in paths["images_val"].iterdir() if p.is_file()) \
            if paths["images_val"].exists() else []

    @property
    def teacher(self):
        if self._teacher is None:
            from locate_anything import worker_from_config
            self.log.info("Loading teacher for dashboard...")
            self._teacher = worker_from_config(self.config)
        return self._teacher

    @property
    def yolo(self):
        if self._yolo is None:
            from ultralytics import YOLO
            weights = self.paths["root"] / "best.pt"
            self._yolo = YOLO(str(weights)) if weights.exists() else None
        return self._yolo


def run_teacher(state: DashboardState, image: Image.Image):
    from locate_anything import LocateAnythingWorker

    gen_mode = state.config.get("teacher", {}).get("generation_mode", "fast")
    kwargs = {"generation_mode": gen_mode} if state.teacher.backend == "local" else {}
    res = state.teacher.detect(image, state.class_descriptions, **kwargs)
    boxes = LocateAnythingWorker.parse_boxes(res["answer"], image.width, image.height)
    xyxy = []
    labels = []
    for b in boxes:
        xyxy.append([b["x1"], b["y1"], b["x2"], b["y2"]])
        cid = b.get("class_id", 0)
        labels.append(state.class_names[cid] if cid < len(state.class_names) else f"class_{cid}")
    return xyxy, labels, res["latency_ms"]


def run_yolo(state: DashboardState, image: Image.Image):
    if state.yolo is None:
        return [], None, None
    start = time.perf_counter()
    res = state.yolo.predict(image, verbose=False)[0]
    latency = (time.perf_counter() - start) * 1000.0
    boxes, labels = [], []
    for b in res.boxes:
        boxes.append(b.xyxy[0].tolist())
        cls_id = int(b.cls[0]) if hasattr(b, 'cls') and b.cls is not None else 0
        conf = float(b.conf[0])
        name = state.class_names[cls_id] if cls_id < len(state.class_names) else f"class_{cls_id}"
        labels.append(f"{name} {conf:.2f}")
    return boxes, labels, latency


def run_orchestrator(state: DashboardState, image: Image.Image):
    """Zero-shot bounding-box baseline via an optional OpenAI-compatible LLM."""
    dcfg = state.config.get("dashboard", {}) or {}
    endpoint = dcfg.get("orchestrator_endpoint")
    if not endpoint:
        return [], None
    import base64
    import io
    import re
    import requests

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    url = endpoint.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url + ("/chat/completions" if url.endswith("/v1") else "/v1/chat/completions")
    prompt = (
        f"Return bounding boxes for every {state.phrase} in the image as JSON "
        f"list of [x1,y1,x2,y2] in pixel coordinates. Image is "
        f"{image.width}x{image.height}. Only output JSON."
    )
    start = time.perf_counter()
    try:
        resp = requests.post(
            url,
            json={
                "model": dcfg.get("orchestrator_model", "gpt-4o-mini"),
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ]}],
                "max_tokens": 512,
            },
            timeout=120,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        nums = re.findall(r"-?\d+\.?\d*", text)
        boxes = [list(map(float, nums[i:i + 4])) for i in range(0, len(nums) - 3, 4)]
    except Exception as exc:  # noqa: BLE001
        state.log.warning("Orchestrator baseline failed: %s", exc)
        boxes = []
    latency = (time.perf_counter() - start) * 1000.0
    return boxes, latency


def build_app(state: DashboardState):
    import gradio as gr

    def infer(image, show_teacher, show_yolo, show_orch):
        if image is None:
            return None, "Upload an image."
        image = image.convert("RGB")
        vis = image.copy()
        speed = {}
        if show_teacher:
            boxes, labels, ms = run_teacher(state, image)
            vis = draw_boxes(vis, boxes, color=(255, 64, 64), labels=labels, dashed=True)
            speed["Locate Anything"] = f"{ms:.0f} ms"
        if show_yolo:
            boxes, labels, ms = run_yolo(state, image)
            if ms is not None:
                vis = draw_boxes(vis, boxes, color=(64, 200, 64), labels=labels)
                speed["YOLO"] = f"{ms:.0f} ms"
            else:
                speed["YOLO"] = "no weights"
        if show_orch:
            boxes, ms = run_orchestrator(state, image)
            if ms is not None:
                vis = draw_boxes(vis, boxes, color=(80, 120, 255))
                speed["Orchestrator"] = f"{ms:.0f} ms"
            else:
                speed["Orchestrator"] = "disabled"
        panel = "\n".join(f"{k}: {v}" for k, v in speed.items()) or "Nothing selected."
        return vis, panel

    def show_val(idx, show_teacher, show_yolo):
        if not state.val_images:
            return None, "No validation images."
        idx = int(idx) % len(state.val_images)
        img_path = state.val_images[idx]
        image = Image.open(img_path).convert("RGB")
        pred_path = state.paths["predictions"] / f"{img_path.stem}.json"
        vis = image.copy()
        if pred_path.exists():
            pred = read_json(pred_path)
            if show_teacher:
                t_boxes = pred.get("teacher_boxes", [])
                t_xyxy = [b["xyxy"] if isinstance(b, dict) and "xyxy" in b else b for b in t_boxes]
                vis = draw_boxes(vis, t_xyxy, color=(255, 64, 64), dashed=True)
            if show_yolo:
                y_boxes = pred.get("yolo_boxes", [])
                y_xyxy = [b["xyxy"] for b in y_boxes]
                y_labels = []
                for b in y_boxes:
                    cid = b.get("class_id", 0)
                    conf = b.get("confidence", 0)
                    name = state.class_names[cid] if cid < len(state.class_names) else f"class_{cid}"
                    y_labels.append(f"{name} {conf:.2f}")
                vis = draw_boxes(vis, y_xyxy, color=(64, 200, 64), labels=y_labels)
        return vis, f"{idx + 1} / {len(state.val_images)} — {img_path.name}"

    def load_failures():
        idx_path = state.paths["failures"] / "index.json"
        if not idx_path.exists():
            return []
        return [(f["render"], f"#{f['rank']} {f['reason']}") for f in read_json(idx_path)]

    report_md = state.paths["report"].read_text() if state.paths["report"].exists() else "_No report yet._"

    with gr.Blocks(title="YO-YOLO Dashboard") as app:
        class_display = ", ".join(state.class_names)
        gr.Markdown(f"# YO-YOLO — `{class_display}`")

        with gr.Tab("Live Inference"):
            with gr.Row():
                with gr.Column():
                    inp = gr.Image(type="pil", label="Upload image")
                    t_tog = gr.Checkbox(value=True, label="Locate Anything (teacher)")
                    y_tog = gr.Checkbox(value=True, label="YOLO (student)")
                    o_tog = gr.Checkbox(value=False, label="Orchestrator (LLM baseline)")
                    btn = gr.Button("Run", variant="primary")
                with gr.Column():
                    out = gr.Image(type="pil", label="Predictions")
                    speed = gr.Textbox(label="Speed panel", lines=4)
            btn.click(
                infer,
                [inp, t_tog, y_tog, o_tog],
                [out, speed],
                show_progress="full"
            )

        with gr.Tab("Validation Browser"):
            with gr.Row():
                vt = gr.Checkbox(value=True, label="Teacher Ground Truth")
                vy = gr.Checkbox(value=True, label="YOLO Predictions")
            state_idx = gr.State(0)
            vout = gr.Image(type="pil", label="Validation image")
            vcap = gr.Textbox(label="Image", interactive=False)
            with gr.Row():
                prev = gr.Button("← Previous", variant="secondary")
                nxt = gr.Button("Next →", variant="secondary")

            def step(idx, delta, st, sy):
                idx = (int(idx) + delta) % max(len(state.val_images), 1)
                img, cap = show_val(idx, st, sy)
                return idx, img, cap

            prev.click(
                lambda i, st, sy: step(i, -1, st, sy),
                [state_idx, vt, vy],
                [state_idx, vout, vcap],
                show_progress="minimal"
            )
            nxt.click(
                lambda i, st, sy: step(i, 1, st, sy),
                [state_idx, vt, vy],
                [state_idx, vout, vcap],
                show_progress="minimal"
            )
            app.load(lambda: show_val(0, True, True), None, [vout, vcap])

        with gr.Tab("Failure Gallery"):
            gallery = gr.Gallery(label="Top disagreements", columns=4, height=600)
            refresh = gr.Button("Load failures", variant="secondary")
            refresh.click(load_failures, None, gallery, show_progress="full")
            app.load(load_failures, None, gallery)

        with gr.Tab("Training Report"):
            gr.Markdown(report_md)

    app.queue()
    return app


def _default_port(config: dict) -> int:
    backend = config.get("teacher", {}).get("backend", "local")
    if backend == "endpoint":
        return 8080
    return 7860


def _check_port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def _get_process_on_port(port: int) -> str | None:
    import subprocess
    try:
        result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="YO-YOLO: launch the Gradio dashboard")
    ap.add_argument("--config", required=True)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--background", action="store_true", help="Run in background with nohup")
    ap.add_argument("--force", action="store_true", help="Kill existing process on port")
    args = ap.parse_args()

    config = resolve_config(load_config(args.config))
    paths = working_paths(config["working_dir"])
    log = get_logger("launch_dashboard", config["working_dir"])

    port = args.port if args.port is not None else _default_port(config)

    try:
        import gradio  # noqa: F401
    except ImportError:
        log.error("gradio is not installed. `pip install gradio`.")
        sys.exit(1)

    if _check_port_in_use(port):
        pid = _get_process_on_port(port)
        if args.force and pid:
            import os
            import signal
            os.kill(int(pid), signal.SIGTERM)
            log.info("Killed existing process %s on port %d", pid, port)
        elif not args.background:
            log.error("Port %d is in use (PID: %s). Use --force to kill or --port to change.", port, pid or "unknown")
            sys.exit(1)

    state = DashboardState(config, paths, log)
    app = build_app(state)
    share = args.share or (config.get("dashboard", {}) or {}).get("share", False)

    if args.background:
        import subprocess
        log_file = paths["logs"] / "dashboard.log"
        log.info("Launching dashboard in background on port %d (log: %s)", port, log_file)
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--config", args.config,
            "--port", str(port),
        ]
        if args.share:
            cmd.append("--share")
        with open(log_file, "w") as f:
            subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        log.info("Dashboard started in background. Check %s for output.", log_file)
    else:
        log.info("Launching dashboard on port %d (share=%s)", port, share)
        app.launch(server_port=port, share=share)


if __name__ == "__main__":
    main()
