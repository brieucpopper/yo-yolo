"""Phase 10 — Interactive dashboard (Gradio).

Single-tab: Live Inference — drag-and-drop an image, run Locate Anything /
YOLO / an optional "Orchestrator" LLM baseline, with an always-visible speed
panel.

Box colors:  teacher = warm palette (reds/oranges), YOLO = green palette,
             orchestrator = blue/purple palette.  Each class_id within a model
             gets a distinct shade — no dashes.

Orchestrator notes:
  - Image is resized to 1000×1000 before sending to the LLM.
  - Prompt asks for [ymin, xmin, ymax, xmax] relative to 1000×1000.
  - One request is issued per class_description; results are aggregated.

Teacher coordinate fix:
  - Image is resized ONCE in the infer() function before being dispatched to
    all models, so teacher and YOLO boxes are always in the same coordinate
    space as the output image.

Usage:
    python scripts/launch_dashboard.py --config config.yaml [--share] [--port 7860]
    python scripts/launch_dashboard.py --config config.yaml --background
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import threading
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (  # noqa: E402
    draw_boxes,
    get_logger,
    get_model_class_color,
    load_config,
    print_summary,
    resolve_config,
    working_paths,
)

_MAX_IMAGE_DIM = 2048
_ORCHESTRATOR_SIZE = 1000  # LLM always sees a 1000×1000 image


class DashboardState:
    """Lazily-loaded models for the dashboard."""

    def __init__(self, config: dict, paths: dict, log):
        self.config = config
        self.paths = paths
        self.log = log
        self.class_names = config.get("class_names", [config.get("class_name", "object")])
        self.class_descriptions = config.get(
            "class_descriptions", [config.get("class_description", self.class_names[0])]
        )
        self._teacher = None
        self._yolo = None
        self._teacher_lock = threading.Lock()
        self._yolo_lock = threading.Lock()
        self._model_errors: dict[str, str] = {}

        self.log.info("DashboardState init — classes: %s", self.class_names)
        self.log.info("Working dir: %s", paths["root"])

        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="yoyolo"
        )

        self.log.info("Submitting background preload for teacher model...")
        self._executor.submit(self._preload_teacher)
        weights_path = paths["root"] / "best.pt"
        if weights_path.exists():
            self.log.info("Found YOLO weights at %s — submitting background preload.", weights_path)
            self._executor.submit(self._preload_yolo)
        else:
            self.log.info("No YOLO weights found at %s — skipping YOLO preload.", weights_path)

    def _preload_teacher(self):
        self.log.info("[preload] Teacher model — starting background load...")
        try:
            _ = self.teacher
            self.log.info("[preload] Teacher model — ready.")
        except Exception as exc:
            self.log.error("[preload] Teacher model — FAILED: %s", exc)

    def _preload_yolo(self):
        self.log.info("[preload] YOLO model — starting background load...")
        try:
            _ = self.yolo
            self.log.info("[preload] YOLO model — ready.")
        except Exception as exc:
            self.log.error("[preload] YOLO model — FAILED: %s", exc)

    @property
    def teacher(self):
        with self._teacher_lock:
            if self._teacher is None:
                if "teacher" in self._model_errors:
                    self.log.warning("Retrying teacher load after previous error: %s", self._model_errors["teacher"])
                    self._model_errors.pop("teacher")
                try:
                    from locate_anything import worker_from_config
                    self.log.info("Loading teacher model...")
                    self._teacher = worker_from_config(self.config)
                    self.log.info("Teacher model ready.")
                except Exception as exc:
                    self._model_errors["teacher"] = str(exc)
                    self.log.error("Failed to load teacher: %s", exc)
                    raise
            return self._teacher

    @property
    def yolo(self):
        with self._yolo_lock:
            if "yolo" in self._model_errors:
                raise RuntimeError(f"YOLO failed to load: {self._model_errors['yolo']}")
            if self._yolo is None:
                weights = self.paths["root"] / "best.pt"
                if not weights.exists():
                    self.log.info("No YOLO weights at %s", weights)
                    return None
                try:
                    from ultralytics import YOLO
                    self.log.info("Loading YOLO weights from %s...", weights)
                    self._yolo = YOLO(str(weights))
                    self.log.info("YOLO model ready.")
                except Exception as exc:
                    self._model_errors["yolo"] = str(exc)
                    self.log.error("Failed to load YOLO: %s", exc)
                    raise
            return self._yolo

    def shutdown(self):
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self._executor.shutdown(wait=False)


def _resize_image(image: Image.Image, max_dim: int = _MAX_IMAGE_DIM) -> Image.Image:
    w, h = image.size
    if max(w, h) <= max_dim:
        return image
    scale = max_dim / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def run_teacher(state: DashboardState, image: Image.Image):
    """Run Locate Anything on image (already resized by caller).

    Returns (boxes_xyxy, class_ids, labels, latency_ms).
    """
    from locate_anything import LocateAnythingWorker

    state.log.info("[teacher] Running inference — image %dx%d...", image.width, image.height)
    t0 = time.perf_counter()
    gen_mode = state.config.get("teacher", {}).get("generation_mode", "fast")
    kwargs = {"generation_mode": gen_mode} if state.teacher.backend == "local" else {}
    res = state.teacher.detect(image, state.class_descriptions, **kwargs)
    boxes_raw = LocateAnythingWorker.parse_boxes(res["answer"], image.width, image.height)
    boxes, class_ids, labels = [], [], []
    for b in boxes_raw:
        boxes.append([b["x1"], b["y1"], b["x2"], b["y2"]])
        cid = b.get("class_id", 0)
        class_ids.append(cid)
        labels.append(state.class_names[cid] if cid < len(state.class_names) else f"class_{cid}")
    state.log.info(
        "[teacher] Done — %d box(es), latency=%.0f ms (wall=%.0f ms)",
        len(boxes), res["latency_ms"], (time.perf_counter() - t0) * 1000,
    )
    return boxes, class_ids, labels, res["latency_ms"]


def run_yolo(state: DashboardState, image: Image.Image):
    """Run YOLO on image (already resized by caller).

    Returns (boxes_xyxy, class_ids, labels, latency_ms).
    """
    if state.yolo is None:
        state.log.info("[yolo] No weights loaded — skipping.")
        return [], [], [], None
    state.log.info("[yolo] Running inference — image %dx%d...", image.width, image.height)
    start = time.perf_counter()
    res = state.yolo.predict(image, verbose=False)[0]
    latency = (time.perf_counter() - start) * 1000.0
    boxes, class_ids, labels = [], [], []
    for b in res.boxes:
        boxes.append(b.xyxy[0].tolist())
        cid = int(b.cls[0]) if hasattr(b, "cls") and b.cls is not None else 0
        conf = float(b.conf[0])
        name = state.class_names[cid] if cid < len(state.class_names) else f"class_{cid}"
        class_ids.append(cid)
        labels.append(f"{name} {conf:.2f}")
    state.log.info("[yolo] Done — %d box(es), latency=%.0f ms", len(boxes), latency)
    return boxes, class_ids, labels, latency


def run_orchestrator(state: DashboardState, image: Image.Image):
    """Zero-shot bbox baseline via an optional OpenAI-compatible LLM.

    The image is resized to 1000×1000 before being sent. One request is issued
    per class_description. The model is instructed to return
    [ymin, xmin, ymax, xmax] relative to 1000×1000. Coordinates are scaled
    back to the actual image dimensions.

    Returns (boxes_xyxy, class_ids, labels, latency_ms) or ([], [], [], None).
    """
    dcfg = state.config.get("dashboard", {}) or {}
    endpoint = dcfg.get("orchestrator_endpoint")
    if not endpoint:
        state.log.info("[orchestrator] No endpoint configured — skipping.")
        return [], [], [], None

    import base64
    import io
    import requests

    img_w, img_h = image.width, image.height

    # Encode the 1000×1000 version once for all class calls
    img_1k = image.resize((_ORCHESTRATOR_SIZE, _ORCHESTRATOR_SIZE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img_1k.convert("RGB").save(buf, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    url = endpoint.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions" if url.endswith("/v1") else "/v1/chat/completions"

    _connect_timeout = 10
    _read_timeout = int(dcfg.get("orchestrator_timeout", 30))
    model_id = dcfg.get("orchestrator_model", "gpt-4o-mini")

    all_boxes, all_class_ids, all_labels = [], [], []
    total_start = time.perf_counter()

    for class_id, (desc, class_name) in enumerate(
        zip(state.class_descriptions, state.class_names)
    ):
        prompt = (
            f"Return bounding boxes around every {desc} in this image. "
            f"For each one return [ymin, xmin, ymax, xmax] where each value is "
            f"an integer relative to a {_ORCHESTRATOR_SIZE}x{_ORCHESTRATOR_SIZE} image. "
            f"Output ONLY a JSON array of arrays, e.g. [[100, 200, 300, 400], ...]. "
            f"No explanation, no markdown, just the JSON array."
        )
        try:
            state.log.info(
                "[orchestrator] Calling for class '%s' — image %dx%d (sent as %dx%d)...",
                class_name, img_w, img_h, _ORCHESTRATOR_SIZE, _ORCHESTRATOR_SIZE,
            )
            resp = requests.post(
                url,
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ]}],
                    "max_tokens": 512,
                },
                timeout=(_connect_timeout, _read_timeout),
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]

            # Parse JSON array, falling back to regex on failure
            boxes_1k = []
            try:
                m = re.search(r"\[.*\]", text, re.DOTALL)
                if m:
                    parsed = json.loads(m.group())
                    for item in parsed:
                        if isinstance(item, (list, tuple)) and len(item) >= 4:
                            boxes_1k.append([float(v) for v in item[:4]])
            except Exception:
                nums = re.findall(r"-?\d+\.?\d*", text)
                boxes_1k = [
                    [float(nums[i]), float(nums[i+1]), float(nums[i+2]), float(nums[i+3])]
                    for i in range(0, len(nums) - 3, 4)
                ]

            # Scale [ymin, xmin, ymax, xmax] from 1000×1000 → actual image dims
            S = float(_ORCHESTRATOR_SIZE)
            for b in boxes_1k:
                ymin, xmin, ymax, xmax = b
                x1 = xmin / S * img_w
                y1 = ymin / S * img_h
                x2 = xmax / S * img_w
                y2 = ymax / S * img_h
                all_boxes.append([x1, y1, x2, y2])
                all_class_ids.append(class_id)
                all_labels.append(class_name)

            state.log.info(
                "[orchestrator] class '%s' → %d box(es)", class_name, len(boxes_1k)
            )
        except Exception as exc:
            state.log.warning("[orchestrator] Failed for class '%s': %s", class_name, exc)

    latency = (time.perf_counter() - total_start) * 1000.0
    state.log.info(
        "[orchestrator] Done — %d total box(es), latency=%.0f ms", len(all_boxes), latency
    )
    return all_boxes, all_class_ids, all_labels, latency


def _hex(rgb: tuple) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def build_app(state: DashboardState):
    import gradio as gr

    T_COLOR = get_model_class_color("teacher", 0)
    Y_COLOR = get_model_class_color("yolo", 0)
    O_COLOR = get_model_class_color("orch", 0)
    T_HEX, Y_HEX, O_HEX = _hex(T_COLOR), _hex(Y_COLOR), _hex(O_COLOR)

    def _composite(data, show_teacher, show_yolo, show_orch):
        if data is None:
            return None
        image = data["image"].copy()
        for name, show in [("teacher", show_teacher), ("yolo", show_yolo), ("orch", show_orch)]:
            entry = data.get(name)
            if show and entry is not None:
                boxes, class_ids, labels, _ = entry
                if boxes:
                    colors = [get_model_class_color(name, cid) for cid in class_ids]
                    image = draw_boxes(image, boxes, color=colors, labels=labels)
        return image

    def infer(image, run_t, run_y, run_o):
        if image is None:
            return None, "", None

        image = _resize_image(image.convert("RGB"))
        state.log.info("[infer] Request — image %dx%d  models: teacher=%s yolo=%s orch=%s",
                       image.width, image.height, run_t, run_y, run_o)
        speed: dict[str, str] = {}

        futures: dict[str, concurrent.futures.Future] = {}
        if run_t:
            futures["teacher"] = state._executor.submit(run_teacher, state, image)
        if run_y:
            futures["yolo"] = state._executor.submit(run_yolo, state, image)
        if run_o:
            futures["orch"] = state._executor.submit(run_orchestrator, state, image)

        if not futures:
            state.log.info("[infer] No models selected.")
            data = {"image": image}
            return _composite(data, True, True, True), "Nothing selected.", data

        results: dict[str, object] = {}
        future_to_name = {v: k for k, v in futures.items()}
        try:
            for future in concurrent.futures.as_completed(future_to_name, timeout=300):
                name = future_to_name[future]
                try:
                    results[name] = future.result()
                    state.log.info("[infer] %s completed.", name)
                except Exception as exc:
                    state.log.warning("[infer] %s inference failed: %s", name, exc)
                    speed[name] = f"error: {exc}"
        except concurrent.futures.TimeoutError:
            state.log.warning("[infer] Timed out — cancelling remaining futures.")
            for name, fut in futures.items():
                if name not in results:
                    fut.cancel()
                    speed[name] = "timeout"

        data: dict = {"image": image}
        for name in futures:
            if name in results:
                data[name] = results[name]
                boxes, _, _, ms = results[name]
                if name == "teacher":
                    speed["Locate Anything"] = f"{ms:.0f} ms" if ms is not None else "done"
                elif name == "yolo":
                    speed["YOLO"] = f"{ms:.0f} ms" if ms is not None else "no weights"
                elif name == "orch":
                    speed["Orchestrator"] = f"{ms:.0f} ms" if ms is not None else "disabled"

        vis = _composite(data, True, True, True)
        panel = "\n".join(f"{k}: {v}" for k, v in speed.items()) or "No results."
        state.log.info("[infer] Done — %s", panel.replace("\n", "  "))
        return vis, panel, data

    def on_vis_change(data, show_teacher, show_yolo, show_orch):
        return _composite(data, show_teacher, show_yolo, show_orch)

    with gr.Blocks(title="YO-YOLO Dashboard", css="""
        .legend-row { display: flex; gap: 12px; margin: 8px 0; flex-wrap: wrap; }
        .legend-item { display: flex; align-items: center; gap: 4px; }
        .legend-swatch { display: inline-block; width: 14px; height: 14px; border-radius: 3px; }
    """) as app:
        class_display = ", ".join(state.class_names)
        gr.Markdown(f"# YO-YOLO — `{class_display}`")

        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Upload image")

                gr.Markdown("**Models to run**")
                cb_run_t = gr.Checkbox(value=True,  label="Locate Anything (teacher)")
                cb_run_y = gr.Checkbox(value=True,  label="YOLO (student)")
                cb_run_o = gr.Checkbox(value=False, label="Orchestrator (LLM baseline)")
                btn = gr.Button("Run", variant="primary")

            with gr.Column():
                out = gr.Image(type="pil", label="Predictions")

                gr.Markdown("**Show/hide predictions**")
                with gr.Row(elem_classes="legend-row"):
                    gr.HTML(
                        f'<span class="legend-item">'
                        f'<span class="legend-swatch" style="background:{T_HEX}"></span>'
                        f'Locate Anything</span>'
                    )
                    cb_show_t = gr.Checkbox(value=True, label="Click to show/hide bbox", min_width=40)
                with gr.Row(elem_classes="legend-row"):
                    gr.HTML(
                        f'<span class="legend-item">'
                        f'<span class="legend-swatch" style="background:{Y_HEX}"></span>'
                        f'YOLO</span>'
                    )
                    cb_show_y = gr.Checkbox(value=True, label="Click to show/hide bbox", min_width=40)
                with gr.Row(elem_classes="legend-row"):
                    gr.HTML(
                        f'<span class="legend-item">'
                        f'<span class="legend-swatch" style="background:{O_HEX}"></span>'
                        f'Orchestrator</span>'
                    )
                    cb_show_o = gr.Checkbox(value=True, label="Click to show/hide bbox", min_width=40)

                speed_box = gr.Textbox(label="Speed panel", lines=4)

        result_state = gr.State(None)

        btn.click(
            infer,
            [inp, cb_run_t, cb_run_y, cb_run_o],
            [out, speed_box, result_state],
            show_progress="full",
        )

        for toggle in [cb_show_t, cb_show_y, cb_show_o]:
            toggle.change(
                on_vis_change,
                [result_state, cb_show_t, cb_show_y, cb_show_o],
                out,
            )

    app.queue(default_concurrency_limit=4)
    return app


def _default_port(config: dict) -> int:
    return 7860


def _check_port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("localhost", port)) == 0


def _get_process_on_port(port: int) -> str | None:
    import subprocess
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
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
    ap.add_argument("--background", action="store_true", help="Spawn a detached background server and return")
    ap.add_argument("--force", action="store_true", help="Kill existing process on port before starting")
    args = ap.parse_args()

    config = resolve_config(load_config(args.config))
    paths = working_paths(config["working_dir"])
    log = get_logger("launch_dashboard", config["working_dir"])

    port = args.port if args.port is not None else _default_port(config)

    try:
        import gradio as gradio_mod
        log.info("Gradio version: %s", gradio_mod.__version__)
    except ImportError:
        log.error("gradio is not installed. `pip install gradio`.")
        sys.exit(1)

    if args.background:
        import subprocess
        logs_dir = paths["logs"]
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "dashboard.log"
        pid_file = logs_dir / "dashboard.pid"
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--config", args.config,
            "--port", str(port),
        ]
        if args.share:
            cmd.append("--share")
        if args.force:
            cmd.append("--force")
        with open(log_file, "w") as f:
            proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        pid_file.write_text(str(proc.pid))
        log.info("Dashboard started in background (PID %d) on port %d.", proc.pid, port)
        log.info("URL:  http://localhost:%d", port)
        log.info("Log:  %s", log_file)
        log.info("Stop: kill $(cat %s)", pid_file)
        print_summary({
            "stage": "launch_dashboard",
            "launched": True,
            "background": True,
            "pid": proc.pid,
            "port": port,
            "url": f"http://localhost:{port}",
            "log": str(log_file),
            "pid_file": str(pid_file),
        })
        return

    if _check_port_in_use(port):
        pid = _get_process_on_port(port)
        if args.force and pid:
            import os
            import signal
            try:
                os.kill(int(pid), signal.SIGTERM)
                log.info("Sent SIGTERM to PID %s on port %d.", pid, port)
                for _ in range(10):
                    time.sleep(0.5)
                    if not _check_port_in_use(port):
                        break
                else:
                    os.kill(int(pid), signal.SIGKILL)
                    log.info("Sent SIGKILL to PID %s (still alive after SIGTERM).", pid)
                    time.sleep(0.5)
            except (ProcessLookupError, ValueError):
                pass
        else:
            log.error(
                "Port %d is in use (PID: %s). Use --force to kill or --port to change.",
                port, pid or "unknown",
            )
            sys.exit(1)

    log.info("Building dashboard state...")
    state = DashboardState(config, paths, log)
    log.info("Building Gradio app...")
    app = build_app(state)
    share = args.share or (config.get("dashboard", {}) or {}).get("share", False)

    log.info("Launching dashboard on port %d (share=%s).", port, share)
    log.info("allowed_paths: %s", config["working_dir"])
    try:
        app.launch(
            server_port=port,
            share=share,
            allowed_paths=[str(config["working_dir"])],
        )
        try:
            app.block()
        except AttributeError:
            log.info("app.block() not available — using threading.Event fallback.")
            threading.Event().wait()
    except KeyboardInterrupt:
        log.info("Keyboard interrupt received — shutting down.")
    finally:
        log.info("Dashboard shutting down — cleaning up executor.")
        state.shutdown()


if __name__ == "__main__":
    main()
