"""LocateAnythingWorker — teacher model wrapper for NVIDIA Locate Anything 3B.

Supports two backends:

* ``local``    — loads the model in-process with ``transformers`` (needs a GPU).
                 This is the default and the only backend that supports the
                 ``generation_mode`` argument ("fast" | "slow" | "hybrid").
* ``endpoint`` — talks to an OpenAI-compatible HTTP server (e.g. a llama.cpp
                 server running on a GPU box). ``generation_mode`` is ignored.

Both backends expose the same convenience methods (``ground_multi``, ``detect``,
...) and the same static parsers (``parse_boxes`` / ``parse_points``).

Every ``predict`` call records its wall-clock latency in the returned dict under
the key ``latency_ms`` so downstream stages can build the speed comparison.
"""

from __future__ import annotations

import base64
import io
import re
import time
from typing import Optional

from PIL import Image

# ``torch`` / ``transformers`` are only needed for the local backend. Import
# lazily so the endpoint backend (and lightweight tooling such as the dashboard)
# can run on a machine without a GPU stack installed.
try:  # pragma: no cover - import guard
    import torch
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment]


_BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
_POINT_RE = re.compile(r"<box><(\d+)><(\d+)></box>")


class LocateAnythingWorker:
    """Stateful worker that serves perception queries from a single model."""

    def __init__(
        self,
        model_path: str = "nvidia/Locate-Anything-3B",
        backend: str = "local",
        device: str = "cuda",
        dtype: str = "bfloat16",
        endpoint: Optional[str] = None,
        endpoint_model: str = "locate-anything",
        request_timeout: int = 300,
    ):
        self.backend = backend
        self.device = device
        self.model_path = model_path
        self.endpoint = endpoint
        self.endpoint_model = endpoint_model
        self.request_timeout = request_timeout

        if backend == "local":
            self._init_local(model_path, device, dtype)
        elif backend == "endpoint":
            if not endpoint:
                raise ValueError("backend='endpoint' requires an `endpoint` URL")
            self.endpoint = self._normalize_endpoint(endpoint)
        else:
            raise ValueError(f"Unknown backend: {backend!r} (use 'local' or 'endpoint')")

    # ------------------------------------------------------------------ setup

    def _init_local(self, model_path: str, device: str, dtype: str) -> None:
        if torch is None:
            raise RuntimeError(
                "The local backend requires torch + transformers. Install them or "
                "use backend='endpoint'."
            )
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = (
            AutoModel.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )
            .to(device)
            .eval()
        )

    @staticmethod
    def _normalize_endpoint(endpoint: str) -> str:
        """Accept ``host:port`` or full URL and return a chat-completions URL."""
        ep = endpoint.strip()
        if not ep.startswith(("http://", "https://")):
            ep = f"http://{ep}"
        ep = ep.rstrip("/")
        if not ep.endswith("/chat/completions"):
            if ep.endswith("/v1"):
                ep = f"{ep}/chat/completions"
            else:
                ep = f"{ep}/v1/chat/completions"
        return ep

    # ---------------------------------------------------------------- predict

    def predict(
        self,
        image: Image.Image,
        question: str,
        generation_mode: str = "hybrid",
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        verbose: bool = False,
    ) -> dict:
        """Run a single perception query and return ``{answer, latency_ms, ...}``."""
        start = time.perf_counter()
        if self.backend == "local":
            answer = self._predict_local(
                image, question, generation_mode, max_new_tokens, temperature, verbose
            )
        else:
            answer = self._predict_endpoint(image, question, max_new_tokens, temperature)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {"answer": answer, "latency_ms": latency_ms}

    def _predict_local(
        self,
        image: Image.Image,
        question: str,
        generation_mode: str,
        max_new_tokens: int,
        temperature: float,
        verbose: bool,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)

        pixel_values = inputs["pixel_values"].to(self.dtype)
        input_ids = inputs["input_ids"]
        image_grid_hws = inputs.get("image_grid_hws", None)

        with torch.no_grad():
            response = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=inputs["attention_mask"],
                image_grid_hws=image_grid_hws,
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                generation_mode=generation_mode,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                verbose=verbose,
            )

        return response[0] if isinstance(response, tuple) else response

    def _predict_endpoint(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        import requests  # local import keeps the module import-light

        # Resize large images so base64 doesn't blow the server context window
        max_dim = 1000
        if max(image.width, image.height) > max_dim:
            ratio = max_dim / max(image.width, image.height)
            image = image.resize(
                (int(image.width * ratio), int(image.height * ratio)),
                Image.LANCZOS,
            )

        data_url = f"data:image/png;base64,{self._encode_image(image)}"
        payload = {
            "model": self.endpoint_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": question},
                    ],
                }
            ],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        resp = requests.post(self.endpoint, json=payload, timeout=self.request_timeout)
        resp.raise_for_status()
        body = resp.json()
        return body["choices"][0]["message"]["content"]

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    # ---- Convenience methods for each task -------------------------------

    def detect(self, image: Image.Image, categories: list[str], **kwargs) -> dict:
        """Object detection / document layout analysis."""
        cats = "</c>".join(categories)
        prompt = f"Locate all the instances that matches the following description: {cats}."
        return self.predict(image, prompt, **kwargs)

    def ground_single(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Phrase grounding — single instance."""
        prompt = f"Locate a single instance that matches the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def ground_multi(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Phrase grounding — multiple instances."""
        prompt = f"Locate all the instances that match the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def ground_text(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Text grounding."""
        prompt = f"Please locate the text referred as {phrase}."
        return self.predict(image, prompt, **kwargs)

    def detect_text(self, image: Image.Image, **kwargs) -> dict:
        """Scene text detection."""
        prompt = "Detect all the text in box format."
        return self.predict(image, prompt, **kwargs)

    def point(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Pointing."""
        prompt = f"Point to: {phrase}."
        return self.predict(image, prompt, **kwargs)

    # ---- Utility: parse model output -------------------------------------

    @staticmethod
    def parse_boxes(answer: str, image_width: int, image_height: int) -> list[dict]:
        """Parse model output into pixel-coordinate bounding boxes.

        Coordinates in model output are normalized integers in [0, 1000].
        For multi-class, format is <box><class_id><x1><y1><x2><y2></box>
        For single-class, format is <box><x1><y1><x2><y2></box>
        """
        boxes = []
        for m in _BOX_RE.finditer(answer):
            groups = m.groups()
            if len(groups) == 5:
                class_id, x1, y1, x2, y2 = (int(g) for g in groups)
            elif len(groups) == 4:
                class_id = 0
                x1, y1, x2, y2 = (int(g) for g in groups)
            else:
                continue
            boxes.append(
                {
                    "class_id": class_id,
                    "x1": x1 / 1000 * image_width,
                    "y1": y1 / 1000 * image_height,
                    "x2": x2 / 1000 * image_width,
                    "y2": y2 / 1000 * image_height,
                }
            )
        return boxes

    @staticmethod
    def parse_points(answer: str, image_width: int, image_height: int) -> list[dict]:
        """Parse model output into pixel-coordinate points."""
        points = []
        for m in _POINT_RE.finditer(answer):
            x, y = int(m.group(1)), int(m.group(2))
            points.append(
                {
                    "x": x / 1000 * image_width,
                    "y": y / 1000 * image_height,
                }
            )
        return points


def worker_from_config(config: dict) -> LocateAnythingWorker:
    """Build a :class:`LocateAnythingWorker` from a resolved config dict."""
    teacher = config.get("teacher", {}) if isinstance(config.get("teacher"), dict) else {}
    backend = teacher.get("backend") or ("endpoint" if config.get("locate_anything_endpoint") else "local")
    return LocateAnythingWorker(
        model_path=teacher.get("model_path", "nvidia/Locate-Anything-3B"),
        backend=backend,
        device=teacher.get("device", "cuda"),
        dtype=teacher.get("dtype", "bfloat16"),
        endpoint=config.get("locate_anything_endpoint") or teacher.get("endpoint"),
        endpoint_model=teacher.get("endpoint_model", "locate-anything"),
        request_timeout=teacher.get("request_timeout", 300),
    )
