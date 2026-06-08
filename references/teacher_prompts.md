# Locate Anything — teacher prompts & output format

The teacher is **NVIDIA Locate Anything 3B**, wrapped by
[`LocateAnythingWorker`](../locate_anything.py). YO-YOLO uses **phrase
grounding (multi-instance)** for annotation.

## Primary call

```python
worker.ground_multi(image, class_description, generation_mode="fast")
```

which sends the prompt:

```
Locate all the instances that match the following description: <class_description>.
```

`generation_mode` (local backend only):

- `fast`   — MTP, fastest (default for annotation)
- `slow`   — NTP/AR, most accurate
- `hybrid` — balance

The endpoint backend ignores `generation_mode` (plain text generation).

## Other task prompts (available, not used by default)

| Method | Prompt shape |
|--------|--------------|
| `detect(cats)` | `Locate all the instances that matches the following description: a</c>b.` |
| `ground_single(phrase)` | `Locate a single instance that matches ...` |
| `ground_text(phrase)` | `Please locate the text referred as ...` |
| `detect_text()` | `Detect all the text in box format.` |
| `point(phrase)` | `Point to: ...` |

## Output format

The model emits boxes as normalized integers in `[0, 1000]`:

```
<box><x1><y1><x2><y2></box>
```

Parsed by `LocateAnythingWorker.parse_boxes(answer, width, height)` into pixel
`{x1, y1, x2, y2}` dicts. Points use `<box><x><y></box>` via `parse_points`.

## YOLO conversion

Pixel `xyxy` → normalized YOLO `class_id xc yc w h` via
`common.xyxy_to_yolo`. Single class, so `class_id` is always `0`.

## Backends

- **local** — `transformers` `AutoModel.from_pretrained(..., trust_remote_code=True)`
  on CUDA. Supports `generation_mode`.
- **endpoint** — OpenAI-compatible `/v1/chat/completions` with a base64
  `image_url` (e.g. a llama.cpp server). Set via `locate_anything_endpoint`
  (`host:port` or full URL).

Every `predict` call returns `latency_ms`, stored per image for the speed
comparison in the report and dashboard.
