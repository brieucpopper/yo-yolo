---
name: yo-yolo
description: 'Build Your Own YOLO (YO-YOLO). Turn a folder of images + a natural-language object description into a trained, deployable YOLO detector — auto-labeled by NVIDIA Locate Anything 3B as a teacher. USE WHEN the user wants to: create/train a custom object detector, auto-annotate/auto-label images from a text prompt, distill a vision-language model into a fast YOLO, build a dataset from a description, or run the YO-YOLO pipeline (parse, preview, annotate, QA, review, train, evaluate, failure mining, report, dashboard). Keywords: YOLO, object detection, auto-annotation, auto-labeling, Locate Anything, teacher-student, dataset bootstrapping, train detector, Ultralytics, FiftyOne, Gradio.'
argument-hint: 'images dir + what to detect (e.g. "./photos, detect ripe bananas")'
---

# YO-YOLO — Your Own YOLO

Orchestrate a 10-phase pipeline that converts a natural-language object
description + a folder of images into a trained YOLO detector, with optional
review, evaluation, failure analysis, and an interactive dashboard.

## Your Role (orchestrator)

You are an **intentionally simple orchestrator**. Your only jobs are:

1. Collect inputs into a config file.
2. Run the stage scripts in order.
3. Show intermediate results to the user.
4. Pause for approval at the gates (preview, optional review).
5. Summarize at the end.

Do **not** write training/annotation code or make autonomous modeling
decisions. The scripts already do that. Only edit a script if the user asks for
behavior the scripts don't support (e.g. fancier dataset selection — see
[parse_dataset.py](./scripts/parse_dataset.py)).

**Use uv for everything python related**, e.g. `uv run xx.py`
If the working directory does not have uv, then create a uv environment with python 3.13, and install these requirements

**Virtual environment reuse:** Use `--venv-path /path/to/venv` with `build_detector.py` to reuse an existing uv environment from a previous YO-YOLO project. If not provided, uv will be used if available, otherwise system python.

```# Core
pillow>=10.0
numpy>=1.24
pyyaml>=6.0
tqdm>=4.66
requests>=2.31
matplotlib>=3.7
opencv-python>=4.8

# Teacher model (local backend). Install a CUDA build of torch that matches
# your GPU/driver; the endpoint backend does not need torch.
torch>=2.1
transformers>=4.44

# Student detector + training
ultralytics>=8.3

# Optional review + dashboard
fiftyone>=0.24
gradio>=4.0
```

**Communicate progress clearly.** The scripts print `tqdm` progress bars and a
final machine-readable line `YOYOLO_SUMMARY {json}`. After each stage, parse
that JSON and tell the user what happened (counts, latency, metrics, files
produced) and what comes next. Surface any difficulties (empty annotations,
missing GPU, install errors) plainly.

## Step 0 — Collect config

Ask the user for the following — in this order, at the very start:

1. **`images_dir`** — folder of raw images (**required, ask first**)
2. **`class_description`** — natural-language description of what to detect (**required**)
3. **`working_dir`** — where outputs go (default `./yo_yolo_{current_datetime}`)
4. Teacher access — `localhost:port` (default) or `locate_anything_endpoint`
5. Optional: `max_images`, `review_after_annotation`

Create the `working_dir`, then create a `config.yaml` by copying
[config.example.yaml](./config.example.yaml) and filling in the values.
The short `class_name` is derived automatically from the description
(e.g. "A ripe yellow banana fruit" → `banana`); only set it explicitly if the
user wants a specific name.

**Multi-class:** `class_description` may be a YAML list; `class_name` a
matching list. Each (model, class_id) pair renders in its own solid color —
warm palette for teacher, green palette for YOLO, blue/purple for orchestrator.

## Step-by-step pipeline

Run each from the skill root directory. All scripts accept `--config <path>` as
their only required argument. After each stage, parse the `YOYOLO_SUMMARY` line
and report to the user.

| # | Phase | Command | Gate |
|---|-------|---------|------|
| 1 | Parse dataset | `uv run python scripts/parse_dataset.py --config <config.yaml>` | — |
| 2 | Preview (4 imgs) | `uv run python scripts/preview_annotation.py --config <config.yaml>` | **show grid, ask approval** |
| 3 | Full annotation | `uv run python scripts/annotate_dataset.py --config <config.yaml>` | — |
| 4 | Dataset QA | `uv run python scripts/dataset_qa.py --config <config.yaml>` | flag issues |
| 5 | Review (optional) | `uv run python scripts/launch_fiftyone.py --config <config.yaml>` | only if `review_after_annotation` |
| 6 | Train YOLO | `uv run python scripts/train_yolo.py --config <config.yaml>` | — |
| 7 | Evaluate | `uv run python scripts/evaluate.py --config <config.yaml>` | report metrics |
| 8 | Failure mining | `uv run python scripts/failure_mining.py --config <config.yaml>` | — |
| 9 | Report | `uv run python scripts/generate_report.py --config <config.yaml>` | — |
| 10 | Dashboard + Voxel | see Phase 10 section below | — |

### Phase 2 gate — preview approval (important)

After phase 2, the script writes `preview_grid.png` in the working dir. **Show
it to the user** and explicitly ask **Continue** or **Abort**. This prevents
annotating an entire dataset with a bad prompt. If boxes look wrong, help the
user refine `class_description` and re-run phase 2 before continuing.

**Always print the full absolute path to the preview grid image** (e.g.
`/abs/path/to/yo_yolo_.../preview_grid.png`) so the user can open it directly.

### Phase 5 gate — optional review

Only run if `review_after_annotation` is true. FiftyOne is long-running; launch
it, tell the user the URL, and continue when they're done. Skip gracefully if
FiftyOne isn't installed.

### Phase 6 — training

Long-running. Pass through user overrides like `--epochs`, `--batch`,
`--device 0`. Report duration and where `best.pt` landed.

### Phase 10 — dashboard + Voxel51 (both in background)

Launch **both** servers in the background. Parse each `YOYOLO_SUMMARY` to
confirm they started, then print the URLs and report path explicitly and stop.
Do NOT block on the dashboard.

```bash
# 1. FiftyOne — background, port 5151
uv run python scripts/launch_fiftyone.py --config <config.yaml> \
    --background --port 5151 --split val

# 2. Gradio dashboard — background, port 7860
uv run python scripts/launch_dashboard.py --config <config.yaml> \
    --background --port 7860
```

After both commands return, wait ~4 s for the servers to bind, then tell the
user:

```
Pipeline complete.

Report:    <absolute path to report.md>
FiftyOne:  http://localhost:5151
Dashboard: http://localhost:7860

To stop:
  kill $(cat <working_dir>/logs/fiftyone.pid)
  kill $(cat <working_dir>/logs/dashboard.pid)
```

Both processes are detached (`start_new_session=True`) and will survive after
the orchestrator exits.

### After pipeline completion

Always print these three items explicitly:
- Full absolute path to `report.md`
- FiftyOne URL: **http://localhost:5151**
- Gradio dashboard URL: **http://localhost:7860**

## One-shot alternative

If the user just wants everything to run without manual gating:

```bash
python build_detector.py --config config.yaml --yes
```

Flags: `--yes` (skip preview prompt), `--no-review`, `--no-dashboard`.

At the end, `build_detector.py` prints the report path and launches both
servers in the background.

## Outputs (in `working_dir`)

```
manifest.json · preview_grid.png · dataset/ (images,labels,metadata,data.yaml)
qa_report.md · qa_stats.json · best.pt · last.pt · results.csv
eval/metrics.json · eval/predictions/ · failures/ · report.md
dashboard_assets/ · logs/
```

## Box color scheme

No dashed lines. Colors are solid and determined by `(model, class_id)`:

| Model | Class 0 | Class 1 | Class 2 | … |
|---|---|---|---|---|
| Teacher (Locate Anything) | red | orange | hot pink | warm palette |
| YOLO (student) | green | teal | lime | cool-green palette |
| Orchestrator (LLM) | blue | purple | sky blue | blue/purple palette |

## Troubleshooting

- **Empty annotations in preview** → prompt too specific/abstract; refine
  `class_description`, re-run phase 2.
- **CUDA / torch errors on local backend** → use `locate_anything_endpoint`
  instead, or install a matching CUDA torch build.
- **Orchestrator boxes look wrong** → verify the endpoint returns valid JSON.
  The orchestrator resizes the image to 1000×1000 and expects
  `[ymin, xmin, ymax, xmax]` relative to that grid.

If needed (by default it should not be needed, just call the existing scripts, you can see) [references/teacher_prompts.md](./references/teacher_prompts.md) for how the teacher prompts and output format work.
