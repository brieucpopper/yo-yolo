# YO-YOLO — Your Own YOLO

> Describe what you want to detect → auto-label with **NVIDIA Locate Anything 3B** → optionally review → train **YOLO** → inspect → deploy.

Turn a folder of images and a one-line object description into a fast,
deployable YOLO detector. A vision-language **teacher** (Locate Anything 3B)
auto-labels your images; a lightweight **YOLO11n student** learns from those
labels and runs orders of magnitude faster.

No manual annotation. No active learning. No complex agent planning — just
**describe → label → train → inspect.**

## Phases

| # | Phase | What it does |
|---|-------|--------------|
| 1 | Parse | Scan images, split train/val, derive a class name |
| 2 | Preview | Annotate 4 images so you can sanity-check the prompt |
| 3 | Annotate | Run the teacher over the full dataset → YOLO labels |
| 4 | QA | Dataset stats + label validation |
| 5 | Review | *(optional)* Inspect labels in Voxel51 |
| 6 | Train | Train YOLO11n with sensible augmentations |
| 7 | Evaluate | mAP50 / mAP50-95 / precision / recall |
| 8 | Failure mining | Surface top teacher-vs-student disagreements |
| 9 | Report | Assemble a markdown report |
| 10 | Dashboard | Web interface — test the model interactively (teacher vs YOLO vs LLM) |

## Quick Start

1. **Ensure the 3B teacher is reachable.**
   - *Local backend:* a CUDA GPU with `torch` + `transformers`; the model loads
     in-process from `nvidia/Locate-Anything-3B`.
   - *Endpoint backend:* a served, OpenAI-compatible model (e.g. a llama.cpp /
     vLLM server on a GPU). Note its `host:port`.

2. **Set up the skill.** Install dependencies from the skill root:

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure your run.** Copy the example and edit it:

   ```bash
   cp config.example.yaml config.yaml
   # set: images_dir, class_description, working_dir
   # teacher: leave local, OR set locate_anything_endpoint: "host:port"
   ```

4. **Preview before committing.** Annotate a handful of images and check the
   grid — refine `class_description` if the boxes look wrong:

   ```bash
   python scripts/parse_dataset.py      --config config.yaml
   python scripts/preview_annotation.py --config config.yaml   # open preview_grid.png
   ```

5. **Build the detector.** Run the full pipeline (or go phase by phase):

   ```bash
   python build_detector.py --config config.yaml
   # flags: --yes (skip preview prompt), --no-review, --no-dashboard
   ```

6. **Inspect & deploy.** Open the web interface, Voxel51 review, or read `report.md`:

   ```bash
   python scripts/launch_dashboard.py --config config.yaml    # http://localhost:7860 (test model interactively)
   python scripts/launch_fiftyone.py --config config.yaml     # http://localhost:5151 (explore annotations)
   ```

## Outputs (in `working_dir`)

`manifest.json` · `preview_grid.png` · `dataset/` (images, labels, metadata,
`data.yaml`) · `qa_report.md` · `best.pt` / `last.pt` · `eval/metrics.json` ·
`failures/` · `report.md` · `dashboard_assets/` · `logs/`

## Driving it with an agent

This repo doubles as a VS Code **agent skill** — see [SKILL.md](./SKILL.md) for
the orchestrated, step-by-step workflow with approval gates. For the full
design, rationale, and roadmap, see
[full_agent_context.md](./full_agent_context.md).
