# YO-YOLO — Full Agent Context

A detailed companion to [README.md](./README.md) and [SKILL.md](./SKILL.md):
the *why* behind the design, the data contracts between stages, and a roadmap
of ideas to push it further.

---

## 1. Concept

YO-YOLO distills a heavy, prompt-driven vision-language model into a tiny,
fast, single-purpose detector.

```
Natural-language description + folder of images
        │
        ▼  Locate Anything 3B  (teacher — slow, general, no training)
   auto-generated bounding boxes
        │
        ▼  YOLO11n  (student — fast, specialized, deployable)
   best.pt
```

The teacher never ships. It exists only to label data. The student is what you
deploy: a few-MB model running in single-digit milliseconds.

**Design philosophy (V1):** minimal user input, no active learning, no agent
planning, a single student architecture. The orchestrator is deliberately
"dumb" — it runs scripts in order, shows results, and asks for approval at two
gates. All real logic lives in the stage scripts.

---

## 2. Architecture

```
yo-yolo/
├── SKILL.md                # agent orchestration contract (the "brain" is thin)
├── README.md               # concise human entry point
├── full_agent_context.md   # this file
├── config.example.yaml     # single source of truth for a run
├── requirements.txt
├── locate_anything.py      # teacher wrapper: local + endpoint backends
├── build_detector.py       # optional one-shot runner (calls the scripts in order)
├── references/
│   └── teacher_prompts.md  # prompt shapes + output format
└── scripts/
    ├── common.py           # config, paths, logging, geometry, drawing
    ├── parse_dataset.py        (1)
    ├── preview_annotation.py   (2)
    ├── annotate_dataset.py     (3)
    ├── dataset_qa.py           (4)
    ├── launch_fiftyone.py      (5, optional)
    ├── train_yolo.py           (6)
    ├── evaluate.py             (7)
    ├── failure_mining.py       (8)
    ├── generate_report.py      (9)
    └── launch_dashboard.py     (10)
```

### Stage contract

Every stage script follows the same conventions so the agent (or
`build_detector.py`) can chain them blindly:

- **Input:** `--config config.yaml` (and optional flag overrides).
- **State:** everything lives under `working_dir`; stages communicate only
  through files on disk — there is no in-memory pipeline object.
- **Progress:** `tqdm` bars + human-readable `logging` to console and
  `logs/<stage>.log`.
- **Machine-readable result:** a final line `YOYOLO_SUMMARY {json}` the
  orchestrator parses to report counts / metrics / output paths.
- **Failure:** non-zero exit code; missing optional dependencies print a clear
  message and exit cleanly rather than crashing the pipeline.

### Teacher backends (`locate_anything.py`)

| Backend | How | `generation_mode` | When |
|---------|-----|-------------------|------|
| `local` | `transformers` `AutoModel(..., trust_remote_code=True)` on CUDA | `fast` / `slow` / `hybrid` | GPU box, model loaded in-process |
| `endpoint` | POST to OpenAI-compatible `/v1/chat/completions` with base64 `image_url` | ignored | served model (llama.cpp / vLLM) |

Selected automatically: if `locate_anything_endpoint` is set → `endpoint`,
otherwise `local`. Every `predict()` records `latency_ms`, which feeds the
speed comparison in the report and dashboard.

---

## 3. Data flow & key artifacts

```
parse_dataset      → manifest.json                     (image list, split, class_name)
preview_annotation → preview_grid.png                  (prompt sanity check)
annotate_dataset   → dataset/{images,labels}/{train,val}/
                     dataset/metadata/<img>.json        (raw answer, boxes, latency)
                     dataset/data.yaml
                     dataset/annotation_stats.json
dataset_qa         → qa_report.md, qa_stats.json
train_yolo         → best.pt, last.pt, results.csv, train_summary.json
evaluate           → eval/metrics.json
                     eval/predictions/<img>.json        (teacher_boxes, yolo_boxes)
failure_mining     → failures/<rank>_<img>.png, failures/index.json
generate_report    → report.md, dashboard_assets/*.png
launch_dashboard   → Gradio app (reads everything above)
```

The two files that tie it all together:

- **`metadata/<img>.json`** — per-image teacher output: raw answer string,
  pixel `xyxy` boxes, and latency. The audit trail for annotation.
- **`eval/predictions/<img>.json`** — teacher vs YOLO boxes side by side; the
  input to both failure mining and the dashboard's validation browser.

---

## 4. Decisions & rationale

- **YOLO11n by default.** Smallest/fastest; the point is a deployable student.
  Swappable via `train.model` (e.g. `yolov8n.pt`).
- **Single class in V1.** The teacher is grounded on one description → one
  class (`class_id` always `0`). Keeps labels, QA, and metrics simple.
- **Preview gate before full annotation.** Annotating thousands of images with
  a bad prompt is the most expensive mistake; 4 images catch it for ~seconds.
- **Files over a framework.** Stages are independently runnable and debuggable;
  you can re-run just `train_yolo.py` without re-annotating.
- **`fast` generation mode for bulk annotation.** Throughput matters more than
  marginal box quality across a whole dataset; `slow`/`hybrid` remain available.
- **Orchestrator LLM as a baseline, not a tool.** The dashboard's zero-shot LLM
  box prediction exists to *demonstrate* why a trained detector wins on both
  accuracy and speed — it is intentionally weak.

---

## 5. Known limitations (V1)

- **Single class only.** No multi-class or hierarchical labels yet.
- **Teacher = ground truth.** Metrics measure agreement with the teacher, not
  absolute correctness. A systematically wrong teacher yields a confidently
  wrong student. The preview gate + QA + failure mining mitigate but don't
  eliminate this.
- **No label de-duplication / NMS on teacher output.** Overlapping teacher
  boxes pass through as-is.
- **Random train/val split.** No stratification or dedup of near-duplicate
  frames (relevant for video-derived datasets → optimistic val metrics).
- **No confidence from the teacher.** Locate Anything boxes are unscored, so we
  can't threshold weak detections during annotation.

---

## 6. Roadmap — ideas to improve

Roughly ordered from highest leverage / lowest effort to more ambitious.

### Near-term, low effort
- **Teacher box post-processing:** class-agnostic NMS + min/max area filtering
  in `annotate_dataset.py` to drop duplicates and obvious noise.
- **Multi-class support:** accept a list of descriptions; one teacher pass per
  class (or `detect()` with `</c>`-joined categories), `nc > 1` in `data.yaml`.
- **Stratified / dedup split:** perceptual-hash near-duplicate grouping so the
  same scene can't leak across train/val.
- **Confidence-style filtering:** run the teacher twice (`fast` + `slow`) and
  keep boxes that agree (IoU vote) as a cheap pseudo-confidence.
- **Resumable annotation:** skip images that already have a `metadata/*.json`,
  so interrupted runs continue instead of restarting.
- **Config-driven export:** auto-export `best.pt` to ONNX / TensorRT for
  deployment at the end of training.

### Medium effort
- **Active learning loop.** The natural next step beyond V1:
  1. Annotate + train on a small seed set.
  2. Run the student on the *unlabeled* remainder.
  3. Score images by *uncertainty* (low/medium confidence, many near-threshold
     boxes) and *teacher-student disagreement*.
  4. Send only the most informative images back to the teacher (expensive) for
     labeling.
  5. Retrain. Repeat until metrics plateau.
  This spends teacher compute where it matters and typically reaches target
  accuracy with far fewer teacher calls. `failure_mining.py` already computes
  the disagreement signal that would seed step 3.
- **Human-in-the-loop correction:** let FiftyOne edits write back to the YOLO
  labels (curated ground truth overrides teacher labels), then retrain.
- **Cross-validation / multiple seeds** for more trustworthy small-dataset
  metrics, with variance reported.
- **Auto hyperparameter tuning** (Ultralytics `model.tune()`) gated behind a
  budget so it stays optional.
- **Distillation beyond boxes:** use teacher soft signals (multiple samples,
  agreement maps) as label smoothing / soft targets for the student.

### Larger / research-flavored
- **Iterative prompt refinement:** an agent loop that inspects preview boxes,
  proposes a better `class_description`, and re-previews until the user
  approves — closing the loop on the single biggest quality lever.
- **Ensemble / multi-teacher labeling:** combine Locate Anything with another
  open-vocabulary detector (e.g. GroundingDINO, OWLv2) and fuse boxes; train on
  the consensus for higher-quality pseudo-labels.
- **Self-training / pseudo-label bootstrapping:** after an initial student,
  use its high-confidence predictions on unlabeled data as extra labels
  (with the teacher arbitrating conflicts).
- **Continuous / online datasets:** watch a folder, incrementally annotate new
  images, and periodically fine-tune — turning YO-YOLO into a standing service.
- **Segmentation / keypoints:** Locate Anything also points; extend the student
  to YOLO-seg or pose for richer outputs from the same prompt-driven workflow.
- **Cost & quality dashboard:** track teacher GPU-seconds vs achieved mAP across
  iterations to make the accuracy/compute trade-off explicit.

---

## 7. For the orchestrating agent

If you are the agent driving this skill, remember:

- You are a sequencer, not a modeler. Run scripts, parse `YOYOLO_SUMMARY`,
  report clearly, and **stop at the gates** (preview approval; optional review).
- Surface difficulties plainly: empty annotations, missing GPU/torch, install
  errors, low mAP. Suggest the obvious fix (refine the prompt, switch to the
  endpoint backend, install a package) rather than improvising new code.
- Only edit a stage script when the user asks for behavior the scripts don't
  support — and keep the stage contract (config in, files + `YOYOLO_SUMMARY`
  out) intact.
