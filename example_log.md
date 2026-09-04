# YO-YOLO Volleyball Example — Execution Trace

Goal: longest volleyball rally via distilled `yellow ball` detector.
Teacher: NVIDIA Locate Anything 3B via llama.cpp endpoint `localhost:8080`.
Student: YOLO11n. Video: Google Drive `1TgrVhX8BbW-FzP__RckkjXJ4G0GmG9k5`.

## Step 0 — Environment
- `uv venv --python 3.13 .venv` + `pillow numpy pyyaml tqdm requests matplotlib opencv-python ultralytics`
- llama.cpp endpoint OK: model `LocateAnything-3B-Q4_K_M.gguf`, `/health ok`
- Video `data/volleyball.mp4`: 149MB, 8246 frames, 25fps, 1920x1080, 329.8s (~5.5min)
- Download: `uvx gdown <drive-link> -O data/volleyball.mp4`
- Note: no system `ffmpeg`; use OpenCV for sampling + mp4 writing.

## Step 1 — Prompt selection
- preview with `yellow volleyball ball`: 1 box / 4 imgs, avg 586ms
- tested 4 prompts on 5 spread frames: `volleyball`=3, `small round ball in play`=2, `a volleyball ball`=1, `yellow volleyball ball`=1
- decision: use `volleyball` (class_name `ball`). Lesson: simple noun > over-specific color prompt for Locate Anything.
- re-ran parse after config change.

## Step 2 — Full annotation
- prompt `volleyball`: 272 boxes / 512 imgs, 240 empty (46.9%), avg 449ms, total teacher 230s (~2img/s via llama.cpp endpoint)
- vs initial `yellow volleyball ball` would have been ~25% recall. Prompt choice = 4x data.
- preview after fix: 4 boxes / 4 imgs.

## Step 3 — Parallel annotation (--workers, progress every 100)
- Added `--workers N` to `scripts/annotate_dataset.py` (ThreadPoolExecutor, one worker per thread;
  endpoint backend is stateless `requests.post`, safe to share). Local backend falls back to 1.
- Added retry (3 attempts, backoff 2s/4s) so transient server drops don't wipe labels
  (failures raise *before* writing, old label files survive).
- Added `log.info("Progress %d/%d ...")` every 100 images.
- Lesson: `--workers 4` against `lserver -np 4` killed the server (connection refused)
  and zeroed the run summary. `--workers 2` survived: full tqdm `Annotating x2` bar,
  100/200/300/400/500 checkpoints, done in ~2:50 vs 4:13 sequential (~1.5x).
- Final: 273 boxes / 512 imgs, 241 empty (47%), avg 606ms/req (shared GPU), 310 teacher-seconds.
- QA: 273 objects, 0 issues, pass.

## Step 4 — Train + evaluate (YOLO11n, 30 epochs, RTX 5080)
- Fixed `scripts/train_yolo.py` LossPlateauStopper: `trainer.loss_items` is a dict
  in ultralytics 8.4 (`loss.get("box", ...)`), was `loss[0]` -> KeyError.
- Train ~124s. Val: P=0.907 R=0.557 mAP50=0.591 mAP50-95=0.274 (vs teacher as GT).
- Eval latency: YOLO 10.88ms vs teacher ~606ms -> ~55x speedup. This is the slide.
- failure_mining: 50 cases. report.md generated.

## Step 5 — Longest rally (`scripts/track_rally.py`, new)
- Full-video YOLO inference (8246 frames, conf 0.2) + simple tracking:
  keep best box/frame -> interpolate gaps <=25 frames (1s @25fps) ->
  moving-average smooth (w=5) -> rallies split on gaps >25 frames.
- Raw ball visible 2825/8246 (34%), 47.6% after gap-fill. 32 rallies found.
- Longest: frames 68-550, t=2.72s-22.0s, 19.32s, 483 frames.
- Outputs in `runs/volleyball_rally/rally/`: detections.csv, rallies.json,
  longest_rally.mp4 (25MB, bbox + rally timer), presence_timeline.png,
  trajectory_map.png (parabolic serve/receive arcs clearly visible).
- Verified clip frames at 0/100/240/400: tight boxes in wide + closeup views.
