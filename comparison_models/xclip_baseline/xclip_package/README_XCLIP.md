# X-CLIP (Ni et al., 2022) – Training, Inference & Metrics for Your Project

This package wires **Hugging Face Transformers' X-CLIP** to your existing project structure.
It reads paths from your `project_config.py` (the one you uploaded) and expects the split CSVs and extracted frames layout already in place.

> **Model**: Expanding Language-Image Pretrained Models for General Video Recognition (X-CLIP, ECCV 2022) – Ni et al.  
> Hugging Face docs: `transformers` model `XCLIPModel` + `XCLIPProcessor`.

## What you get

- `train_xclip.py` — Fine-tunes X-CLIP with **InfoNCE (CLIP-style)** on positive query–window pairs sampled from your GT segments. Batch negatives are used implicitly.
- `eval_xclip.py` — Sliding-window inference for each (video, text query) in your **test CSV**, producing:
  - **Frame-level**: AUROC, AUPRC, AP (and macro-averages).
  - **Segment-level**: mAP@tIoU={0.3,0.5,0.7}, Recall@{1,5} (greedy matching).
  - Saves **JSON + TXT** summaries and optional per-video CSV of scores.
- `xclip/data.py` — Dataset utilities reading your CSVs and assembling 16-frame windows from `/extracted_frames/<video_id>/*.jpg`.
- `xclip/model.py` — Thin wrapper on `XCLIPModel` including **cross-frame attention**, **multi-frame integration**, and **video-specific prompts** (via HF config).
- `xclip/losses.py` — Symmetric InfoNCE (video→text and text→video).
- `xclip/metrics.py` — AUROC/AUPRC/AP, tIoU, mAP@tIoU, Recall@K, plus helpers.
- `xclip/utils.py` — Logging, seeding, file IO.

## Assumptions about your CSVs
We try to be permissive and detect columns automatically. We look for the following column names (case-insensitive):

- `video` or `video_id` — folder name under `EXTRACTED_FRAMES_DIR`.
- `query` or `text` — natural-language description to localize.
- Either **frame** columns: `start_frame`, `end_frame`, or **time** columns: `start_time`, `end_time` (in **seconds**).
- Optional: `fps` per video. If missing, we will use `config.DATA.FRAME_RATE`.

You can override with flags (see each script’s `--help`).

## Quickstart

```bash
# (on your university server)
pip install -r requirements.txt

# Train (optional fine-tune)
python train_xclip.py --epochs 5 --batch-size 16

# Evaluate on your test split; produce metrics & files
python eval_xclip.py --stride 8 --save-per-video
```

Outputs go under: `${config.OUTPUT_DIR}/xclip/<timestamp>/`:
- `metrics.json`, `metrics.txt`
- `scores_<video>_<queryhash>.csv` (if `--save-per-video`)

## Notes
- This is **faithful** to the *Ni et al., 2022* X-CLIP formulation by using the official HF implementation (same architectural pieces: cross-frame attention and multi-frame integration). It is not the different *Ma et al., 2022* "multi-grained" retrieval variant.
- For long videos, adjust `--max-windows-per-video` to balance speed vs. coverage.
- If your frames are not `*.jpg` or names are irregular, use `--frame-glob`.
