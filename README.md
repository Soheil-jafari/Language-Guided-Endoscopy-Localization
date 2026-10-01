# Language-Guided Endoscopy Localization

Research code for language-conditioned relevance scoring in Cholec80 videos.
The main model combines a TimeSformer visual backbone, CLIP text encoder, a
cross-attention head and a short-window temporal head. Optional components are
LoRA, evidential binary outputs, and temporal/optical-flow consistency.

This revision repairs data, training, checkpoint and evaluation defects. **New
training and evaluation are required. Historical dissertation numbers are not
validated results from this revision.** See [REPAIR_NOTES.md](REPAIR_NOTES.md) for
the complete change/migration checklist and verification limits.

## One-command run (HPC / batch scheduler)

The whole experiment (data download, frame extraction, labels, splits, training,
evaluation and a summary report) runs from a single resumable command:

```sh
bash setup_env.sh                                            # once: conda env "lgel"
bash run.sh --preset full --root /path/to/big/scratch/lgel   # resubmit the same command to resume
```

See [README_HPC.md](README_HPC.md) for the inputs it needs, the presets, the offline
workflow and troubleshooting. The sections below describe the individual tools that
`main.py` drives.

## Environment

The local CPU checks use Python 3.12, PyTorch 2.5.1 and torchvision 0.20.1.
Install the matching CPU or CUDA PyTorch build for the machine, then install
`requirements.txt`. Run `python -m pytest tests -q` before an experiment.
The optional official Mamba backend requires its own compatible CUDA installation;
it is never silently replaced by the custom SSM.

## Data contract

- JPEG names contain **zero-based source-video frame IDs**, not positions in a
  sparse folder: `CHOLEC80__video01/frame_0000025.jpg` means source frame 25.
- `video_metadata.json` maps each video ID to `source_fps` and source
  `frame_count`. Sampling defaults to 1 FPS; source FPS is read from metadata.
  These tools assume constant-frame-rate videos.
- Seven phase concepts and seven tool concepts have canonical IDs in
  `data_contract.py`. A query's original sentence reaches the text encoder.
  `query_kind` and `concept_id` identify its **supervised target**, including for
  independently written paraphrases. These labels do not establish unseen-concept
  generalization.
- Missing tool observations have label `-100`; they are not negative labels.
  Phase rows define intervals until the next phase row. Corrupt/missing images,
  contradictory labels, and unrecognized targets stop a run.
- Loose-frame folders and flat per-video ZIP archives are supported.

## Prepare and check a new run

Use fresh output paths. Recover the original video split lists if comparing to a
historical run. `create_splits.py` can create a **new** reproducible split using
explicit ratios, but cannot recover old splits.

```sh
python dataset_preprocessing/build_dataset.py annotations --phases /data/phase_annotations --tools /data/tool_annotations --output /data/parsed_v2.csv
python dataset_preprocessing/build_dataset.py extract --videos /data/videos --frames /data/frames_v2 --metadata /data/video_metadata.json --sample-fps 1
```

A split JSON has exactly `train`, `val`, and `test` lists containing standardized
video IDs such as `CHOLEC80__video01`. For a deliberately new split, use:

```sh
python dataset_preprocessing/create_splits.py --metadata /data/video_metadata.json --output /data/splits_v2.json --train-ratio 0.8 --val-ratio 0.1 --seed 42
python dataset_preprocessing/build_dataset.py triplets --annotations /data/parsed_v2.csv --metadata /data/video_metadata.json --frames /data/frames_v2 --splits /data/splits_v2.json --output /data/triplets_v2 --sample-fps 1
python audit_data.py --train /data/triplets_v2/cholec80_train_triplets.csv --val /data/triplets_v2/cholec80_val_triplets.csv --test /data/triplets_v2/cholec80_test_triplets.csv --annotations /data/parsed_v2.csv --metadata /data/video_metadata.json --frames /data/frames_v2 --output /data/preflight_v2.json --decode-all
```

The extractor refuses existing video folders/archives. Existing correctly indexed
frames can be reused with independently verified source metadata and a successful
preflight; a folder's JPEG count is not source-video duration.

## Train and resume

Edit a JSON configuration override; an example is
[`examples/run_config.json`](examples/run_config.json). Set actual data,
pretrained-backbone and new checkpoint paths. Unknown configuration fields fail.
An empty backbone-weight path deliberately selects random visual initialization;
a nonempty missing/incompatible path is an error.

```sh
python train.py --config examples/run_config.json
python train.py --config examples/run_config.json --resume_from /runs/new_run/latest_model.pth
```

Resume restores an **epoch-boundary** checkpoint, not a partially completed batch.
It requires matching configuration and input-file fingerprints. `--finetune_from`
loads same-architecture weights strictly and starts a new experiment. Old checkpoints
lacking the repaired metadata are not accepted as repaired-run inference/resume.

## Export predictions and evaluate

Use full split manifests for evaluation. Training subsets never reduce validation.
The proposed model exports one averaged score per video/query/source-frame:

```sh
python predict.py --checkpoint /runs/new_run/best_model.pth --triplets /data/triplets_v2/cholec80_val_triplets.csv --metadata /data/video_metadata.json --frames /data/frames_v2 --annotations /data/parsed_v2.csv --output /results/proposed_val.csv
python build_reference.py --triplets /data/triplets_v2/cholec80_val_triplets.csv --metadata /data/video_metadata.json --frames /data/frames_v2 --annotations /data/parsed_v2.csv --output /results/val_reference.csv
python evaluate.py --predictions /results/proposed_val.csv --reference /results/val_reference.csv --split validation --select-threshold --output /results/proposed_calibration.json
```

Repeat the two exports on the **test** manifest, then evaluate with the saved
validation operating threshold:

```sh
python evaluate.py --predictions /results/proposed_test.csv --reference /results/test_reference.csv --split test --calibration /results/proposed_calibration.json --output /results/proposed_test_metrics.json
```

Frame reports contain AP, AUROC, fixed-threshold F1, Brier score, ECE, and AURC,
with per-video and per-query breakdowns. Undefined metrics are `null`. EDL runs
also export evidence uncertainty and report its separate error-ranking AURC.
Confidence/uncertainty is not proof of unknown-event recognition.

For a standalone video and arbitrary sentence:

```sh
python inference.py --checkpoint /runs/new_run/best_model.pth --video /data/videos/video01.mp4 --query "Calot triangle dissection phase" --output /results/example
```

`--raw-patch-maps` exports raw relevance-head patch logits, **not attention maps
or explanations of the final temporal score**. Inference uses short windows and
does not carry hour-long state.

## Baselines and temporal metrics

`benchmark.py` runs Hugging Face CLIP or Microsoft's X-CLIP on the shared frame
grid. Specify the exact pretrained model name; it is saved with the output.
`train_xclip.py` is an adapted supervised contrastive training script. It uses
the official model's video/text logits and observed multi-positive targets.
These are not a CLIP linear probe or the original papers' evaluation protocols.

```sh
python benchmark.py --model clip --model-name openai/clip-vit-large-patch14 --triplets /data/triplets_v2/cholec80_val_triplets.csv --metadata /data/video_metadata.json --frames /data/frames_v2 --annotations /data/parsed_v2.csv --output /results/clip_val.csv
```

Run `python benchmark.py --help` and `python train_xclip.py --help` for X-CLIP
options. Scores are rescaled cosine similarities, not calibrated probabilities;
select each model's threshold on validation only. Microsoft's X-CLIP is the
Ni et al. model; do not identify it as the different Ma et al. retrieval model.

The code in `comparison_models/Moment-DETR` is a **ResNet50/RoBERTa adapted DETR**,
not an official Moment-DETR reproduction. Its revised preprocessing groups all
spans for a video/query and keeps absent queries. See that directory's README.

Use its `run_preprocessing.py` to create complete sampled-grid reference segments.
`export_segments.py` converts proposed/CLIP/X-CLIP frame CSVs against that same
reference; it requires a fixed threshold or a validation calibration JSON. Then:

```sh
python evaluate.py --segments --predictions /results/model_segments.jsonl --reference /data/moment/test.jsonl --split test --output /results/model_temporal_metrics.json
```

Temporal reports distinguish pooled AP at tIoU 0.3/0.5/0.7, positive-query R@1,
and absent-query false alarms. AP at one tIoU is not labelled an overall mAP.
Incomplete tool annotations cannot be silently converted into complete absence
or segment targets. Old evaluation entry points delegate to the new explicit-input
CLIs; old arguments and implicit artifact searches have been retired.

## Third-party code and data

This repository is MIT-licensed except for adapted third-party code, which keeps its
original licence:

- `backbone/vision_transformer.py` is adapted from
  [TimeSformer](https://github.com/facebookresearch/TimeSformer) (CC BY-NC 4.0) and
  [pytorch-image-models](https://github.com/huggingface/pytorch-image-models) (Apache 2.0).
- Pretrained models (the M2CRL backbone checkpoint, CLIP, X-CLIP) and the Cholec80
  dataset are **not** distributed here and are subject to their owners' terms.
