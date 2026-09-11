# Adapted ResNet50/RoBERTa DETR baseline

This repository implementation is not an official Moment-DETR reproduction.
It uses ImageNet ResNet50 features, RoBERTa text, custom Transformer layers and a
start/width span head. Report it under that adapted name.

Run each script with `--help`; all paths are explicit. From this directory:

1. `run_preprocessing.py --triplets ... --metadata ... --frames ... --annotations ... --output /data/moment/train.jsonl`.
   Repeat for `val.jsonl` and `test.jsonl`. Each video/query has all positive spans,
   or an empty list for an absent target. Targets use declared sampled-grid cells.
   Unobserved tool frames cause an error rather than invented negative segments.
2. `run_feature_extraction.py --frames ... --metadata ... --output /data/features`.
   The NPZs retain source-frame IDs, timestamps, duration and sampling rate.
3. Write a JSON override, for example
   `{"ann_path":"/data/moment","feature_path":"/data/features","epochs":3,"batch_size":2,"num_workers":0}`.
   Then `run_training.py --config /data/moment_config.json --output /runs/adapted_detr_v2`.
4. `run_evaluation.py --resume /runs/adapted_detr_v2/best.ckpt --output /results/adapted_detr_test.jsonl --split test`.

Use `python` before each script name. Install dependencies from the root
`requirements.txt`. Checkpoints have a repaired schema and complete configuration;
old checkpoints are not silently loaded into revised experiments. Resume training
with the same arguments and `--resume-from .../latest.ckpt`.

`max_v_len` is an explicit feature-downsampling limit, not source-video duration.
Changing it changes the experiment. Ground-truth spans are never snapped to this
token grid or to a padded batch's length. Shared temporal evaluation uses pooled
AP and positive-query R@1 and includes absent-query false alarms. Frame metrics
require rasterization on the separate common source-frame reference; no padded
256-token approximation is reported as a full-frame benchmark.
