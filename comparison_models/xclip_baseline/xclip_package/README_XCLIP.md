# Repaired X-CLIP entry points

Use the root `train_xclip.py` and `benchmark.py --model xclip` CLIs. The scripts
in this directory launch those shared implementations; run them with `--help`.
Install the root requirements. Legacy dense-position frame loaders are retired.

This is Microsoft's HF X-CLIP (Ni et al.) with an adapted contrastive fine-tuning
protocol. It is not the different Ma et al. retrieval model. Video-conditioned
text embeddings are handled through `logits_per_video`. Repeated/co-occurring
positive targets are not treated as diagonal-only negatives. Unobserved targets
are excluded from contrastive denominators.

Inputs are the same explicit source-frame triplets/metadata/annotation files used
by the proposed model. Training uses windows containing an observed positive and
requires at least two concepts. Validation includes all eligible positive windows;
the common full-video evaluation separately includes negative and absent targets.
The final smaller batch is merged if it would contain just one item.

Rescaled cosine scores need a validation-selected threshold. They are not calibrated
event probabilities. Specify an exact model name and use repaired checkpoints.
The JSONL-to-CSV converter remains an interchange tool preserving seconds and
absent records; it is no longer the training-data ingestion path.
