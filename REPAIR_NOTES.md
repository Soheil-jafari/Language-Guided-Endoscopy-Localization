# Repair and migration notes

Base revision: `f4e68e3c4f29603a613c7b63e7175bf9b3c61ff2`.
This patch supersedes the earlier limited `audit_fixes.patch`. Apply it to the
base revision, not on top of that earlier patch. No GitHub commits or pushes were
made. There are no new Cholec80 performance results in this package.

## Implemented repairs

| Area | Change |
|---|---|
| Checkpoints | Unwrap real tensor dictionaries; normalize DataParallel prefixes; strict model loads; fail on missing/incompatible checkpoints. New checkpoints contain the complete configuration, schema version, validation threshold, optimizer/scheduler/scaler, RNG and input-file hashes. |
| Resume | Resume only at a completed epoch with matching configuration and input hashes. Best loss is updated before latest is saved. No accidental restart when a requested checkpoint is missing. |
| Labels | One canonical phase/tool identity across aliases. Explicit metadata supports paraphrases while passing the original sentence to CLIP. Contradictory rows and unrecognized label text fail. |
| Missing observations | Raw annotation parsing outer-merges phase/tool rows; missing tool observations remain `-100`. Both BCE and EDL ignore them. Unknown wording never defaults to a negative. |
| Sampling | One absolute-time sampling rule using source FPS and zero-based source-frame IDs. Fractional source FPS does not accumulate rounded-stride drift. Folder/ZIP position is never treated as source-frame index. |
| Images | Missing/corrupt frames fail; no black/red image or previous-frame fallback. Only genuinely short videos are explicitly padded; padded labels are ignored. ZIP handles are reopened per worker process. |
| Splits | Check video overlap, preserve full validation under training subsets, require explicit split ratios for new split generation. Historical 64/8/8 lists are not reconstructed or claimed. |
| Validation | Cover every video/query in the supplied manifest; average overlap predictions once per source frame. Select checkpoints by deduplicated validation frame NLL. |
| Accumulation | Normalize gradients by actual observed-frame count, including a partial final group and unequal batch sizes. Do not advance scheduler on an AMP-skipped update. |
| LoRA | Freeze all non-adapter CLIP text parameters when PEFT is selected; apply adapter dropout to the input; record trainable parameter counts per component. |
| Text fusion | Exclude padding tokens from confidence-fusion pooling; use the instance's configuration rather than a global flag. |
| EDL | Replace the claimed-but-not-implemented KL with target-adjusted Dirichlet KL plus expected NLL and annealing. Evidence order is positive/negative. Export uncertainty separately from relevance. |
| Flow | Compute backward RAFT flow, correct half-pixel coordinates for `align_corners=False`, scale flow vectors with resolution, and exclude out-of-bounds samples. Never use randomly initialized RAFT as a fallback. |
| Backbone checkpointing | Bind each Transformer block during gradient recomputation, avoiding the loop-variable closure bug. Test gradients with checkpointing enabled/disabled. |
| Architecture identity | EndoMamba's placeholder now fails explicitly. Official Mamba never silently changes to another implementation. The optional custom recurrent SSM is labelled separately and its unused projection/invalid Conv1d memory-format call are removed. |
| Inference | Use saved model configuration and training sampling; cover the tail; do not silently cap/subsample or shift timestamps after decode errors. Merge short gaps correctly and clamp segments to actual duration. |
| Explanations | Export raw relevance-head patch logits with an accurate description. Their spatial mean reconstructs the raw head output; they do not explain the final temporal/evidential output. |
| CLIP | Repair the executable entry point. Explicitly choose the exact pretrained model and use its matching processor. This is similarity scoring, not an unimplemented linear probe. |
| X-CLIP | Use HF `logits_per_video`; its `text_embeds` tensor has video/query/embedding axes, not token pooling. Multi-positive training uses observed targets, handles duplicate concepts/paraphrases and co-occurring positives, and masks unobserved pairings. Preserve seconds in interchange exports and retain explicit absent records. |
| Adapted DETR data | One record per video/query with all spans or an empty target. Store source timestamps with features; never infer duration from the last sparse JPEG. Keep exact normalized targets independent of padded batch length. |
| Adapted DETR loss | Handle all-empty and mixed target batches without NaN span losses; implement actual generalized temporal IoU in start/end coordinates; include foreground matching cost. |
| Adapted DETR evaluation | Use common temporal metrics, include absent queries, and stop reporting padded-token frame scores as comparable frame metrics. No test-set threshold sweep. |
| Shared metrics | Correct AP integration, matched-GT selection, tied-rank AUROC, explicit missing/empty predictions, class-confidence AURC, masked labels and undefined metrics. Distinguish pooled temporal AP, R@1 and absent-query false alarms. Frame evaluation checks exact reference keys and reports per-video/query breakdowns. |
| Artifact provenance | Explicit input/output files replace automatic fallback searches across old runs. New run/output paths protect existing results. |

## Intentional interface changes

The old entry points contained incompatible sampling/evaluation assumptions. Their
CLI implementations were consolidated rather than leaving both old and new paths
available for accidental mixing:

| Previous path | Repaired entry point |
|---|---|
| `prepare_cholec80.py`, both extraction scripts | `dataset_preprocessing/build_dataset.py` subcommands |
| `clip_baseline.py`, `clip_eval.py` | `benchmark.py --model clip --model-name ...` |
| Nested X-CLIP train/eval/infer scripts | Root `train_xclip.py` / `benchmark.py --model xclip ...` |
| `evaluat_segments.py`, `make_report.py`, `calip_temporal_eval.py` | `evaluate.py` with explicit prediction/reference inputs |
| Training's single-query evaluation flags | `inference.py` for visualization; `predict.py` and common evaluators for benchmarks |
| Adapted DETR single-query evaluator | The revised `run_evaluation.py`; use a deliberately prepared subset manifest for case studies |

Compatibility launchers show the new `--help`; old argument names are not silently
interpreted. Old nested X-CLIP helper APIs for dense-index loading and private
metrics were replaced by shared helpers. The old rendered heatmap/video pipeline
is replaced by accurately named raw patch-map exports. `make_report.py` now emits
the shared JSON metrics rather than assembling a report from guessed artifacts.

## Before a server run

1. Preserve historical data, splits, outputs and checkpoints. Build fresh parsed
   annotations/manifests or audit existing files against this contract. Previously
   imputed zero labels cannot be reconstructed as missing observations from that
   CSV alone; use the original phase/tool annotations.
2. Supply real source FPS/frame counts. Existing JPEGs can be reused only if the
   source numbering and required samples pass preflight. Extract into a separate
   directory if needed. These readers assume constant-frame-rate video.
3. Use `audit_data.py --decode-all` before expensive training. Review unknown-label
   counts and actual split lists. Sampling at a rate with no tool observations
   must not be presented as complete tool supervision.
4. Use a new output directory and an explicit configuration. `--finetune_from`
   requires exactly matching state shapes/keys; it is not partial architecture
   migration. Old objective/sampling results remain old results.
5. Run a short real-data forward/backward/checkpoint round-trip before scaling up.
   The local tests deliberately use small randomly initialized models and synthetic
   data, not your trained model or downloaded surgical weights.

## Verification and remaining limits

The included CPU tests exercise the actual small framework in both uncertainty
modes, CLIP text LoRA, HF X-CLIP tensor axes, dataset images/annotations, checkpoint
round-trips and epoch resume, accumulated updates, warping, empty/absent targets,
GIoU, and common evaluation/export paths. The delivery's validation report records
the final test count, compilation, command checks and patch application check.

Not validated locally: real Cholec80 files, existing M2CRL checkpoints/key mappings,
full-size pretrained model downloads, CUDA/AMP overflow on real hardware, RAFT
quality on surgery, official Mamba kernels, throughput, GPU memory, model accuracy,
calibration or publication claims. The flow mask excludes out-of-bounds samples,
not occlusions. Resume is tested on CPU; bitwise cross-hardware reproducibility is
not promised. The supported trainers use one device; the previous incomplete
distributed validation path is not retained.

An official EndoMamba integration and an official Moment-DETR reproduction are
**not implemented by this patch**. It prevents their names being applied to
different architectures. The existing baseline remains an adapted ResNet50/
RoBERTa DETR, and CLIP/X-CLIP scores remain uncalibrated similarities.

The four research protocols (seen concepts, unseen wording, held-out concepts,
absent/unknown events) need separately designed datasets and experiments. Code
repairs do not create those annotations or establish open-vocabulary recognition.
