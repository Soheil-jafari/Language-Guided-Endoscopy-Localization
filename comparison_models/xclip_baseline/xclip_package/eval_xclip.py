# Save this file as: eval_xclip.py

import argparse, json, time
from pathlib import Path
from collections import defaultdict
from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import DataLoader

from xclip.model import XCLIPWrapper
from xclip.metrics import safe_auc_roc, ap_auprc, merge_binary_sequence, map_at_tious, recall_at_k
from xclip.utils import set_seed, ensure_dir, load_project_config, RunPaths, sha1_short
from xclip.data import (
    load_triplets, get_frame_paths, is_frame_list_format, 
    load_annotations_from_frame_list, VideoWindowDataset # <-- IMPORT THE NEW DATASET
)

def assign_window_scores_to_frames(n_frames: int, window_starts: List[int], window_scores: List[float], window_len: int):
    """Aggregates window scores back to frame-level scores by averaging overlaps."""
    scores = np.zeros(n_frames, dtype=float)
    counts = np.zeros(n_frames, dtype=float)
    for s, sc in zip(window_starts, window_scores):
        e = min(n_frames, s + window_len)
        scores[s:e] += sc
        counts[s:e] += 1.0
    # Avoid division by zero for frames that were not in any window
    counts[counts == 0] = 1.0
    return scores / counts

def frames_to_segments(scores: np.ndarray, threshold: float, min_gap: int=0, min_len: int=1):
    """Converts a sequence of frame scores into binary segments."""
    ybin = (scores >= threshold).astype(int).tolist()
    return merge_binary_sequence(ybin, min_gap=min_gap, min_len=min_len)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--frame-glob", type=str, default="*.jpg")
    parser.add_argument("--thresholds", type=str, default="0.3,0.5,0.7")
    parser.add_argument("--seg-th", type=float, default=0.5)
    parser.add_argument("--save-per-video", action="store_true")
    parser.add_argument("--model-name", type=str, default="microsoft/xclip-base-patch32")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--tag", type=str, default="xclip_eval_fast")
    parser.add_argument("--max-windows-per-video", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=32, help="Number of windows per forward pass")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of CPU cores for parallel data loading")
    args = parser.parse_args()

    set_seed(42)
    config = load_project_config()
    paths = RunPaths.from_config(config, tag=args.tag)
    ensure_dir(paths.out_dir); ensure_dir(paths.per_video_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device={device}, CUDA available={torch.cuda.is_available()}, GPUs={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"[INFO] GPU{i}: {torch.cuda.get_device_name(i)}")

    model = XCLIPWrapper(model_name=args.model_name)
    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()

    if args.ckpt and Path(args.ckpt).exists():
        state = torch.load(args.ckpt, map_location=device)
        # Handle DataParallel wrapped model
        model_to_load = model.module if isinstance(model, torch.nn.DataParallel) else model
        model_to_load.load_state_dict(state["model"], strict=False)
        print(f"[INFO] Loaded fine-tuned checkpoint from {args.ckpt}")
    
    # Get processor from the underlying model, handling DataParallel wrapper
    processor = model.module.processor if isinstance(model, torch.nn.DataParallel) else model.processor

    # --- DATA PREPARATION (Same as before) ---
    test_csv = config.TEST_TRIPLETS_CSV_PATH
    head = pd.read_csv(test_csv, nrows=5)
    items = defaultdict(lambda: {"gts": [], "fps": None})

    if is_frame_list_format(head):
        # Your specific implementation for this format might be needed
        print("[INFO] CSV format: frame_path, text_query, relevance_label")
        items_dict, _ = load_annotations_from_frame_list(test_csv, default_fps=config.DATA.FRAME_RATE)
        for k, v in items_dict.items(): items[k] = v
    else:
        print("[INFO] CSV format: triplets (video/query + start/end)")
        for tri in load_triplets(test_csv, default_fps=config.DATA.FRAME_RATE):
            items[(tri.video, tri.query)]["gts"].append((tri.start_frame, tri.end_frame))
            items[(tri.video, tri.query)]["fps"] = tri.fps

    # --- NEW DATALOADING PIPELINE ---
    dataset = VideoWindowDataset(items, config, args, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers, # <--- THE MOST IMPORTANT CHANGE FOR SPEED
        pin_memory=True,              # Speeds up CPU-to-GPU memory transfer
        shuffle=False
    )
    
    print(f"[INFO] Created dataset with {len(dataset)} windows to evaluate.")

    # --- NEW EVALUATION LOOP ---
    # Step 1: Get scores for all windows in parallel.
    window_scores_by_item = defaultdict(lambda: {"starts": [], "scores": []})
    amp_dtype = torch.float16 if "cuda" in device.type else None

    t0 = time.time()
    for batch in tqdm(dataloader, desc="🚀 Evaluating Windows"):
        inputs, videos, queries, start_frames = batch
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(amp_dtype is not None), dtype=amp_dtype):
            # The model is already wrapped in DataParallel if multiple GPUs are used
            out = model(**inputs) 
            
            # The output from DataParallel is a list of tensors, one per GPU.
            # We need to gather them. If only 1 GPU, it's not a list.
            if isinstance(out, (list, tuple)):
                lv = torch.cat([o.logits_per_video for o in out])
            else:
                lv = out.logits_per_video
            
            if lv.ndim == 2 and lv.shape[0] == lv.shape[1]:
                bscores = lv.diag().detach().cpu().tolist()
            else:
                bscores = lv.squeeze().detach().cpu().tolist()
                if isinstance(bscores, float): bscores = [bscores]
        
        # Store results grouped by video and query for later aggregation
        for i in range(len(videos)):
            key = (videos[i], queries[i])
            window_scores_by_item[key]["starts"].append(start_frames[i].item())
            window_scores_by_item[key]["scores"].append(bscores[i])
    
    t1 = time.time()
    print(f"[INFO] Model inference for all windows took {t1 - t0:.2f} seconds.")

    # --- METRIC CALCULATION (AGGREGATION STEP) ---
    # Step 2: Aggregate window scores into frame scores and calculate metrics.
    frame_level_metrics = []
    preds_by_item = {}
    gts_by_item = {}

    for (video, query), bundle in tqdm(items.items(), desc="📊 Aggregating & Calculating Metrics"):
        key = (video, query)
        if key not in window_scores_by_item: continue

        results = window_scores_by_item[key]
        frame_paths = get_frame_paths(config.EXTRACTED_FRAMES_DIR, video, args.frame_glob)
        n_frames = len(frame_paths)
        if n_frames == 0: continue
        
        gts = bundle["gts"]
        gts_by_item[(video, query)] = gts
        fps = int(bundle["fps"] if bundle["fps"] else config.DATA.FRAME_RATE)

        frame_scores = assign_window_scores_to_frames(n_frames, results["starts"], results["scores"], args.num_frames)
        y_true = np.zeros(n_frames, dtype=int)
        for (s, e) in gts:
            y_true[max(0, s) : min(n_frames, e)] = 1

        # Calculate Frame-level metrics
        auroc = safe_auc_roc(y_true, frame_scores)
        ap = ap_auprc(y_true, frame_scores)
        frame_level_metrics.append({"video": video, "query": query, "auroc": auroc, "ap": ap})

        # Calculate Segment-level metrics
        min_gap = int(0.1 * fps); min_len = int(0.5 * fps)
        segs = frames_to_segments(frame_scores, threshold=args.seg_th, min_gap=min_gap, min_len=min_len)
        seg_preds = [((s, e), float(frame_scores[s:e].max() if e > s else 0.0)) for (s, e) in segs]
        preds_by_item[(video, query)] = seg_preds

        # Save per-video scores if requested
        if args.save_per_video:
            qhash = sha1_short(query)
            out_csv = paths.per_video_dir / f"scores_{video}_{qhash}.csv"
            pd.DataFrame({"frame": range(n_frames), "score": frame_scores, "label": y_true}).to_csv(out_csv, index=False)

    # --- FINAL REPORTING (Same as before) ---
    macro_auroc = float(np.mean([m["auroc"] for m in frame_level_metrics])) if frame_level_metrics else 0.0
    macro_ap = float(np.mean([m["ap"] for m in frame_level_metrics])) if frame_level_metrics else 0.0

    tious = tuple(float(x) for x in args.thresholds.split(","))
    map_res = map_at_tious(preds_by_item, gts_by_item, thresholds=tious)
    recall_res = recall_at_k(preds_by_item, gts_by_item, k_list=(1, 5), thr=0.5)

    metrics = {
        "frame_level": {"macro_AUROC": macro_auroc, "macro_AP": macro_ap},
        "segment_level": {"mAP": map_res, "Recall": recall_res},
        "config": vars(args)
    }
    with open(paths.out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    txt_summary = (
        "X-CLIP Evaluation Summary\n=========================\n"
        f"Macro AUROC: {macro_auroc:.4f}\n"
        f"Macro AP (AUPRC): {macro_ap:.4f}\n"
    )
    for k, v in map_res.items(): txt_summary += f"{k}: {v:.4f}\n"
    for k, v in recall_res.items(): txt_summary += f"{k}: {v:.4f}\n"
    
    with open(paths.out_dir / "metrics.txt", "w") as f:
        f.write(txt_summary)
    
    print("\n" + txt_summary)
    print(f"[OK] Wrote metrics to: {paths.out_dir}")

if __name__ == "__main__":
    main()
