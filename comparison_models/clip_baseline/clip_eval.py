import argparse, os, re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

def parse_vid_and_idx(p):
    vid = os.path.basename(os.path.dirname(p))          # e.g. CHOLEC80__video01
    m = re.search(r"frame_(\d+)\.jpg$", os.path.basename(p), flags=re.IGNORECASE)
    fidx = int(m.group(1)) if m else 0
    return vid, fidx

def sweep_best_f1(y_true, scores):
    # sweep 200 thresholds over [min, max]
    lo, hi = float(np.min(scores)), float(np.max(scores))
    thr_list = np.linspace(lo, hi, 200)
    best_f1, best_thr, acc_at_best = 0.0, 0.5, 0.0
    for thr in thr_list:
        y_pred = (scores >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            acc_at_best = accuracy_score(y_true, y_pred)
    return best_f1, best_thr, acc_at_best

def main():
    ap = argparse.ArgumentParser(description="Evaluate CLIP per-frame scores against parsed_annotations.")
    ap.add_argument("--scores_csv", required=True, help="CSV from clip_baseline.py (frame_path,score,frame_idx).")
    ap.add_argument("--parsed_ann_csv", required=True,
                    help=".../parsed_annotations/CHOLEC80_parsed_annotations.csv")
    ap.add_argument("--text_query", required=True, help="The text that was scored (must match parsed annotations).")
    args = ap.parse_args()

    df = pd.read_csv(args.scores_csv)
    if "frame_path" not in df or "score" not in df:
        raise ValueError("scores_csv must have columns: frame_path, score (and frame_idx preferred).")

    # Extract standardized_video_id & frame_idx from paths
    vids, idxs = zip(*[parse_vid_and_idx(p) for p in df["frame_path"].tolist()])
    df["standardized_video_id"] = vids
    df["frame_idx_from_path"]   = idxs

    # Load parsed annotations and filter to the exact text_query used
    ann = pd.read_csv(args.parsed_ann_csv)
    # Columns expected from your printout:
    # standardized_video_id, frame_idx, original_label, text_query, relevance_label
    # Keep only rows where text_query matches exactly the one you scored
    ann_q = ann[ann["text_query"].astype(str) == args.text_query].copy()

    # Join on (video, frame_idx)
    merged = pd.merge(
        df,
        ann_q[["standardized_video_id","frame_idx","relevance_label"]],
        left_on=["standardized_video_id","frame_idx_from_path"],
        right_on=["standardized_video_id","frame_idx"],
        how="inner"
    )
    if merged.empty:
        raise RuntimeError("No overlap between scores and parsed annotations for this text_query.")

    y_true = merged["relevance_label"].astype(int).to_numpy()
    scores = merged["score"].to_numpy()

    # Metrics
    try:
        auroc = roc_auc_score(y_true, scores)
    except ValueError:
        auroc = float("nan")
    try:
        auprc = average_precision_score(y_true, scores)
    except ValueError:
        auprc = float("nan")

    best_f1, thr_star, acc_at_thr = sweep_best_f1(y_true, scores)

    # Also plain accuracy at 0.5 (optional)
    acc_05 = accuracy_score(y_true, (scores >= 0.5).astype(int))

    print("\n=== CLIP Evaluation ===")
    print(f"text_query      : {args.text_query}")
    print(f"samples         : {len(y_true)}")
    print(f"AUROC           : {auroc:.3f}")
    print(f"AUPRC           : {auprc:.3f}")
    print(f"Best F1         : {best_f1:.3f} at thr={thr_star:.3f}")
    print(f"Acc@thr*        : {acc_at_thr:.3f}")
    print(f"Acc@0.5         : {acc_05:.3f}")

    # One-liner row you can paste in a table
    print("\nTABLE ROW →")
    print({
        "model": "CLIP",
        "ckpt": os.path.basename(args.scores_csv),
        "text_query": args.text_query,
        "N": len(y_true),
        "AUROC": round(auroc, 3),
        "AUPRC": round(auprc, 3),
        "BestF1": round(best_f1, 3),
        "thr*": round(thr_star, 3),
        "Acc@thr*": round(acc_at_thr, 3),
        "Acc@0.5": round(acc_05, 3),
    })

if __name__ == "__main__":
    main()

