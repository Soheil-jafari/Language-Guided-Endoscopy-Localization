
import argparse, os, re
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

def parse_vid_and_idx(p):
    vid = os.path.basename(os.path.dirname(p))          # e.g. CHOLEC80__video01
    m = re.search(r"frame_(\d+)\.jpg$", os.path.basename(p), flags=re.IGNORECASE)
    fidx = int(m.group(1)) if m else 0
    return vid, fidx

def _segments_from_binary(binary_indices):
    """
    Given a sorted list/array of indices where label==1,
    return contiguous [start, end] (inclusive) segments.
    """
    segments = []
    if len(binary_indices) == 0:
        return segments
    start = prev = int(binary_indices[0])
    for idx in binary_indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
        else:
            segments.append((start, prev))
            start = prev = idx
    segments.append((start, prev))
    return segments

def segments_from_series(frame_idx, mask_01, scores=None):
    """
    Build segments from 0/1 mask per frame.
    If scores is provided, the segment confidence is max(score) inside the segment; else 1.0.
    Returns list of dicts: {"start": s, "end": e, "conf": c}
    """
    assert len(frame_idx) == len(mask_01)
    pos_idx = np.array(frame_idx)[mask_01.astype(bool)]
    segs = _segments_from_binary(pos_idx)
    out = []
    for s, e in segs:
        if scores is not None:
            # Find indices within s..e to compute segment confidence
            m = (frame_idx >= s) & (frame_idx <= e)
            conf = float(np.max(scores[m])) if np.any(m) else 0.0
        else:
            conf = 1.0
        out.append({"start": int(s), "end": int(e), "conf": float(conf)})
    return out

def tiou(segA, segB):
    a0, a1 = segA["start"], segA["end"]
    b0, b1 = segB["start"], segB["end"]
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = (a1 - a0 + 1) + (b1 - b0 + 1) - inter
    return inter / union if union > 0 else 0.0

def prf_at_tiou(preds, gts, thr):
    """
    Greedy matching (confidence-descending) to compute TP/FP/FN at a given tIoU threshold.
    preds: list of segments with 'conf'
    gts: list of segments
    """
    preds_sorted = sorted(preds, key=lambda x: -x["conf"])
    matched_gt = set()
    tp, fp = 0, 0
    for i, p in enumerate(preds_sorted):
        best_j, best_iou = -1, 0.0
        for j, g in enumerate(gts):
            if j in matched_gt:
                continue
            iou = tiou(p, g)
            if iou >= thr and iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0:
            tp += 1
            matched_gt.add(best_j)
        else:
            fp += 1
    fn = len(gts) - len(matched_gt)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0.0
    return tp, fp, fn, prec, rec, f1

def ap_at_tiou(preds, gts, thr):
    """
    Compute Average Precision for one query (or group) at tIoU thr.
    Uses confidence-ranked predictions and classic precision-recall AP computation.
    """
    if len(preds) == 0 and len(gts) == 0:
        return 1.0
    if len(gts) == 0:
        # If no GT, define AP=1.0 when there are also no predictions; otherwise 0 (all predictions are FP).
        return 0.0 if len(preds) > 0 else 1.0

    preds_sorted = sorted(preds, key=lambda x: -x["conf"])
    matched_gt = set()
    tp_seq, fp_seq = [], []
    for p in preds_sorted:
        # Best GT match
        best_j, best_iou = -1, 0.0
        for j, g in enumerate(gts):
            if j in matched_gt:
                continue
            iou = tiou(p, g)
            if iou >= thr and iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0:
            matched_gt.add(best_j)
            tp_seq.append(1)
            fp_seq.append(0)
        else:
            tp_seq.append(0)
            fp_seq.append(1)

    if len(tp_seq) == 0:
        return 0.0

    tp_cum = np.cumsum(tp_seq)
    fp_cum = np.cumsum(fp_seq)
    rec = tp_cum / max(1, len(gts))
    prec = tp_cum / np.maximum(1, tp_cum + fp_cum)

    # Interpolated AP (VOC-style)
    # Append (0,1) and (1,0) endpoints for safety
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    # Sum over recall steps where recall changes
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)

def main():
    ap = argparse.ArgumentParser(description="Temporal localization metrics (tIoU, mAP@{0.3,0.5,0.7}, PRF) from per-frame scores + parsed annotations.")
    ap.add_argument("--scores_csv", required=True, help="CSV from clip_baseline.py (frame_path,score,frame_idx). May include multiple videos.")
    ap.add_argument("--parsed_ann_csv", required=True, help=".../parsed_annotations/CHOLEC80_parsed_annotations.csv or equivalent with per-frame relevance.")
    ap.add_argument("--text_query", required=True, help="The text that was scored (must match parsed annotations).")
    ap.add_argument("--threshold", type=float, default=None, help="Decision threshold for binarizing scores into segments. If omitted, will sweep and choose thr* that maximizes F1 on per-frame labels.")
    ap.add_argument("--tiou_list", type=str, default="0.3,0.5,0.7", help="Comma-separated tIoU thresholds.")
    args = ap.parse_args()

    df = pd.read_csv(args.scores_csv)
    if "frame_path" not in df or "score" not in df:
        raise ValueError("scores_csv must have columns: frame_path, score (and frame_idx preferred).")

    # Extract standardized_video_id & frame_idx from paths
    vids, idxs = zip(*[parse_vid_and_idx(p) for p in df["frame_path"].tolist()])
    df["standardized_video_id"] = vids
    df["frame_idx_from_path"]   = idxs

    ann = pd.read_csv(args.parsed_ann_csv)
    ann_q = ann[ann["text_query"].astype(str) == args.text_query].copy()

    merged = pd.merge(
        df,
        ann_q[["standardized_video_id","frame_idx","relevance_label"]],
        left_on=["standardized_video_id","frame_idx_from_path"],
        right_on=["standardized_video_id","frame_idx"],
        how="inner"
    ).sort_values(["standardized_video_id","frame_idx"])

    if merged.empty:
        raise RuntimeError("No overlap between scores and parsed annotations for this text_query.")

    # If no threshold provided, do a simple sweep to pick best per-frame F1 (like clip_eval)
    thr = args.threshold
    if thr is None:
        y_true = merged["relevance_label"].astype(int).to_numpy()
        scores = merged["score"].to_numpy()
        lo, hi = float(np.min(scores)), float(np.max(scores))
        thr_list = np.linspace(lo, hi, 200)
        best_f1, best_thr = 0.0, 0.5
        for t in thr_list:
            y_pred = (scores >= t).astype(int)
            tp = np.sum((y_true==1) & (y_pred==1))
            fp = np.sum((y_true==0) & (y_pred==1))
            fn = np.sum((y_true==1) & (y_pred==0))
            prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
            rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
            f1   = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
            if f1 > best_f1:
                best_f1, best_thr = f1, t
        thr = float(best_thr)

    tiou_list = [float(x) for x in args.tiou_list.split(",")]

    # Group by video; build GT and Pred segments per video, then pool for metrics
    results = {t: {"TP":0,"FP":0,"FN":0,"prec":0.0,"rec":0.0,"f1":0.0,"AP_list":[]} for t in tiou_list}

    for vid, g in merged.groupby("standardized_video_id", sort=True):
        frame_idx = g["frame_idx"].to_numpy()
        scores = g["score"].to_numpy()
        y_true = g["relevance_label"].astype(int).to_numpy()

        # Build GT segments from contiguous frames where y_true==1
        gt_segs = segments_from_series(frame_idx, y_true, scores=None)

        # Build predicted segments by thresholding scores
        y_pred_01 = (scores >= thr).astype(int)
        pred_segs = segments_from_series(frame_idx, y_pred_01, scores=scores)

        for t in tiou_list:
            tp, fp, fn, prec, rec, f1 = prf_at_tiou(pred_segs, gt_segs, t)
            results[t]["TP"] += tp
            results[t]["FP"] += fp
            results[t]["FN"] += fn
            results[t]["AP_list"].append(ap_at_tiou(pred_segs, gt_segs, t))

    # Aggregate PRF and mAP
    rows = []
    for t in tiou_list:
        TP, FP, FN = results[t]["TP"], results[t]["FP"], results[t]["FN"]
        prec = TP/(TP+FP) if (TP+FP)>0 else 0.0
        rec  = TP/(TP+FN) if (TP+FN)>0 else 0.0
        f1   = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
        mAP  = float(np.mean(results[t]["AP_list"])) if len(results[t]["AP_list"])>0 else 0.0
        rows.append({
            "tIoU": t, "mAP": round(mAP,3), "Precision": round(prec,3), "Recall": round(rec,3), "F1": round(f1,3)
        })

    print("\n=== Temporal Localization (segment-level) ===")
    print(f"text_query         : {args.text_query}")
    print(f"chosen threshold   : {thr:.3f} (scores>=thr → positive frames → segments)")
    print("\nPer tIoU results:")
    for r in rows:
        print(f"tIoU@{r['tIoU']:.2f} → mAP={r['mAP']:.3f}, P={r['Precision']:.3f}, R={r['Recall']:.3f}, F1={r['F1']:.3f}")

    # Compact dict for table use
    out = {"thr*": round(float(thr), 3)}
    for r in rows:
        t = r["tIoU"]
        out[f"mAP@{t}"] = r["mAP"]
        out[f"P@{t}"] = r["Precision"]
        out[f"R@{t}"] = r["Recall"]
        out[f"F1@{t}"] = r["F1"]
    print("\nTABLE ROW →")
    print(out)

if __name__ == "__main__":
    main()
