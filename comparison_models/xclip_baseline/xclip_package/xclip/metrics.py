\
from typing import List, Tuple, Dict
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

def safe_auc_roc(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    # If only one class present, AUROC is undefined; return 0.5
    if len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_score)

def ap_auprc(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        # If positives absent, AP equals prevalence (0); if negatives absent, AP = 1
        return float(np.mean(y_true))
    return average_precision_score(y_true, y_score)

def precision_recall_points(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    p, r, th = precision_recall_curve(y_true, y_score)
    return p, r, th

# ---------- Temporal IoU & segment metrics ----------

def tiou(pred: Tuple[int,int], gt: Tuple[int,int]) -> float:
    ps, pe = pred
    gs, ge = gt
    inter = max(0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return 0.0 if union <= 0 else inter / union

def merge_binary_sequence(y: List[int], min_gap:int=0, min_len:int=1) -> List[Tuple[int,int]]:
    """
    Turn a binary frame array into segments [start, end) with optional gap closing and min length.
    """
    segs = []
    in_seg = False
    start = 0
    for i, v in enumerate(y + [0]): # sentinel
        if v and not in_seg:
            in_seg = True
            start = i
        elif not v and in_seg:
            end = i
            segs.append((start, end))
            in_seg = False
    # merge short gaps
    if min_gap > 0 and segs:
        merged = [segs[0]]
        for s,e in segs[1:]:
            ps,pe = merged[-1]
            if s - pe <= min_gap:
                merged[-1] = (ps, e)
            else:
                merged.append((s,e))
        segs = merged
    # filter short segments
    segs = [(s,e) for (s,e) in segs if (e - s) >= min_len]
    return segs

def greedy_match(preds: List[Tuple[Tuple[int,int], float]], gts: List[Tuple[int,int]], thr: float) -> Tuple[int,int,int]:
    """
    Greedy one-to-one matching by descending score. Returns TP, FP, FN given a threshold.
    """
    used = set()
    tp = 0
    fp = 0
    for (interval, score) in sorted(preds, key=lambda x: -x[1]):
        hit = False
        for gi, gt in enumerate(gts):
            if gi in used: continue
            if tiou(interval, gt) >= thr:
                tp += 1
                used.add(gi)
                hit = True
                break
        if not hit:
            fp += 1
    fn = len(gts) - len(used)
    return tp, fp, fn

def ap_at_tiou(preds: List[Tuple[Tuple[int,int], float]], gts: List[Tuple[int,int]], thr: float) -> float:
    """
    Compute AP for a single (video, query) at tIoU threshold.
    We rank predictions by score; precision-recall are traced by thresholding the ranked list.
    """
    if len(gts) == 0:
        return 0.0
    # sort predictions desc
    ranked = sorted(preds, key=lambda x: -x[1])
    tps = []
    fps = []
    used = set()
    for interval, score in ranked:
        matched = False
        for gi, gt in enumerate(gts):
            if gi in used: continue
            if tiou(interval, gt) >= thr:
                tps.append(1); fps.append(0); used.add(gi); matched = True; break
        if not matched:
            tps.append(0); fps.append(1)
    if not tps:
        return 0.0
    tps = np.array(tps).cumsum()
    fps = np.array(fps).cumsum()
    recalls = tps / max(len(gts), 1)
    precisions = tps / np.maximum(tps + fps, 1e-12)
    # standard 11-point-like interpolation via VOC-style area under PR
    ap = 0.0
    # Append sentinels
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre)-1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    # Compute area
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1]))
    return ap

def map_at_tious(preds_by_item: Dict, gts_by_item: Dict, thresholds=(0.3,0.5,0.7)) -> Dict[str,float]:
    """
    preds_by_item: { key -> [( (s,e), score ), ...] }
    gts_by_item:   { key -> [ (s,e), ... ] }
    Returns dict of mAP at each threshold.
    """
    res = {}
    for thr in thresholds:
        aps = []
        for k in gts_by_item.keys():
            aps.append(ap_at_tiou(preds_by_item.get(k, []), gts_by_item[k], thr))
        res[f"mAP@{thr}"] = float(np.mean(aps) if aps else 0.0)
    return res

def recall_at_k(preds_by_item: Dict, gts_by_item: Dict, k_list=(1,5), thr=0.5) -> Dict[str,float]:
    recalls = {}
    for k in k_list:
        vals = []
        for key, gts in gts_by_item.items():
            preds = sorted(preds_by_item.get(key, []), key=lambda x: -x[1])[:k]
            hit = 0
            for (interval, score) in preds:
                if any(tiou(interval, gt) >= thr for gt in gts):
                    hit = 1; break
            vals.append(hit)
        recalls[f"R@{k}"] = float(np.mean(vals) if vals else 0.0)
    return recalls
