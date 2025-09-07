import numpy as np
import torch
from tqdm import tqdm
from .utils import RunningMeter, get_iou, span_cxw_to_xx

import math

def _frame_edges(num_tokens):
    # normalized [0,1] grid cell edges for each token
    g = 1.0 / float(num_tokens)
    starts = np.arange(num_tokens) * g
    ends   = starts + g
    return np.stack([starts, ends], axis=1)  # [T,2]

def _labels_from_gt(gt_xx, num_tokens):
    # y_true[t] = 1 if token t is inside any GT span
    edges = _frame_edges(num_tokens)         # [T,2]
    ious = get_iou(edges, gt_xx)             # [T,G]
    inside = (ious > 0).any(axis=1).astype(np.float32)
    return inside  # [T]

def _scores_from_preds(pred_xx, conf, num_tokens):
    # y_score[t] = max(conf_p * IoU(token, pred_p))  over proposals p
    edges = _frame_edges(num_tokens)         # [T,2]
    if pred_xx.size == 0:
        return np.zeros((num_tokens,), dtype=np.float32)
    ious = get_iou(edges, pred_xx)           # [T,Q]
    # conf: [Q] in [0,1]
    s = (ious * conf[None, :]).max(axis=1)   # [T]
    return s.astype(np.float32)

def _pr_points(y_true, y_score, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 19)  # 0.00 .. 1.00 step 0.05
    P = y_true.sum()
    pr = []
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(np.float32)
        TP = float((y_pred * y_true).sum())
        FP = float((y_pred * (1 - y_true)).sum())
        FN = float(((1 - y_pred) * y_true).sum())
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        acc  = (y_pred == y_true).mean() if y_true.size else 0.0
        f1   = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0.0
        pr.append((thr, prec, rec, f1, acc, TP, FP, FN))
    return pr  # list of tuples

def _average_precision(y_true, y_score):
    # AP = area under precision-recall curve (11-pt like VOC-ish but dense thresholds)
    pr = _pr_points(y_true, y_score, thresholds=np.linspace(0.0,1.0,101))
    # sort by recall ascending
    pr_sorted = sorted(pr, key=lambda x: x[2])
    recalls   = [x[2] for x in pr_sorted]
    precisions= [x[1] for x in pr_sorted]
    # trapezoid integration over recall
    ap = 0.0
    for i in range(1, len(recalls)):
        dr = recalls[i] - recalls[i-1]
        ap += dr * ((precisions[i] + precisions[i-1]) / 2.0)
    return float(ap)

def _auroc(y_true, y_score):
    # Fast AUC via rank statistic (Mann–Whitney U)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg])))
    # convert to 1-based ranks
    ranks = ranks + 1
    r_pos = ranks[:len(pos)].sum()
    n_pos, n_neg = float(len(pos)), float(len(neg))
    auc = (r_pos - n_pos*(n_pos+1)/2.0) / (n_pos * n_neg)
    return float(auc)

def train_one_epoch(model, loader, optimizer, device, epoch, max_norm, logger, rank):
    model.train()
    loss_meter = RunningMeter()

    progress_bar = tqdm(loader, desc=f"Epoch {epoch}", leave=False, disable=(rank != 0))
    for i, data in enumerate(progress_bar):
        video_feats  = data['video_feats'].to(device)
        video_mask   = data['video_mask'].to(device)
        query        = data['query'].to(device)
        query_mask   = data['query_mask'].to(device)
        targets      = [{k: v.to(device) for k, v in t.items()} for t in data['targets']]

        out  = model(video_feats, video_mask, query, query_mask, targets)
        loss = sum(out['loss_dict'].values())

        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        loss_meter.update(loss.item())
        progress_bar.set_description(f"Epoch {epoch} | Loss: {loss_meter.avg:.4f}")

        # ---- one-time debug on the first step of each epoch (rank 0 only)
        if i == 0 and rank == 0:
            ld = {k: float(v.detach().cpu()) for k, v in out['loss_dict'].items()}
            print(f"\n[DEBUG] epoch {epoch} loss_dict:", ld)
            if 'pred_logits' in out:
                pl = out['pred_logits'].detach().cpu()
                print("[DEBUG] pred_logits shape:", tuple(pl.shape),
                      "min/mean/max:", float(pl.min()), float(pl.mean()), float(pl.max()))
            ps = out['pred_spans'].detach().cpu()
            print("[DEBUG] pred_spans shape:", tuple(ps.shape),
                  "min/mean/max:", float(ps.min()), float(ps.mean()), float(ps.max()))
            print("[DEBUG] target spans (first sample):", data['targets'][0]['spans'][:1])
            print("[DEBUG] duration[0]=", float(data['meta']['duration'][0]),
                  "num_tokens=", int(data['video_feats'].shape[1]))


# --------------------------
# EVAL
# --------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    # detection-style thresholds (tIoU)
    T = [0.3, 0.5, 0.7]

    # accumulators for detection metrics
    map_at = {t: [] for t in T}
    r1_conf_at = {0.5: [], 0.7: []}
    r1_orac_at = {0.5: [], 0.7: []}
    tp_at  = {t: 0 for t in T}
    fp_at  = {t: 0 for t in T}
    fn_at  = {t: 0 for t in T}

    # accumulators for framewise classification metrics
    aurocs, aprs, acc_at_05 = [], [], []
    best_f1s, best_f1_thrs, acc_at_best = [], [], []

    for data in loader:
        video_feats = data['video_feats'].to(device)
        video_mask  = data['video_mask'].to(device)
        query       = data['query'].to(device)
        query_mask  = data['query_mask'].to(device)

        outputs     = model(video_feats, video_mask, query, query_mask)
        pred_spans  = outputs['pred_spans']     # [B, Q, 2] in [cx, w]
        pred_logits = outputs['pred_logits']    # [B, Q, C] or [B, Q, 1]
        B, Q = pred_spans.shape[:2]
        # confidence-of-being-foreground
        if pred_logits.shape[-1] == 1:
            conf_all = pred_logits.sigmoid().squeeze(-1)                # [B, Q]
        else:
            probs = pred_logits.softmax(-1)                              # [B, Q, C]
            conf_all = (probs[..., :-1].max(dim=-1).values
                        if probs.shape[-1] >= 2 else probs[..., 0])      # [B, Q]

        order_all = torch.argsort(conf_all, dim=1, descending=True)      # [B, Q]

        # iterate samples
        for i in range(B):
            gt_cw = data['targets'][i]['spans'].cpu()
            if gt_cw.numel() == 0:
                continue

            # ---- normalize and prepare ----
            gt_xx = span_cxw_to_xx(gt_cw).clamp(0, 1).numpy()            # [G,2]
            pred_xx   = span_cxw_to_xx(pred_spans[i]).clamp(0, 1).cpu().numpy()
            conf_i    = conf_all[i].detach().cpu().numpy()
            ord_i     = order_all[i].detach().cpu().numpy()
            pred_sorted = pred_xx[ord_i]
            conf_sorted = conf_i[ord_i]

            # --------- detection metrics ----------
            ious_all = get_iou(pred_xx, gt_xx)                           # [Q,G]
            max_iou  = ious_all.max() if ious_all.size else 0.0
            r1_orac_at[0.5].append(max_iou >= 0.5)
            r1_orac_at[0.7].append(max_iou >= 0.7)

            if Q > 0:
                best_pred = pred_sorted[:1]
                best_iou  = get_iou(best_pred, gt_xx)
                top1 = float(best_iou.max()) if best_iou.size else 0.0
                r1_conf_at[0.5].append(top1 >= 0.5)
                r1_conf_at[0.7].append(top1 >= 0.7)

            for t in T:
                # mAP @ t
                map_at[t].append(calculate_ap_with_scores(pred_sorted, gt_xx, conf_sorted, t))

                # TP/FP/FN via greedy matching
                ious = get_iou(pred_sorted, gt_xx)                       # [Q,G]
                matched_gt = set()
                tp = fp = 0
                for k in range(len(pred_sorted)):
                    j = int(np.argmax(ious[k])) if ious.shape[1] > 0 else -1
                    best = ious[k, j] if j >= 0 else 0.0
                    if j >= 0 and best >= t and j not in matched_gt:
                        tp += 1
                        matched_gt.add(j)
                    else:
                        fp += 1
                fn = max(0, gt_xx.shape[0] - len(matched_gt))
                tp_at[t] += tp; fp_at[t] += fp; fn_at[t] += fn

            # --------- framewise classification metrics ----------
            Ttokens = int(video_feats.shape[1])
            y_true  = _labels_from_gt(gt_xx, Ttokens)                    # [T]
            y_score = _scores_from_preds(pred_xx, conf_i, Ttokens)       # [T]

            # AUROC, AUPRC(AP), Acc@0.5
            aurocs.append(_auroc(y_true, y_score))
            aprs.append(_average_precision(y_true, y_score))
            acc_at_05.append(((y_score >= 0.5).astype(np.float32) == y_true).mean())

            # Best F1 / Acc over thresholds
            pr = _pr_points(y_true, y_score, thresholds=np.linspace(0.05,0.95,19))
            best = max(pr, key=lambda x: x[3])  # maximize F1
            thr_star, f1_star, acc_star = best[0], best[3], best[4]
            best_f1s.append(f1_star); best_f1_thrs.append(thr_star); acc_at_best.append(acc_star)

    # ----- reduce to scalars -----
    results = {}

    # detection metrics
    for t in T:
        results[f"mAP@{t}"] = float(np.mean(map_at[t])) if map_at[t] else 0.0
    results["mAP"] = float(np.mean([results[f"mAP@{t}"] for t in T])) if T else 0.0
    for t in T:
        TP, FP, FN = tp_at[t], fp_at[t], fn_at[t]
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1   = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0.0
        results[f"TP@{t}"] = int(TP); results[f"FP@{t}"] = int(FP); results[f"FN@{t}"] = int(FN)
        results[f"Precision@{t}"] = float(prec); results[f"Recall@{t}"] = float(rec); results[f"F1@{t}"] = float(f1)

    results["R1@0.5"]        = float(np.mean(r1_conf_at[0.5])) if r1_conf_at[0.5] else 0.0
    results["R1@0.7"]        = float(np.mean(r1_conf_at[0.7])) if r1_conf_at[0.7] else 0.0
    results["R1@0.5_oracle"] = float(np.mean(r1_orac_at[0.5])) if r1_orac_at[0.5] else 0.0
    results["R1@0.7_oracle"] = float(np.mean(r1_orac_at[0.7])) if r1_orac_at[0.7] else 0.0

    # framewise classification metrics
    results["AUROC"]          = float(np.mean(aurocs)) if aurocs else 0.0
    results["AUPRC"]          = float(np.mean(aprs))   if aprs else 0.0  # Average Precision
    results["Acc@0.5"]        = float(np.mean(acc_at_05)) if acc_at_05 else 0.0
    results["BestF1"]         = float(np.mean(best_f1s)) if best_f1s else 0.0
    results["BestF1_thr"]     = float(np.mean(best_f1_thrs)) if best_f1_thrs else 0.0
    results["Acc@BestF1_thr"] = float(np.mean(acc_at_best)) if acc_at_best else 0.0

    return results
def calculate_ap_with_scores(pred, gt, conf, iou_thresh):
    """
    pred: np.ndarray [Q, 2] (start,end) in the same normalized units as gt
    gt:   np.ndarray [G, 2]
    conf: np.ndarray [Q] confidence scores for each pred
    iou_thresh: float (e.g., 0.3 / 0.5 / 0.7)

    Returns per-sample AP (area under precision–recall) at the given tIoU threshold,
    using greedy 1-to-1 matching while traversing predictions in descending confidence.
    """
    # Edge cases
    if pred.size == 0:
        return 0.0
    if gt.size == 0:
        # No positives in GT: AP is undefined; treat as 0.0 for consistency
        return 0.0

    # Sort predictions by confidence (desc)
    order = np.argsort(-conf)
    pred = pred[order]
    conf = conf[order]

    # IoUs between each pred and each GT
    ious = get_iou(pred, gt)  # [Q, G]

    # Greedy matching
    G = gt.shape[0]
    matched_gt = np.zeros(G, dtype=bool)
    tp = np.zeros(len(pred), dtype=np.float32)
    fp = np.zeros(len(pred), dtype=np.float32)

    for i in range(len(pred)):
        if G == 0:
            fp[i] = 1.0
            continue
        j = int(np.argmax(ious[i]))  # best GT for this pred
        best_iou = float(ious[i, j]) if ious.size else 0.0
        if best_iou >= iou_thresh and not matched_gt[j]:
            tp[i] = 1.0
            matched_gt[j] = True
        else:
            fp[i] = 1.0

    # Precision–Recall curve
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    recall    = cum_tp / (G + 1e-9)

    # Make recall strictly increasing for integration
    # (optional but stabilizes small-sample AP)
    recall_prev = 0.0
    for k in range(len(recall)):
        if recall[k] < recall_prev:
            recall[k] = recall_prev
        recall_prev = recall[k]

    # Trapezoidal area under PR curve
    ap = 0.0
    for k in range(1, len(recall)):
        dr = recall[k] - recall[k - 1]
        ap += dr * (precision[k] + precision[k - 1]) * 0.5
    return float(ap)
