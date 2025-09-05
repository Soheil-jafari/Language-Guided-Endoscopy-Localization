import numpy as np
import torch
from tqdm import tqdm
from .utils import RunningMeter, get_iou, span_cxw_to_xx

# --------------------------
# TRAIN
# --------------------------
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
    metrics = {
        'R1@0.5': [], 'R1@0.7': [],        # using confidence-selected best
        'mAP@0.5': [], 'mAP': [],
        'R1@0.5_oracle': [], 'R1@0.7_oracle': []  # using best IoU over ALL queries (no confidence)
    }

    for data in loader:
        video_feats = data['video_feats'].to(device)
        video_mask  = data['video_mask'].to(device)
        query       = data['query'].to(device)
        query_mask  = data['query_mask'].to(device)

        outputs     = model(video_feats, video_mask, query, query_mask)
        pred_spans  = outputs['pred_spans']     # [B, Q, 2] in [cx, w], normalized
        pred_logits = outputs['pred_logits']    # [B, Q, C] or [B, Q, 1]

        # ---- confidence-of-being-foreground (robust)
        if pred_logits.shape[-1] == 1:
            conf = pred_logits.sigmoid().squeeze(-1)             # [B, Q]
        else:
            probs = pred_logits.softmax(-1)                       # [B, Q, C]
            if probs.shape[-1] >= 2:
                conf = probs[..., :-1].max(dim=-1).values         # max over all foreground classes
            else:
                # degenerate case C==1: treat as foreground prob
                conf = probs[..., 0]

        best_indices = conf.argmax(dim=1)                         # [B]

        B = pred_spans.shape[0]
        for i in range(B):
            if data['targets'][i]['spans'].numel() == 0:
                continue

            # [cx,w] -> [x1,x2], clamp, rescale
            pred_xx = span_cxw_to_xx(pred_spans[i]).clamp(0, 1).cpu().numpy()  # [Q, 2]
            gt_xx   = span_cxw_to_xx(data['targets'][i]['spans'].cpu()).numpy()# [G, 2]
            dur     = 1
            pred    = pred_xx * dur
            gt      = gt_xx * dur

            # -------- ORACLE RECALL (ignores confidence) --------
            ious_all = get_iou(pred, gt)                 # [Q, G]
            max_iou  = ious_all.max() if ious_all.size else 0.0
            metrics['R1@0.5_oracle'].append(max_iou >= 0.5)
            metrics['R1@0.7_oracle'].append(max_iou >= 0.7)

            # -------- Recall using confidence-selected best -------
            bi        = int(best_indices[i].item())
            best_pred = pred[bi:bi+1]                    # [1, 2]
            iou_best  = get_iou(best_pred, gt)           # [1, G]
            best = float(iou_best.max()) if iou_best.size else 0.0
            metrics['R1@0.5'].append(best >= 0.5)
            metrics['R1@0.7'].append(best >= 0.7)

            # -------- mAP with confidence ordering ---------------
            conf_i = conf[i].detach().cpu().numpy()      # [Q]
            ap_05  = calculate_ap_with_scores(pred, gt, conf_i, 0.5)
            ap_075 = calculate_ap_with_scores(pred, gt, conf_i, 0.75)
            ap_095 = calculate_ap_with_scores(pred, gt, conf_i, 0.95)
            metrics['mAP@0.5'].append(ap_05)
            metrics['mAP'].append(np.mean([ap_05, ap_075, ap_095]))

    for k in metrics:
        metrics[k] = float(np.mean(metrics[k])) if len(metrics[k]) > 0 else 0.0
    return metrics


def calculate_ap_with_scores(pred, gt, conf, iou_thresh):
    if pred.size == 0:
        return 0.0
    order = np.argsort(-conf)                 # sort by confidence desc
    pred_sorted = pred[order]
    ious = get_iou(pred_sorted, gt)           # [N, M]
    tp = (ious >= iou_thresh).any(axis=1)     # [N]
    if tp.sum() == 0:
        return 0.0
    precision = np.cumsum(tp) / (np.arange(len(tp)) + 1)
    return float(precision.mean())
