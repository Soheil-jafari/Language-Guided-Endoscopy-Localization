import json, torch, numpy as np
from moment_detr_module.configs import Config
from moment_detr_module.modeling import MomentDETR
from moment_detr_module.dataset import MomentDETRDataset, collate_fn
from moment_detr_module.engine import span_cxw_to_xx, get_iou

def main(ckpt, video_id, query, gt_start=None, gt_end=None, duration=None):
    device = torch.device("cuda", 0)
    cfg = Config()
    model = MomentDETR(cfg).to(device)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    # Build one sample using dataset tokenizer/features
    # We locate the annotation by video_id in the JSONL 'test' file:
    import os
    ann_file = os.path.join(cfg.ann_path, "test.jsonl")
    found = None
    with open(ann_file, "r", encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            if a["video"] == video_id:
                found = a; break
    assert found is not None, f"video_id {video_id} not found in test.jsonl"

    # Override query if provided
    found["query"] = query

    # Build dataset item like __getitem__ does
    ds = MomentDETRDataset(cfg, "test")
    # quick way: reuse dataset internals by faking a batch of size 1
    # (we’ll create the tensors the same way ds.__getitem__ does)
    from transformers import RobertaTokenizer
    tok = RobertaTokenizer.from_pretrained('roberta-base')

    import os
    feat = np.load(os.path.join(cfg.feature_path, f"{video_id}.npz"))["features"].astype(np.float32)
    if feat.shape[0] > cfg.max_v_len:
        idxs = np.linspace(0, feat.shape[0]-1, cfg.max_v_len).astype(int); feat = feat[idxs]
    word = tok(found["query"], add_special_tokens=True, max_length=cfg.max_q_len,
               padding='max_length', return_tensors='pt', truncation=True)

    video_feats = torch.from_numpy(feat)[None].to(device)        # [1,T,D]
    video_mask  = torch.ones((1, video_feats.shape[1]), dtype=torch.bool, device=device)
    query_ids   = word["input_ids"].long().to(device)
    query_mask  = word["attention_mask"].long().to(device)

    with torch.no_grad():
        out = model(video_feats, video_mask, query_ids, query_mask)
        pred_spans  = out["pred_spans"][0].detach().cpu()        # [Q,2] (cx,w)
        pred_logits = out["pred_logits"][0].detach().cpu()       # [Q,C or 1]

    # confidence score
    if pred_logits.shape[-1] == 1:
        conf = pred_logits.sigmoid().squeeze(-1).numpy()         # [Q]
    else:
        probs = pred_logits.softmax(-1).numpy()
        conf  = probs[..., :-1].max(axis=-1) if probs.shape[-1] >= 2 else probs[...,0]

    # top1 prediction (after converting to [x1,x2] in [0,1])
    pred_xx = span_cxw_to_xx(pred_spans).clamp(0,1).numpy()      # [Q,2]
    best = int(conf.argmax())
    top1 = pred_xx[best:best+1]                                   # [1,2]

    # ground truth
    if gt_start is None or gt_end is None:
        # use test.jsonl timestamps if not supplied
        gt = np.array(found["timestamps"], dtype=np.float32) / float(found["duration"])
    else:
        assert duration is not None, "Provide 'duration' when giving raw seconds."
        gt = np.array([[gt_start, gt_end]], dtype=np.float32) / float(duration)

    # tIoU numbers
    tiou = float(get_iou(top1, gt).max()) if gt.size else 0.0
    print({
        "video_id": video_id,
        "query": query,
        "top1_pred_norm": top1.tolist(),   # [x1,x2] in [0,1]
        "tIoU_top1_vs_GT": tiou,
        "top1_confidence": float(conf[best]),
    })

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--video_id", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--gt_start", type=float)
    ap.add_argument("--gt_end",   type=float)
    ap.add_argument("--duration", type=float)
    args = ap.parse_args()
    main(args.ckpt, args.video_id, args.query, args.gt_start, args.gt_end, args.duration)
