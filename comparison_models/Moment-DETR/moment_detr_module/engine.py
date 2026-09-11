"""Adapted DETR engine with shared temporal metrics and no test threshold search."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import json
import numpy as np
import torch
from metrics import temporal_ap
from evaluate import segment_metrics
from .utils import span_cxw_to_xx


def train_one_epoch(model,loader,optimizer,device,epoch,max_norm,logger=None,rank=0):
    model.train(); total=0.; count=0
    for data in loader:
        targets=[{k:v.to(device) for k,v in t.items()} for t in data['targets']]
        out=model(data['video_feats'].to(device),data['video_mask'].to(device),data['query'].to(device),data['query_mask'].to(device),targets)
        loss=sum(out['loss_dict'].values())
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite adapted DETR loss')
        optimizer.zero_grad(); loss.backward()
        if max_norm>0: torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm)
        optimizer.step(); total+=float(loss.detach())*len(targets); count+=len(targets)
    if not count: raise ValueError('Empty training loader')
    return total/count


@torch.no_grad()
def evaluate(model,loader,device,output_path=None,threshold=.5):
    model.eval(); records=[]
    for data in loader:
        out=model(data['video_feats'].to(device),data['video_mask'].to(device),data['query'].to(device),data['query_mask'].to(device))
        spans=span_cxw_to_xx(out['pred_spans']).clamp(0,1).cpu().numpy()
        scores=out['pred_logits'].softmax(-1)[...,0].cpu().numpy()
        for i in range(len(spans)):
            duration=float(data['meta']['duration'][i])
            pred=[[float(a*duration),float(b*duration),float(s)] for (a,b),s in zip(spans[i],scores[i]) if b>a]
            gt=(data['targets'][i]['segments']*duration).tolist()
            records.append(dict(video_id=data['meta']['video_id'][i],text_query=data['meta']['query'][i],duration=duration,
                                ground_truth=gt,predictions=pred))
    if output_path:
        path=Path(output_path)
        if path.exists(): raise FileExistsError(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    # Frame metrics are not fabricated on a padded 256-token grid. Export segments
    # and rasterize on the shared source-frame reference when comparison is needed.
    return segment_metrics(records,threshold)


def calculate_ap_with_scores(pred,gt,conf,iou_thresh):
    return temporal_ap({'pair':[list(p)+[float(s)] for p,s in zip(pred,conf)]},{'pair':np.asarray(gt).reshape(-1,2).tolist()},iou_thresh)
