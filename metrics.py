"""Shared metrics. Unknown labels (-100) are excluded, never changed to negatives."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_arrays(labels, scores):
    y, p = np.asarray(labels).reshape(-1), np.asarray(scores,dtype=float).reshape(-1)
    if y.shape != p.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Scores must match labels and be finite probabilities in [0,1]')
    if not np.isin(y,[-100,0,1]).all():
        raise ValueError('Invalid binary labels')
    mask = y != -100
    return y[mask].astype(int), p[mask]


def binary_metrics(labels, scores, threshold=0.5, bins=15):
    y,p = binary_arrays(labels,scores)
    if not 0 <= threshold <= 1 or bins <= 0:
        raise ValueError('Invalid threshold/bin count')
    if not len(y):
        raise ValueError('No observed labels to evaluate')
    pred = p >= threshold
    tp,fp,fn = np.sum(pred&(y==1)),np.sum(pred&(y==0)),np.sum(~pred&(y==1))
    ece = 0.
    for i in range(bins):
        m = (p>=i/bins)&((p<(i+1)/bins) if i<bins-1 else (p<=1))
        if m.any():
            ece += m.mean()*abs(p[m].mean()-y[m].mean())
    # Predicted-class confidence. Keep tied confidence groups together to avoid
    # arbitrary input-order dependence. Right-step area at achievable coverages.
    confidence = np.where(pred,p,1-p)
    order = np.argsort(-confidence,kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(confidence[order]) != 0)+1,len(y)]
    risks = np.cumsum((pred != y)[order])[ends-1]/ends
    aurc = np.sum(risks*np.diff(np.r_[0,ends/len(y)]))
    return dict(n=len(y), positives=int(y.sum()), accuracy=float((pred==y).mean()),
        precision=float(tp/max(tp+fp,1)), recall=float(tp/max(tp+fn,1)),
        f1=float(2*tp/max(2*tp+fp+fn,1)),
        auroc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
        average_precision=float(average_precision_score(y,p)) if y.sum() else None,
        brier=float(np.mean((p-y)**2)), ece=float(ece), aurc=float(aurc), threshold=float(threshold))


def uncertainty_aurc(labels,scores,uncertainty,threshold=.5):
    raw_y=np.asarray(labels); raw_u=np.asarray(uncertainty,dtype=float)
    if raw_y.shape!=raw_u.shape or not np.isfinite(raw_u).all() or np.any((raw_u<0)|(raw_u>1)):
        raise ValueError('Uncertainty must match labels and lie in [0,1]')
    y,p=binary_arrays(labels,scores); u=raw_u[raw_y!=-100]
    if not len(y): raise ValueError('No observed labels')
    order=np.argsort(u,kind='stable'); ends=np.r_[np.flatnonzero(np.diff(u[order])!=0)+1,len(y)]
    risk=np.cumsum(((p>=threshold)!=y)[order])[ends-1]/ends
    return float(np.sum(risk*np.diff(np.r_[0,ends/len(y)])))


def validation_threshold(labels,scores):
    y,p = binary_arrays(labels,scores)
    if len(np.unique(y)) != 2:
        raise ValueError('Threshold selection requires positive AND negative validation labels')
    candidates = np.unique(np.r_[0.,p,.5,1.])
    # Sort once; cumulative counts avoid quadratic scans on long videos.
    order=np.argsort(p,kind='stable'); sorted_p=p[order]
    prefix=np.r_[0,np.cumsum(y[order])]
    positions=np.searchsorted(sorted_p,candidates,side='left')
    tp=y.sum()-prefix[positions]; fp=len(y)-positions-tp; fn=y.sum()-tp
    f1=2*tp/np.maximum(2*tp+fp+fn,1)
    best=np.lexsort((candidates,-np.abs(candidates-.5),f1))[-1]
    return float(candidates[best])


def segments_from_scores(times,scores,duration,threshold=.5,min_duration=0.,merge_gap=0.):
    times=np.asarray(times,dtype=float); scores=np.asarray(scores,dtype=float)
    if len(times)!=len(scores) or not len(times) or np.any(np.diff(times)<=0) or times[0]<0 or times[-1]>=duration:
        raise ValueError('Invalid sample times or video duration')
    if not np.isfinite(times).all() or not np.isfinite(scores).all() or not np.isfinite(duration) or min_duration<0 or merge_gap<0:
        raise ValueError('Invalid segmentation values')
    if np.any((scores<0)|(scores>1)) or not 0<=threshold<=1:
        raise ValueError('Expected probabilities and probability threshold')
    # Left-closed cells [sample time, next sample time), final cell ends at duration.
    ends=np.r_[times[1:],duration]
    groups=[]
    for i in np.flatnonzero(scores>=threshold):
        a,b=float(times[i]),float(ends[i])
        if groups and a-groups[-1][1] <= merge_gap+1e-9:
            groups[-1][1]=b; groups[-1][2]=max(groups[-1][2],float(scores[i]))
        else:
            groups.append([a,b,float(scores[i])])
    return [x for x in groups if x[1]-x[0]>=min_duration]


def temporal_iou(a,b):
    intersection=max(0.,min(a[1],b[1])-max(a[0],b[0]))
    return intersection/max(a[1]-a[0]+b[1]-b[0]-intersection,1e-12)


def temporal_ap(predictions,ground_truth,iou_threshold=.5):
    """Pooled AP over explicitly supplied video-query pairs; each item is a span list.
    Pred spans are [start,end,score], GT spans [start,end]. Empty GT still costs FP.
    """
    if set(predictions)!=set(ground_truth):
        raise ValueError('Prediction and ground-truth pair sets must match (use explicit empty lists)')
    total=sum(len(v) for v in ground_truth.values())
    if not total:
        return None
    ranked=sorted(((p[2],key,j,p) for key,ps in predictions.items() for j,p in enumerate(ps)), key=lambda x:(-x[0],str(x[1]),x[2]))
    used={key:set() for key in ground_truth}; hits=[]
    for _,key,_,p in ranked:
        choices=[(temporal_iou(p,g),i) for i,g in enumerate(ground_truth[key]) if i not in used[key]]
        best=max(choices,default=(-1,-1))
        hit=best[0]>=iou_threshold
        hits.append(hit)
        if hit: used[key].add(best[1])
    if not hits: return 0.
    tp=np.cumsum(hits); recall=tp/total; precision=tp/np.arange(1,len(hits)+1)
    recall=np.r_[0,recall,1]; precision=np.r_[0,precision,0]
    precision=np.maximum.accumulate(precision[::-1])[::-1]
    changes=np.flatnonzero(np.diff(recall)>0)
    return float(np.sum((recall[changes+1]-recall[changes])*precision[changes+1]))


def deduplicate_frame_rows(rows):
    """Average overlap predictions once per video/query/source-frame; reject label conflicts."""
    import pandas as pd
    df=pd.DataFrame(rows)
    keys=['video_id','text_query','frame_idx']
    if df.empty: raise ValueError('No predictions')
    if df.groupby(keys,dropna=False).label.nunique().max()>1:
        raise ValueError('Conflicting labels for an overlapping frame')
    aggregation=dict(score=('score','mean'),label=('label','first'))
    if 'evidence_uncertainty' in df: aggregation['evidence_uncertainty']=('evidence_uncertainty','mean')
    return df.groupby(keys,as_index=False,sort=True).agg(**aggregation)
