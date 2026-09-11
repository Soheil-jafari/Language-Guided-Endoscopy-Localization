"""Evaluate explicit frame predictions; tune thresholds on validation only.

CSV columns: video_id,text_query,frame_idx,score,label. Predictions must cover the
same explicitly labelled frame keys as a separate reference CSV. The reference
does not need a score column. No directory searching or fallback model artifacts.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from checkpoint_utils import file_hash
from metrics import binary_metrics,deduplicate_frame_rows,validation_threshold,temporal_ap,temporal_iou,uncertainty_aurc


def checked_frames(predictions,reference):
    p=pd.read_csv(predictions); r=pd.read_csv(reference)
    keys=['video_id','text_query','frame_idx']
    if not set(keys+['score']).issubset(p) or not set(keys+['label']).issubset(r): raise ValueError('Missing frame columns')
    if p.duplicated(keys).any() or r.duplicated(keys).any(): raise ValueError('Deduplicate window outputs before evaluation')
    columns=keys+['score']+(['evidence_uncertainty'] if 'evidence_uncertainty' in p else [])
    merged=r.merge(p[columns],on=keys,how='outer',indicator=True,validate='one_to_one')
    if not (merged['_merge']=='both').all(): raise ValueError('Prediction/reference frame keys do not match exactly')
    return merged


def segment_metrics(records,threshold=.5):
    predictions={}; truth={}; absent=[]; r1={t:[] for t in [.3,.5,.7]}
    for row in records:
        key=(row['video_id'],row['text_query'])
        if key in truth: raise ValueError(f'Duplicate video-query: {key}')
        gt=row['ground_truth']; pred=row['predictions']; duration=float(row['duration'])
        for span in gt+pred:
            if not np.isfinite(span).all() or not 0<=span[0]<span[1]<=duration: raise ValueError(f'Invalid span {key}: {span}')
        if any(len(x)!=3 or not 0<=x[2]<=1 for x in pred): raise ValueError('Predictions require [start,end,score]')
        predictions[key]=pred; truth[key]=gt
        top=max(pred,key=lambda x:x[2],default=None)
        if gt:
            overlap=max((temporal_iou(top,g) for g in gt),default=0) if top else 0
            for t in r1: r1[t].append(overlap>=t)
        else: absent.append(any(x[2]>=threshold for x in pred))
    if not records: raise ValueError('No segment records')
    result={f'pooled_AP@{t}':temporal_ap(predictions,truth,t) for t in r1}
    result.update({f'R1@{t}':float(np.mean(v)) if v else None for t,v in r1.items()})
    result['absent_query_false_positive_rate']=float(np.mean(absent)) if absent else None
    result['absent_query_count']=len(absent)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--predictions',required=True)
    p.add_argument('--reference'); p.add_argument('--output',required=True)
    p.add_argument('--split',choices=['validation','test'],required=True)
    p.add_argument('--select-threshold',action='store_true'); p.add_argument('--calibration')
    p.add_argument('--threshold',type=float,default=.5); p.add_argument('--segments',action='store_true')
    a=p.parse_args()
    if Path(a.output).exists(): raise FileExistsError(a.output)
    if a.select_threshold and (a.split!='validation' or a.segments): raise ValueError('Threshold selection requires validation frame predictions')
    if a.calibration:
        saved=json.loads(Path(a.calibration).read_text())
        if saved.get('split')!='validation': raise ValueError('Calibration must come from validation')
        a.threshold=saved['threshold']
    if a.segments:
        records=[json.loads(line) for line in Path(a.predictions).read_text().splitlines() if line.strip()]
        if not a.reference: raise ValueError('--reference is required to verify every segment video/query pair')
        refs=[json.loads(line) for line in Path(a.reference).read_text().splitlines() if line.strip()]
        expected={(r.get('video',r.get('video_id')),r.get('query',r.get('text_query'))):r for r in refs}
        actual={(r['video_id'],r['text_query']):r for r in records}
        if len(expected)!=len(refs) or len(actual)!=len(records) or set(expected)!=set(actual):
            raise ValueError('Segment prediction/reference pairs do not match exactly')
        for key,row in actual.items():
            ref=expected[key]
            if not np.isclose(row['duration'],ref['duration']): raise ValueError('Segment duration mismatch')
            row['ground_truth']=ref.get('timestamps',ref.get('ground_truth'))
            if row['ground_truth'] is None: raise ValueError('Reference lacks ground truth')
        result=segment_metrics(records,a.threshold)
    else:
        if not a.reference: raise ValueError('--reference is required for frame evaluation')
        df=checked_frames(a.predictions,a.reference)
        if a.select_threshold: a.threshold=validation_threshold(df.label,df.score)
        result=binary_metrics(df.label,df.score,a.threshold)
        if 'evidence_uncertainty' in df:
            result['evidence_uncertainty_aurc']=uncertainty_aurc(df.label,df.score,df.evidence_uncertainty,a.threshold)
        result['per_video']={str(v):binary_metrics(g.label,g.score,a.threshold) for v,g in df.groupby('video_id') if (g.label!=-100).any()}
        result['per_query']={str(q):binary_metrics(g.label,g.score,a.threshold) for q,g in df.groupby('text_query') if (g.label!=-100).any()}
    result.update(split=a.split,threshold=a.threshold,predictions_sha256=file_hash(a.predictions))
    if a.reference: result['reference_sha256']=file_hash(a.reference)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
