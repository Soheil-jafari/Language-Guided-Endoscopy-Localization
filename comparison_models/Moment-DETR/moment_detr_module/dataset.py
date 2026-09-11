import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer


class MomentDETRDataset(Dataset):
    def __init__(self,cfg,split):
        self.cfg=cfg; self.split=split
        self.tokenizer=RobertaTokenizer.from_pretrained('roberta-base')
        self.annotations=[json.loads(s) for s in (Path(cfg.ann_path)/(split+'.jsonl')).read_text().splitlines() if s.strip()]
        if not self.annotations: raise ValueError('Empty segment manifest')
        keys=[(a['video'],a['query']) for a in self.annotations]
        if len(keys)!=len(set(keys)): raise ValueError('Group ALL spans into one record per video/query')

    def __len__(self): return len(self.annotations)

    def __getitem__(self,idx):
        ann=self.annotations[idx]
        with np.load(Path(self.cfg.feature_path)/(ann['video']+'.npz')) as data:
            if not {'features','frame_idx','time_sec','duration','sample_fps'}.issubset(data.files):
                raise ValueError('Regenerate features with source-frame metadata')
            features=data['features'].astype(np.float32); times=data['time_sec']; duration=float(data['duration'])
            if not np.isclose(float(data['sample_fps']),ann['sample_fps']): raise ValueError('Feature/annotation sampling-rate mismatch')
        if features.ndim!=2 or len(features)==0 or len(times)!=len(features) or not np.isfinite(features).all():
            raise ValueError('Invalid feature sequence')
        if not np.isclose(duration,ann['duration']) or np.any(np.diff(times)<=0): raise ValueError('Feature/annotation duration mismatch')
        if len(features)>self.cfg.max_v_len:
            indices=np.linspace(0,len(features)-1,self.cfg.max_v_len).round().astype(int)
            features=features[indices]; times=times[indices]
        spans=torch.as_tensor(ann['timestamps'],dtype=torch.float32).reshape(-1,2)/duration
        if len(spans)>self.cfg.max_v_len:
            raise ValueError('More ground-truth segments than proposal slots; increase max_v_len before training')
        if duration<=0 or not torch.isfinite(spans).all() or (spans<0).any() or (spans>1).any() or (spans[:,0]>=spans[:,1]).any():
            raise ValueError('Invalid ground truth spans')
        words=self.tokenizer(ann['query'],max_length=self.cfg.max_q_len,padding='max_length',return_tensors='pt',truncation=True)
        return dict(video_feats=torch.from_numpy(features),query=words['input_ids'][0],query_mask=words['attention_mask'][0],
            raw_spans=spans,duration=duration,video_id=ann['video'],query_str=ann['query'],time_sec=torch.tensor(times))


def collate_fn(batch):
    feats=torch.zeros(len(batch),max(len(x['video_feats']) for x in batch),batch[0]['video_feats'].shape[1])
    mask=torch.zeros(feats.shape[:2],dtype=torch.bool); targets=[]
    for i,item in enumerate(batch):
        n=len(item['video_feats']); feats[i,:n]=item['video_feats']; mask[i,:n]=True
        spans=item['raw_spans'].reshape(-1,2).float()
        # Ground truth is independent of batch padding and feature downsampling.
        cw=torch.stack(((spans[:,0]+spans[:,1])/2,spans[:,1]-spans[:,0]),dim=-1)
        targets.append(dict(segments=spans,spans=cw,labels=torch.zeros(len(spans),dtype=torch.long)))
    return dict(video_feats=feats,video_mask=mask,query=torch.stack([x['query'] for x in batch]),
        query_mask=torch.stack([x['query_mask'] for x in batch]),targets=targets,
        meta=dict(video_id=[x['video_id'] for x in batch],duration=[x['duration'] for x in batch],
                  query=[x['query_str'] for x in batch],time_sec=[x['time_sec'] for x in batch]))
