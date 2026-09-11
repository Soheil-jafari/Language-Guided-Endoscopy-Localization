"""Adapted X-CLIP contrastive fine-tuning with multi-positive concept targets.

Only clips with an observed positive target enter contrastive training. Repeated
queries/paraphrases of one concept are positives, not negatives of one another.
This is not the official X-CLIP action-recognition training protocol.
"""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from benchmark import prepare_pairs
from data_contract import sample_indices,window_positions
from checkpoint_utils import atomic_save,file_hash,rng_state,restore_rng,load_model_state,read_checkpoint


def multi_positive_loss(logits,concepts,positive_mask=None,observed_mask=None):
    if logits.ndim!=2 or logits.shape[0]!=logits.shape[1] or len(concepts)!=len(logits):
        raise ValueError('Expected square paired-video/text logits')
    mask=torch.as_tensor(positive_mask if positive_mask is not None else [[a==b for b in concepts] for a in concepts],device=logits.device,dtype=torch.bool)
    observed=torch.ones_like(mask) if observed_mask is None else torch.as_tensor(observed_mask,device=logits.device,dtype=torch.bool)
    if not mask.any(0).all() or not mask.any(1).all() or (mask&~observed).any(): raise ValueError('Each video/text needs an observed positive')
    scores=logits.float().masked_fill(~observed,-1e9)
    weights=mask.float()/mask.sum(-1,keepdim=True)
    reverse=mask.T.float()/mask.T.sum(-1,keepdim=True)
    return -.5*((weights*F.log_softmax(scores,dim=-1)).sum(-1).mean()+
                 (reverse*F.log_softmax(scores.T,dim=-1)).sum(-1).mean())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['train-triplets','val-triplets','metadata','frames','annotations','output']: p.add_argument('--'+name,required=True)
    p.add_argument('--model-name',default='microsoft/xclip-base-patch32'); p.add_argument('--sample-fps',type=float,default=1.)
    p.add_argument('--epochs',type=int,default=5); p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--lr',type=float,default=1e-5); p.add_argument('--seed',type=int,default=42); p.add_argument('--resume-from')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args(); out=Path(a.output)
    if a.epochs<1 or a.batch_size<2: raise ValueError('Need epochs>=1 and batch-size>=2 for contrastive learning')
    if out.exists() and any(out.iterdir()) and not a.resume_from: raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    from transformers import XCLIPModel,XCLIPProcessor
    model=XCLIPModel.from_pretrained(a.model_name).to(a.device)
    processor=XCLIPProcessor.from_pretrained(a.model_name); length=model.config.vision_config.num_frames
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    bundles=[prepare_pairs(csv,a.metadata,a.frames,a.annotations,a.sample_fps) for csv in [a.train_triplets,a.val_triplets]]
    if {v for v,q in bundles[0][0]} & {v for v,q in bundles[1][0]}: raise ValueError('Train/validation video leakage')
    datasets=[]
    for pairs,meta,store,ann in bundles:
        records=[]
        for (video,query),spec in sorted(pairs.items()):
            m=meta[video]; grid=sample_indices(m['frame_count'],m['source_fps'],a.sample_fps)
            for start in window_positions(len(grid),length,length):
                ids=grid[start:start+length].tolist()
                if any(ann.label(video,i,spec)==1 for i in ids):
                    ids+=[ids[-1]]*(length-len(ids)); records.append((video,query,spec,ids))
        if len(records)<2 or len({r[2] for r in records})<2: raise ValueError('Need at least two concepts for contrastive learning')
        datasets.append(records)
    signature={k:v for k,v in vars(a).items() if k not in ['resume_from']}
    provenance={k:file_hash(getattr(a,k)) for k in ['train_triplets','val_triplets','metadata','annotations']}
    start=0; best=float('inf')
    if a.resume_from:
        ckpt=read_checkpoint(a.resume_from,require_metadata=False)
        if ckpt.get('baseline_schema')!=2 or ckpt.get('arguments')!=signature or ckpt.get('provenance')!=provenance: raise ValueError('Incompatible X-CLIP resume')
        load_model_state(model,ckpt); optimizer.load_state_dict(ckpt['optimizer']); restore_rng(ckpt['rng'])
        start=ckpt['epoch']; best=ckpt['best']
    for epoch in range(start,a.epochs):
        losses=[]
        for split,records in enumerate(datasets):
            model.train(split==0); order=list(range(len(records)))
            if split==0: random.shuffle(order)
            else: random.Random(a.seed).shuffle(order)
            total=0.; count=0
            chunks=[order[i:i+a.batch_size] for i in range(0,len(order),a.batch_size)]
            if len(chunks)>1 and len(chunks[-1])==1:
                tail=chunks.pop(); chunks[-1].extend(tail)
            for chunk in chunks:
                batch=[records[i] for i in chunk]
                clips=[[np.asarray(bundles[split][2].read(v,i)) for i in ids] for v,q,c,ids in batch]
                inputs=processor(videos=clips,text=[r[1] for r in batch],return_tensors='pt',padding=True)
                with torch.set_grad_enabled(split==0):
                    logits=model(**{k:v.to(a.device) for k,v in inputs.items()}).logits_per_video
                    positives=[]; observed=[]
                    for video,query,spec,ids in batch:
                        values=[[bundles[split][3].label(video,i,candidate[2]) for i in ids] for candidate in batch]
                        positives.append([1 in v for v in values])
                        observed.append([1 in v or all(x==0 for x in v) for v in values])
                    loss=multi_positive_loss(logits,[r[2] for r in batch],positives,observed)
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite X-CLIP loss')
                    if split==0: optimizer.zero_grad(); loss.backward(); optimizer.step()
                total+=float(loss.detach())*len(batch); count+=len(batch)
            if not count: raise ValueError('No contrastive batches')
            losses.append(total/count)
        improved=losses[1]<best
        best=min(best,losses[1])
        ckpt=dict(baseline_schema=2,model_state_dict=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,best=best,
            model_name=a.model_name,sample_fps=a.sample_fps,arguments=signature,provenance=provenance,rng=rng_state())
        if improved: atomic_save(ckpt,out/'best.pt')
        atomic_save(ckpt,out/'latest.pt')
        print(f'Epoch {epoch+1}: train={losses[0]:.5f} validation={losses[1]:.5f}')
    for bundle in bundles: bundle[2].close()


if __name__=='__main__': main()
