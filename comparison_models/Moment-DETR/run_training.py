"""Train the repository's ResNet50/RoBERTa adapted DETR (not official Moment-DETR)."""
import argparse
import json
import random
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from checkpoint_utils import atomic_save,file_hash,rng_state,restore_rng,load_model_state,read_checkpoint
from moment_detr_module.configs import Config
from moment_detr_module.dataset import MomentDETRDataset,collate_fn
from moment_detr_module.engine import train_one_epoch,evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,help='JSON configuration overrides (including ann_path and feature_path)')
    p.add_argument('--output',required=True); p.add_argument('--resume-from')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); a=p.parse_args()
    cfg=Config()
    for k,v in json.loads(Path(a.config).read_text()).items():
        if not hasattr(cfg,k): raise ValueError(f'Unknown config key: {k}')
        setattr(cfg,k,v)
    out=Path(a.output)
    if out.exists() and any(out.iterdir()) and not a.resume_from: raise FileExistsError(out)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    train=MomentDETRDataset(cfg,'train'); val=MomentDETRDataset(cfg,'val')
    if {x['video'] for x in train.annotations}&{x['video'] for x in val.annotations}: raise ValueError('Train/validation video leakage')
    provenance={split:file_hash(Path(cfg.ann_path)/(split+'.jsonl')) for split in ['train','val']}
    # Features are inputs too; replacing them must invalidate exact resume.
    provenance['features']={v:file_hash(Path(cfg.feature_path)/(v+'.npz')) for v in sorted({x['video'] for x in train.annotations+val.annotations})}
    from moment_detr_module.modeling import MomentDETR
    model=MomentDETR(cfg).to(a.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=cfg.lr_drop)
    start=0; best=-1.
    if a.resume_from:
        ckpt=read_checkpoint(a.resume_from,require_metadata=False)
        if ckpt.get('adapted_detr_schema')!=2 or ckpt.get('config')!=vars(cfg) or ckpt.get('provenance')!=provenance:
            raise ValueError('Incompatible adapted DETR resume')
        load_model_state(model,ckpt); optimizer.load_state_dict(ckpt['optimizer']); scheduler.load_state_dict(ckpt['scheduler'])
        restore_rng(ckpt['rng']); start=ckpt['epoch']; best=ckpt['best']
    kw=dict(batch_size=cfg.batch_size,num_workers=cfg.num_workers,collate_fn=collate_fn)
    train_loader=DataLoader(train,shuffle=True,**kw); val_loader=DataLoader(val,shuffle=False,**kw)
    out.mkdir(parents=True,exist_ok=True)
    for epoch in range(start,cfg.epochs):
        loss=train_one_epoch(model,train_loader,optimizer,a.device,epoch,cfg.clip_max_norm)
        scheduler.step(); stats=evaluate(model,val_loader,a.device)
        score=stats['pooled_AP@0.5']
        if score is None: raise ValueError('Validation needs positive segments for model selection')
        improved=score>best; best=max(best,score)
        ckpt=dict(adapted_detr_schema=2,model_state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            epoch=epoch+1,best=best,config=vars(cfg),provenance=provenance,rng=rng_state())
        if improved: atomic_save(ckpt,out/'best.ckpt')
        atomic_save(ckpt,out/'latest.ckpt')
        (out/f'metrics_{epoch+1}.json').write_text(json.dumps(dict(epoch=epoch+1,train_loss=loss,**stats),indent=2))
        print(f'Epoch {epoch+1}: {stats}')


if __name__=='__main__': main()
