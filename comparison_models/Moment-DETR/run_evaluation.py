"""Evaluate a repaired adapted DETR checkpoint; no test threshold tuning."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from torch.utils.data import DataLoader
from checkpoint_utils import read_checkpoint,restore_config,load_model_state
from moment_detr_module.dataset import MomentDETRDataset,collate_fn
from moment_detr_module.engine import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume',required=True); p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--output',required=True); p.add_argument('--annotations'); p.add_argument('--features')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); a=p.parse_args()
    if Path(a.output).exists(): raise FileExistsError(a.output)
    ckpt=read_checkpoint(a.resume,require_metadata=False)
    if ckpt.get('adapted_detr_schema')!=2: raise ValueError('Checkpoint lacks repaired adapted-DETR metadata')
    cfg=restore_config(ckpt['config'])
    if a.annotations: cfg.ann_path=a.annotations
    if a.features: cfg.feature_path=a.features
    from moment_detr_module.modeling import MomentDETR
    model=MomentDETR(cfg).to(a.device); load_model_state(model,ckpt)
    loader=DataLoader(MomentDETRDataset(cfg,a.split),batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.num_workers,collate_fn=collate_fn)
    stats=evaluate(model,loader,a.device,output_path=a.output)
    Path(a.output+'.metrics.json').write_text(json.dumps(stats,indent=2,allow_nan=False))
    print(json.dumps(stats,indent=2))


if __name__=='__main__': main()
