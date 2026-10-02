"""Training with checked data contracts and epoch-boundary resume.

Single-device training is deliberate: data parallelism must not change query pairing.
Use a new output directory for a new experiment; old metrics are not repaired by
reloading old weights. See REPAIR_NOTES.md for migration and validation limits.
"""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from project_config import config
from checkpoint_utils import (SCHEMA_VERSION, atomic_save, config_dict, file_hash,
    load_model_state, read_checkpoint, rng_state, restore_rng)
from dataset import create_dataloaders
from losses import MasterLoss, EvidentialLoss
from metrics import binary_metrics, deduplicate_frame_rows, validation_threshold


def train_one_epoch(model,dataloader,optimizer,criterion,scheduler,device,scaler,
                    amp_dtype=torch.float16,profile_io=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulation=criterion.config.TRAIN.GRADIENT_ACCUMULATION_STEPS
    if accumulation<1: raise ValueError('Accumulation must be positive')
    weighted_loss=0.; total_weight=0; group_weight=0
    for step,batch in enumerate(dataloader):
        video=batch['video_clip'].to(device)
        labels=batch['labels'].to(device)
        weight=int((labels!=-100).sum())
        if weight:
            with torch.autocast(device_type=torch.device(device).type,dtype=amp_dtype,enabled=torch.device(device).type=='cuda'):
                outputs=model(video,batch['input_ids'].to(device),batch['attention_mask'].to(device))
                loss,_,_=criterion(outputs,video,labels)
            if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss at batch {step}')
            # Each objective is averaged within a batch; average batches by their
            # observed frame count. Divide gradients once by the ACTUAL group count.
            scaler.scale(loss*weight).backward()
            weighted_loss+=float(loss.detach())*weight; total_weight+=weight; group_weight+=weight
        if (step+1)%accumulation==0 or step+1==len(dataloader):
            if group_weight:
                scaler.unscale_(optimizer)
                for p in model.parameters():
                    if p.grad is not None: p.grad.div_(group_weight)
                old_scale=scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                if scaler.get_scale()>=old_scale and scheduler is not None:
                    scheduler.step()
            optimizer.zero_grad(set_to_none=True); group_weight=0
    if not total_weight: raise ValueError('Training epoch has no observed targets')
    return weighted_loss/total_weight


@torch.no_grad()
def predict_loader(model,dataloader,device):
    model.eval(); rows=[]
    uncertainty=model.config.MODEL.USE_UNCERTAINTY
    for batch in dataloader:
        outputs=model(batch['video_clip'].to(device),batch['input_ids'].to(device),batch['attention_mask'].to(device))
        output=outputs[0].squeeze(-1)
        probabilities=(output if uncertainty else output.sigmoid()).float().cpu().numpy()
        evidence_uncertainty=(2/(outputs[-1].float().sum(-1)+2)).cpu().numpy() if uncertainty else None
        for b in range(len(probabilities)):
            for t,p in enumerate(probabilities[b]):
                if not bool(batch['valid_frames'][b,t]): continue
                rows.append(dict(video_id=batch['video_id'][b],text_query=batch['text_query'][b],
                    frame_idx=int(batch['frame_indices'][b,t]),score=float(p),label=float(batch['labels'][b,t])))
                if evidence_uncertainty is not None:
                    rows[-1]['evidence_uncertainty']=float(evidence_uncertainty[b,t])
    return deduplicate_frame_rows(rows)


@torch.no_grad()
def validate_one_epoch(model,dataloader,criterion,device,**kwargs):
    # Select checkpoints by deduplicated frame NLL, not a changing annealed objective
    # or repeated tail-window counts. Full validation videos are always traversed.
    df=predict_loader(model,dataloader,device)
    observed=df[df.label!=-100]
    if observed.empty: raise ValueError('Validation has no observed labels')
    y=observed.label.to_numpy(); p=observed.score.to_numpy()
    safe=np.clip(p,1e-7,1-1e-7)
    nll=float(-(y*np.log(safe)+(1-y)*np.log1p(-safe)).mean())
    threshold=validation_threshold(y,p) if len(np.unique(y))==2 else .5
    default=binary_metrics(y,p); chosen=binary_metrics(y,p,threshold)
    return nll,default['auroc'],default['average_precision'],default['accuracy'],chosen['f1'],threshold,chosen['accuracy']


def save_checkpoint(model,optimizer,scheduler,scaler,epoch,best_val_loss,path,extra=None):
    value=dict(schema_version=SCHEMA_VERSION,model_state_dict=model.state_dict(),
        config=config_dict(model.config),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
        scaler=scaler.state_dict(),epoch=epoch,best_val_loss=best_val_loss,rng=rng_state())
    value.update(extra or {})
    atomic_save(value,path)


def load_checkpoint(path,model,optimizer=None,scheduler=None,scaler=None,map_location=None,strict=True,expected_provenance=None):
    if not strict: raise ValueError('Resume always requires strict weights')
    ckpt=read_checkpoint(path)
    if ckpt['config']!=config_dict(model.config):
        raise ValueError('Resume configuration changed; use a new weights-only fine-tune instead')
    if expected_provenance is not None and ckpt.get('provenance')!=expected_provenance:
        raise ValueError('Resume data manifests changed')
    load_model_state(model,ckpt)
    for name,obj in [('optimizer',optimizer),('scheduler',scheduler),('scaler',scaler)]:
        if obj is not None: obj.load_state_dict(ckpt[name])
    restore_rng(ckpt['rng'])
    return ckpt


def main(args):
    from models import LocalizationFramework
    if args.config:
        # Overrides only existing keys; typos cannot silently start a different run.
        def update(obj,values):
            for k,v in values.items():
                if not hasattr(obj,k): raise ValueError(f'Unknown configuration field: {k}')
                if isinstance(v,dict): update(getattr(obj,k),v)
                else: setattr(obj,k,v)
        update(config,json.loads(Path(args.config).read_text()))
    if args.subset is not None: config.TRAIN.SUBSET_RATIO=args.subset
    if config.DATA.NUM_FRAMES != config.DATA.CLIP_LENGTH: raise ValueError('NUM_FRAMES and CLIP_LENGTH must agree')
    if config.TRAIN.NUM_EPOCHS<1: raise ValueError('NUM_EPOCHS must be positive')
    random.seed(config.TRAIN.SEED); np.random.seed(config.TRAIN.SEED); torch.manual_seed(config.TRAIN.SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(config.TRAIN.SEED)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    output=Path(config.CHECKPOINT_DIR)
    if output.exists() and any(output.iterdir()) and not args.resume_from:
        raise FileExistsError('Use a new CHECKPOINT_DIR for a new run; existing experiments are preserved')
    output.mkdir(parents=True,exist_ok=True)
    provenance={name:file_hash(getattr(config,name)) for name in ['TRAIN_TRIPLETS_CSV_PATH','VAL_TRIPLETS_CSV_PATH','CHOLEC80_PARSED_ANNOTATIONS','VIDEO_METADATA_PATH']}
    model=LocalizationFramework(config,initialize_backbone=not(args.resume_from or args.finetune_from)).to(config.TRAIN.DEVICE)
    if args.finetune_from:
        # Same architecture only: partial layer surgery needs an explicit migration.
        load_model_state(model,read_checkpoint(args.finetune_from,require_metadata=False))
    train_loader,val_loader=create_dataloaders(config.TRAIN_TRIPLETS_CSV_PATH,config.VAL_TRIPLETS_CSV_PATH,
        model.text_encoder.tokenizer,config.DATA.CLIP_LENGTH,config.TRAIN.SUBSET_RATIO,settings=config)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config.TRAIN.LEARNING_RATE,weight_decay=config.TRAIN.WEIGHT_DECAY)
    steps=math.ceil(len(train_loader)/config.TRAIN.GRADIENT_ACCUMULATION_STEPS)
    total=steps*config.TRAIN.NUM_EPOCHS; warm=min(steps*config.TRAIN.WARMUP_EPOCHS,total-1)
    def schedule(step):
        if step<warm: return step/max(1,warm)
        return .5*(1+math.cos(math.pi*min(1.,(step-warm)/max(1,total-warm))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    dtype=torch.bfloat16 if config.TRAIN.AMP_DTYPE=='bf16' else torch.float16
    scaler=torch.amp.GradScaler('cuda',enabled=config.TRAIN.DEVICE.startswith('cuda') and dtype==torch.float16)
    criterion=MasterLoss(config).to(config.TRAIN.DEVICE)
    start=0; best=float('inf')
    if args.resume_from:
        ckpt=load_checkpoint(args.resume_from,model,optimizer,scheduler,scaler,expected_provenance=provenance)
        start=ckpt['epoch']; best=ckpt['best_val_loss']
    (output/'run_config.json').write_text(json.dumps(config_dict(config),indent=2))
    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    parameter_counts={name:dict(total=sum(p.numel() for p in module.parameters()),trainable=sum(p.numel() for p in module.parameters() if p.requires_grad))
                      for name,module in model.named_children()}
    (output/'parameter_counts.json').write_text(json.dumps(parameter_counts,indent=2))
    for epoch in range(start,config.TRAIN.NUM_EPOCHS):
        criterion.set_epoch(epoch)
        started=time.time()
        loss=train_one_epoch(model,train_loader,optimizer,criterion,scheduler,config.TRAIN.DEVICE,scaler,dtype)
        trained=time.time()
        val,auc,ap,acc,f1,threshold,chosen=validate_one_epoch(model,val_loader,criterion,config.TRAIN.DEVICE)
        validated=time.time()
        improved=val<best
        if improved: best=val
        extra=dict(provenance=provenance,validation_threshold=threshold)
        if improved: save_checkpoint(model,optimizer,scheduler,scaler,epoch+1,best,output/'best_model.pth',extra)
        save_checkpoint(model,optimizer,scheduler,scaler,epoch+1,best,output/'latest_model.pth',extra)
        row=dict(epoch=epoch+1,train_loss=loss,val_frame_nll=val,auroc=auc,average_precision=ap,accuracy=acc,validation_f1=f1,validation_threshold=threshold,
                 train_seconds=round(trained-started,1),val_seconds=round(validated-trained,1))  # wall time, for planning job lengths
        with (output/'training_metrics.jsonl').open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',help='JSON overrides to project_config.py')
    parser.add_argument('--subset',type=float)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--resume_from',help='Exact epoch-boundary resume of a schema-v2 run')
    group.add_argument('--finetune_from',help='Strict, same-architecture weights-only initialization')
    main(parser.parse_args())
