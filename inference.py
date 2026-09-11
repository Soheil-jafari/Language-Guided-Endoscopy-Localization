"""Strict checkpoint inference on the training time grid, with explicit artifacts."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from checkpoint_utils import read_checkpoint,restore_config,load_model_state,file_hash
from data_contract import sample_indices,window_positions
from dataset import image_transform
from metrics import segments_from_scores


def read_sampled_video(path,sample_fps,image_size=224):
    import cv2
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise OSError(f'Cannot open video: {path}')
    fps=float(cap.get(cv2.CAP_PROP_FPS)); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    grid=sample_indices(count,fps,sample_fps)
    wanted=set(map(int,grid)); frames=[]
    try:
        for idx in range(count):
            ok,frame=cap.read()
            if not ok: raise OSError(f'Video decode failed at source frame {idx}; timestamps are not shifted to hide it')
            if idx in wanted:
                # Keep resized images rather than full-resolution hour-long videos.
                frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                from torchvision.transforms.v2 import functional as VF
                frames.append(VF.resize(torch.from_numpy(frame.copy()).permute(2,0,1),[image_size,image_size],antialias=True))
    finally: cap.release()
    return frames,grid,fps,count/fps


def load_model(checkpoint,device):
    from models import LocalizationFramework
    ckpt=read_checkpoint(checkpoint)
    cfg=restore_config(ckpt['config'])
    cfg.TRAIN.DEVICE=str(device)
    model=LocalizationFramework(cfg,initialize_backbone=False).to(device)
    load_model_state(model,ckpt)
    model.eval()
    return model,ckpt


@torch.no_grad()
def score_frames(model,frames,query,device,raw_patch_maps=False):
    cfg=model.config; length=cfg.DATA.CLIP_LENGTH
    transform=image_transform(cfg.DATA.TRAIN_CROP_SIZE)
    text=model.text_encoder.tokenizer(query,padding='max_length',truncation=True,max_length=cfg.DATA.MAX_TEXT_LENGTH,return_tensors='pt')
    ids=text['input_ids'].to(device); mask=text['attention_mask'].to(device)
    features,_=model.text_encoder(ids,mask)
    sums=np.zeros(len(frames)); counts=np.zeros(len(frames)); maps={}
    model.language_guided_head.return_xai_map=raw_patch_maps
    # Same stride as full-video validation, including the overlapping tail window.
    for start in window_positions(len(frames),length,length):
        chunk=frames[start:start+length]; n=len(chunk)
        chunk=chunk+[chunk[-1]]*(length-n)
        clip=transform(torch.stack(chunk)).permute(1,0,2,3).unsqueeze(0).to(device)
        out=model(clip,ids,mask,text_features=features)
        p=(out[0] if cfg.MODEL.USE_UNCERTAINTY else out[0].sigmoid())[0,:,0].float().cpu().numpy()
        sums[start:start+n]+=p[:n]; counts[start:start+n]+=1
        if raw_patch_maps:
            for j,value in enumerate(out[2][0,:n].float().cpu().numpy()):
                maps.setdefault(start+j,[]).append(value)
    if np.any(counts==0): raise RuntimeError('Uncovered frames')
    return sums/counts, np.stack([np.mean(maps[i],axis=0) for i in range(len(frames))]) if maps else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True); p.add_argument('--video',required=True)
    p.add_argument('--query',required=True); p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--threshold',type=float,help='Fixed threshold; defaults to saved validation threshold')
    p.add_argument('--raw-patch-maps',action='store_true',help='Raw relevance-head decomposition, NOT attention or attribution of the final temporal score')
    args=p.parse_args(); output=Path(args.output)
    if output.exists() and any(output.iterdir()): raise FileExistsError(output)
    model,ckpt=load_model(args.checkpoint,args.device)
    frames,indices,fps,duration=read_sampled_video(args.video,model.config.DATA.SAMPLE_FPS,model.config.DATA.TRAIN_CROP_SIZE)
    scores,maps=score_frames(model,frames,args.query,args.device,args.raw_patch_maps)
    threshold=args.threshold if args.threshold is not None else ckpt['validation_threshold']
    spans=segments_from_scores(indices/fps,scores,duration,threshold,model.config.MIN_SEGMENT_DURATION,model.config.MERGE_GAP)
    output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(dict(frame_idx=indices,time_sec=indices/fps,score=scores)).to_csv(output/'frame_scores.csv',index=False)
    (output/'segments.json').write_text(json.dumps(spans,indent=2))
    (output/'manifest.json').write_text(json.dumps(dict(query=args.query,video=str(Path(args.video).resolve()),source_fps=fps,
        duration=duration,sample_fps=model.config.DATA.SAMPLE_FPS,threshold=threshold,checkpoint_sha256=file_hash(args.checkpoint),
        map_description='Raw relevance-head patch logits; do not explain the final temporal/evidential output'),indent=2))
    if maps is not None: np.savez_compressed(output/'raw_relevance_patch_maps.npz',frame_idx=indices,maps=maps)


if __name__=='__main__': main()
