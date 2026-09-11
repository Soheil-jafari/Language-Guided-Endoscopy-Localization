"""CLIP/X-CLIP inference on explicit video-query manifests and source-frame grids.

Outputs are adapted baseline scores, not calibrated event probabilities. CLIP uses
rescaled cosine similarity; X-CLIP assigns a clip score to its sampled frames.
Thresholds must be selected separately on validation by evaluate.py.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from data_contract import (AnnotationIndex,FrameStore,load_video_metadata,path_identity,
                           sample_indices,validate_triplets,window_positions)
from checkpoint_utils import file_hash,load_model_state,read_checkpoint


def prepare_pairs(triplets,metadata,frames,annotations,sample_fps):
    df=pd.read_csv(triplets); specs=validate_triplets(df)
    pairs={}
    for row,spec in zip(df.to_dict('records'),specs):
        video,_=path_identity(row['frame_path']); pairs[(video,row['text_query'])]=spec
    meta=load_video_metadata(metadata); store=FrameStore(frames); ann=AnnotationIndex(annotations)
    for video,_ in pairs:
        m=meta[video]; grid=sample_indices(m['frame_count'],m['source_fps'],sample_fps)
        if ann.maximum_frame.get(video,-1)>=m['frame_count']: raise ValueError(f'Annotation/source duration mismatch: {video}')
        if set(map(int,grid))-set(store.names(video)): raise FileNotFoundError(f'Missing sampled frames: {video}')
    return pairs,meta,store,ann


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['clip','xclip'],required=True)
    p.add_argument('--model-name',required=True,help='Exact Hugging Face vision-language checkpoint; uses matching processor')
    p.add_argument('--checkpoint',help='Optional repaired X-CLIP fine-tune checkpoint')
    for name in ['triplets','metadata','frames','annotations','output']: p.add_argument('--'+name,required=True)
    p.add_argument('--sample-fps',type=float,default=1.); p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args()
    if Path(a.output).exists(): raise FileExistsError(a.output)
    if a.batch_size<1: raise ValueError('Batch size must be positive')
    from transformers import CLIPModel,CLIPProcessor,XCLIPModel,XCLIPProcessor
    name=a.model_name
    model=(CLIPModel if a.model=='clip' else XCLIPModel).from_pretrained(name).to(a.device).eval()
    processor=(CLIPProcessor if a.model=='clip' else XCLIPProcessor).from_pretrained(name)
    if a.checkpoint:
        if a.model!='xclip': raise ValueError('Only repaired X-CLIP fine-tune checkpoints are supported here')
        ckpt=read_checkpoint(a.checkpoint,require_metadata=False)
        if ckpt.get('baseline_schema')!=2 or ckpt.get('model_name')!=name or ckpt.get('sample_fps')!=a.sample_fps:
            raise ValueError('X-CLIP checkpoint name/time-grid/schema mismatch')
        load_model_state(model,ckpt)
    pairs,meta,store,ann=prepare_pairs(a.triplets,a.metadata,a.frames,a.annotations,a.sample_fps)
    rows=[]
    with torch.no_grad():
        for (video,query),spec in sorted(pairs.items()):
            m=meta[video]; grid=sample_indices(m['frame_count'],m['source_fps'],a.sample_fps)
            sums=np.zeros(len(grid)); counts=np.zeros(len(grid))
            if a.model=='clip':
                for start in range(0,len(grid),a.batch_size):
                    chunk=grid[start:start+a.batch_size]
                    inputs=processor(images=[store.read(video,int(i)) for i in chunk],text=[query],return_tensors='pt',padding=True)
                    logits=model(**{k:v.to(a.device) for k,v in inputs.items()}).logits_per_image[:,0]
                    # Undo learned temperature before mapping cosine to [0,1];
                    # sigmoid(logit) can saturate and destroy ranking precision.
                    sums[start:start+len(chunk)]=((logits.float()/model.logit_scale.exp()+1)/2).clamp(0,1).cpu().numpy(); counts[start:start+len(chunk)]=1
            else:
                length=model.config.vision_config.num_frames
                for start in window_positions(len(grid),length,length):
                    ids=grid[start:start+length].tolist(); n=len(ids); ids+=[ids[-1]]*(length-n)
                    inputs=processor(videos=[[np.asarray(store.read(video,i)) for i in ids]],text=[query],padding=True,return_tensors='pt')
                    # HF X-CLIP text embeddings are VIDEO-CONDITIONED, not pooled tokens.
                    logit=model(**{k:v.to(a.device) for k,v in inputs.items()}).logits_per_video[0,0]
                    sums[start:start+n]+=float(((logit.float()/model.logit_scale.exp()+1)/2).clamp(0,1)); counts[start:start+n]+=1
            for idx,score in zip(grid,sums/counts):
                rows.append(dict(video_id=video,text_query=query,frame_idx=int(idx),time_sec=float(idx/m['source_fps']),
                    score=float(score),label=ann.label(video,int(idx),spec)))
    store.close(); Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(a.output,index=False)
    Path(a.output+'.manifest.json').write_text(json.dumps(dict(model=name,baseline=a.model,sample_fps=a.sample_fps,
        triplets_sha256=file_hash(a.triplets),metadata_sha256=file_hash(a.metadata),checkpoint_sha256=file_hash(a.checkpoint) if a.checkpoint else None,
        score_semantics='Uncalibrated (cosine+1)/2; fit operating threshold on validation only'),indent=2))


if __name__=='__main__': main()
