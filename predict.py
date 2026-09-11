"""Export proposed-model predictions for every video/query in a split manifest."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from inference import load_model
from dataset import EndoscopyLocalizationDataset
from train import predict_loader
from checkpoint_utils import file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint','triplets','metadata','frames','annotations','output']: p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch-size',type=int,default=1); p.add_argument('--workers',type=int,default=0)
    a=p.parse_args()
    if Path(a.output).exists(): raise FileExistsError(a.output)
    model,ckpt=load_model(a.checkpoint,a.device)
    cfg=model.config; cfg.VIDEO_METADATA_PATH=a.metadata; cfg.EXTRACTED_FRAMES_DIR=a.frames; cfg.CHOLEC80_PARSED_ANNOTATIONS=a.annotations
    ds=EndoscopyLocalizationDataset(a.triplets,model.text_encoder.tokenizer,cfg.DATA.CLIP_LENGTH,False,cfg)
    df=predict_loader(model,DataLoader(ds,batch_size=a.batch_size,num_workers=a.workers),a.device)
    df['time_sec']=[row.frame_idx/ds.metadata[row.video_id]['source_fps'] for row in df.itertuples()]
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); df.to_csv(a.output,index=False)
    Path(a.output+'.manifest.json').write_text(json.dumps(dict(checkpoint_sha256=file_hash(a.checkpoint),triplets_sha256=file_hash(a.triplets),
        metadata_sha256=file_hash(a.metadata),validation_threshold=ckpt['validation_threshold'],sample_fps=cfg.DATA.SAMPLE_FPS),indent=2))


if __name__=='__main__': main()
