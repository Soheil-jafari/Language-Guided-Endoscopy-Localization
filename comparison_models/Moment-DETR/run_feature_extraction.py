"""ResNet50 features for the adapted DETR, preserving actual source timestamps."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from data_contract import FrameStore,load_video_metadata,sample_indices


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['frames','metadata','output']: p.add_argument('--'+name,required=True)
    p.add_argument('--sample-fps',type=float,default=1.); p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args()
    from torchvision.models import resnet50,ResNet50_Weights
    weights=ResNet50_Weights.IMAGENET1K_V1
    backbone=resnet50(weights=weights).to(a.device).eval(); backbone.fc=torch.nn.Identity()
    transform=weights.transforms(); meta=load_video_metadata(a.metadata); store=FrameStore(a.frames)
    output=Path(a.output); output.mkdir(parents=True,exist_ok=True)
    for video,m in sorted(meta.items()):
        path=output/(video+'.npz')
        if path.exists(): raise FileExistsError(path)
        ids=sample_indices(m['frame_count'],m['source_fps'],a.sample_fps); pieces=[]
        with torch.no_grad():
            for start in range(0,len(ids),a.batch_size):
                images=torch.stack([transform(store.read(video,int(i))) for i in ids[start:start+a.batch_size]])
                pieces.append(backbone(images.to(a.device)).cpu().numpy())
        np.savez_compressed(path,features=np.concatenate(pieces),frame_idx=ids,time_sec=ids/m['source_fps'],
            duration=m['frame_count']/m['source_fps'],source_fps=m['source_fps'],sample_fps=a.sample_fps)
    store.close()


if __name__=='__main__': main()
