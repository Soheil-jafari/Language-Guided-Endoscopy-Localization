"""Export independent labels on the common frame grid without running a model."""
import argparse
from pathlib import Path
import pandas as pd
from benchmark import prepare_pairs
from data_contract import sample_indices


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['triplets','metadata','frames','annotations','output']: p.add_argument('--'+name,required=True)
    p.add_argument('--sample-fps',type=float,default=1.); a=p.parse_args()
    if Path(a.output).exists(): raise FileExistsError(a.output)
    pairs,meta,store,ann=prepare_pairs(a.triplets,a.metadata,a.frames,a.annotations,a.sample_fps)
    rows=[]
    for (video,query),spec in sorted(pairs.items()):
        m=meta[video]
        for idx in sample_indices(m['frame_count'],m['source_fps'],a.sample_fps):
            rows.append(dict(video_id=video,text_query=query,frame_idx=int(idx),label=ann.label(video,int(idx),spec)))
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(a.output,index=False)
    store.close()


if __name__=='__main__': main()
