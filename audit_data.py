"""CPU preflight of labels, sampled frames, metadata and video-level splits."""
import argparse
import json
from pathlib import Path
import pandas as pd
from data_contract import (AnnotationIndex,FrameStore,load_video_metadata,path_identity,
                           sample_indices,validate_triplets)
from checkpoint_utils import file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['train','val','test','metadata','annotations','frames','output']: p.add_argument('--'+name,required=True)
    p.add_argument('--sample-fps',type=float,default=1.); p.add_argument('--decode-all',action='store_true')
    a=p.parse_args()
    if Path(a.output).exists(): raise FileExistsError(a.output)
    meta=load_video_metadata(a.metadata); ann=AnnotationIndex(a.annotations); store=FrameStore(a.frames)
    summary={}; sets={}
    for split in ['train','val','test']:
        path=getattr(a,split); df=pd.read_csv(path); specs=validate_triplets(df); pairs={}; videos=set()
        for row,spec in zip(df.to_dict('records'),specs):
            video,idx=path_identity(row['frame_path']); videos.add(video); pairs[(video,row['text_query'])]=spec
            if not 0<=idx<meta[video]['frame_count']: raise ValueError('Anchor outside source video')
            if 'relevance_label' in row and int(row['relevance_label'])!=ann.label(video,idx,spec):
                raise ValueError(f'Contradictory annotation at {video}/{idx}/{row["text_query"]}')
        sets[split]=videos
        for video in videos:
            m=meta[video]; grid=sample_indices(m['frame_count'],m['source_fps'],a.sample_fps)
            if ann.maximum_frame.get(video,-1)>=m['frame_count']: raise ValueError(f'Annotation/source duration mismatch: {video}')
            missing=set(map(int,grid))-set(store.names(video))
            if missing: raise FileNotFoundError(f'{video}: {len(missing)} missing sampled frames')
            for idx in (grid if a.decode_all else grid[[0,len(grid)-1]]): store.read(video,int(idx))
        counts={-100:0,0:0,1:0}
        for (video,query),spec in pairs.items():
            m=meta[video]
            for idx in sample_indices(m['frame_count'],m['source_fps'],a.sample_fps): counts[ann.label(video,int(idx),spec)]+=1
        summary[split]=dict(videos=sorted(videos),rows=len(df),unique_video_queries=len(pairs),sampled_labels=counts,triplets_sha256=file_hash(path))
    if sets['train']&sets['val'] or sets['train']&sets['test'] or sets['val']&sets['test']: raise ValueError('Video leakage across splits')
    summary.update(metadata_sha256=file_hash(a.metadata),annotations_sha256=file_hash(a.annotations),sample_fps=a.sample_fps,decoded_all_images=a.decode_all)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(summary,indent=2))
    store.close(); print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
