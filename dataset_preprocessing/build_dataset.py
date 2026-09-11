"""Create explicit source metadata, sampled frames, and concept-labelled manifests.

Never guesses source FPS from JPEG counts. Existing outputs are refused.
Phase labels represent annotated intervals; sparse tool labels remain observations.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from data_contract import AnnotationIndex, FrameStore, canonical_query, load_video_metadata, sample_indices


def parse_annotations(phases,tools,output):
    """Read original Cholec80 phase/tool files without imputing missing tool rows."""
    if Path(output).exists(): raise FileExistsError(output)
    phase_files=sorted(Path(phases).glob('*-phase.txt'))
    if not phase_files: raise ValueError('No *-phase.txt annotation files')
    pieces=[]
    for path in phase_files:
        stem=path.name.removesuffix('-phase.txt'); video='CHOLEC80__'+stem
        phase=pd.read_csv(path,sep=r'\s+')
        if not {'Frame','Phase'}.issubset(phase): raise ValueError(f'Unexpected phase columns: {path}')
        phase=phase[['Frame','Phase']].rename(columns={'Frame':'frame_idx','Phase':'original_label'})
        tool_path=Path(tools)/(stem+'-tool.txt')
        if not tool_path.is_file(): raise FileNotFoundError(tool_path)
        tool=pd.read_csv(tool_path,sep=r'\s+').rename(columns={'Frame':'frame_idx'})
        if 'frame_idx' not in tool or len(tool.columns)!=8: raise ValueError(f'Expected frame and seven tool columns: {tool_path}')
        if phase.frame_idx.duplicated().any() or tool.frame_idx.duplicated().any(): raise ValueError('Duplicate raw annotation frames')
        merged=phase.merge(tool,on='frame_idx',how='outer',validate='one_to_one')
        merged['standardized_video_id']=video; pieces.append(merged)
    df=pd.concat(pieces,ignore_index=True)
    AnnotationIndex(df)  # Validate vocabulary and binary observations before writing.
    Path(output).parent.mkdir(parents=True,exist_ok=True); df.to_csv(output,index=False)


def extract(videos, frames, metadata, sample_fps):
    import cv2
    videos,frames,metadata=Path(videos),Path(frames),Path(metadata)
    if metadata.exists(): raise FileExistsError(metadata)
    sources=sorted(videos.glob('*.mp4'))
    if not sources: raise ValueError('No MP4 videos')
    result={}
    for source in sources:
        video=source.stem if source.stem.startswith('CHOLEC80__') else 'CHOLEC80__'+source.stem
        destination=frames/video
        if destination.exists() or destination.with_suffix('.zip').exists(): raise FileExistsError(destination)
        cap=cv2.VideoCapture(str(source))
        if not cap.isOpened(): raise OSError(f'Cannot open {source}')
        fps=float(cap.get(cv2.CAP_PROP_FPS)); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        grid=sample_indices(count,fps,sample_fps)
        destination.mkdir(parents=True)
        wanted=set(map(int,grid)); decoded=0
        try:
            while decoded<count:
                ok,image=cap.read()
                if not ok: raise OSError(f'Decode failed at {source}:{decoded}; incomplete output must be removed before retry')
                if decoded in wanted and not cv2.imwrite(str(destination/f'frame_{decoded:07d}.jpg'),image):
                    raise OSError('JPEG write failed')
                decoded+=1
        finally: cap.release()
        result[video]=dict(source_fps=fps,frame_count=count,sample_fps=sample_fps,source_file=str(source.resolve()))
    metadata.parent.mkdir(parents=True,exist_ok=True)
    metadata.write_text(json.dumps(result,indent=2))


def build(annotations,metadata,frames,splits,output,sample_fps):
    meta=load_video_metadata(metadata); index=AnnotationIndex(annotations); store=FrameStore(frames)
    split_map=json.loads(Path(splits).read_text())
    if set(split_map)!={'train','val','test'}: raise ValueError('Split JSON requires train, val, test video lists')
    all_ids=[v for values in split_map.values() for v in values]
    if len(all_ids)!=len(set(all_ids)): raise ValueError('Duplicate videos within/across splits')
    if any(not values for values in split_map.values()): raise ValueError('Each split must contain videos')
    output=Path(output)
    if output.exists() and any(output.iterdir()): raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    for split,videos in split_map.items():
        rows=[]
        for video in videos:
            m=meta[video]; ids=sample_indices(m['frame_count'],m['source_fps'],sample_fps)
            missing=set(map(int,ids))-set(store.names(video))
            if missing: raise FileNotFoundError(f'{video}: missing sampled frames, first {min(missing)}')
            # One canonical query per concept; all known negatives are explicit.
            for kind in ['phase','tool']:
                for concept in range(7):
                    spec=(kind,concept)
                    for frame in ids:
                        label=index.label(video,int(frame),spec)
                        if label==-100: continue
                        rows.append(dict(frame_path=str(Path(frames)/video/store.names(video)[int(frame)]),
                            text_query=canonical_query(spec),query_kind=kind,concept_id=concept,relevance_label=label))
        if not rows: raise ValueError(f'{split} has no observed labels')
        pd.DataFrame(rows).to_csv(output/f'cholec80_{split}_triplets.csv',index=False)
    (output/'split_manifest.json').write_text(json.dumps(split_map,indent=2))
    store.close()


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    raw=sub.add_parser('annotations')
    for name in ['phases','tools','output']: raw.add_argument('--'+name,required=True)
    e=sub.add_parser('extract'); e.add_argument('--videos',required=True); e.add_argument('--frames',required=True); e.add_argument('--metadata',required=True); e.add_argument('--sample-fps',type=float,default=1.)
    b=sub.add_parser('triplets')
    for name in ['annotations','metadata','frames','splits','output']: b.add_argument('--'+name,required=True)
    b.add_argument('--sample-fps',type=float,default=1.)
    args=vars(p.parse_args()); command=args.pop('command')
    (extract if command=='extract' else build if command=='triplets' else parse_annotations)(**args)


if __name__=='__main__': main()
