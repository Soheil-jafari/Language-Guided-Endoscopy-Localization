"""Convert complete frame predictions to the same segment JSONL used by adapted DETR."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from data_contract import load_video_metadata,sample_indices
from metrics import segments_from_scores


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['predictions','reference','metadata','output']: p.add_argument('--'+name,required=True)
    p.add_argument('--sample-fps',type=float,default=1.)
    p.add_argument('--threshold',type=float); p.add_argument('--calibration')
    p.add_argument('--min-duration',type=float,default=0.); p.add_argument('--merge-gap',type=float,default=0.)
    a=p.parse_args()
    if a.calibration:
        calibration=json.loads(Path(a.calibration).read_text())
        if calibration['split']!='validation': raise ValueError('Calibration must be validation-only')
        if a.threshold is not None: raise ValueError('Choose threshold OR calibration, not both')
        a.threshold=calibration['threshold']
    if a.threshold is None: raise ValueError('Supply a fixed threshold or validation calibration')
    if Path(a.output).exists(): raise FileExistsError(a.output)
    df=pd.read_csv(a.predictions); meta=load_video_metadata(a.metadata)
    if df.duplicated(['video_id','text_query','frame_idx']).any(): raise ValueError('Deduplicate overlapping predictions first')
    references=[json.loads(s) for s in Path(a.reference).read_text().splitlines() if s.strip()]
    pairs={(r['video'],r['query']) for r in references}
    if len(pairs)!=len(references) or pairs!=set(zip(df.video_id,df.text_query)): raise ValueError('Prediction/reference pair sets differ')
    records=[]
    for row in references:
        video,query=row['video'],row['query']; m=meta[video]
        part=df[(df.video_id==video)&(df.text_query==query)].sort_values('frame_idx')
        grid=sample_indices(m['frame_count'],m['source_fps'],a.sample_fps)
        if not np.array_equal(part.frame_idx.to_numpy(),grid): raise ValueError(f'{video}/{query}: incomplete or mismatched time grid')
        duration=m['frame_count']/m['source_fps']
        if not np.isclose(row['duration'],duration): raise ValueError('Duration mismatch')
        spans=segments_from_scores(grid/m['source_fps'],part.score,duration,a.threshold,a.min_duration,a.merge_gap)
        records.append(dict(video_id=video,text_query=query,duration=duration,ground_truth=row['timestamps'],predictions=spans))
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(''.join(json.dumps(r)+'\n' for r in records))


if __name__=='__main__': main()
