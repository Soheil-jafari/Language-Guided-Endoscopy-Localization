"""Make one complete sampled-grid segment target per video/query, in seconds."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from benchmark import prepare_pairs
from data_contract import sample_indices
from metrics import segments_from_scores


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['triplets','metadata','frames','annotations','output']: p.add_argument('--'+name,required=True)
    p.add_argument('--sample-fps',type=float,default=1.)
    a=p.parse_args()
    if Path(a.output).exists(): raise FileExistsError(a.output)
    pairs,meta,store,ann=prepare_pairs(a.triplets,a.metadata,a.frames,a.annotations,a.sample_fps)
    records=[]
    for (video,query),spec in sorted(pairs.items()):
        m=meta[video]; grid=sample_indices(m['frame_count'],m['source_fps'],a.sample_fps)
        labels=[ann.label(video,int(i),spec) for i in grid]
        if -100 in labels:
            raise ValueError(f'{video}/{query}: unobserved grid points cannot establish complete segment/absence targets. Use a compatible annotated grid or obtain complete labels.')
        duration=m['frame_count']/m['source_fps']
        spans=segments_from_scores(grid/m['source_fps'],labels,duration)
        records.append(dict(video=video,query=query,duration=duration,timestamps=[s[:2] for s in spans],
            query_kind=spec[0],concept_id=spec[1],sample_fps=a.sample_fps,target_resolution='sample-grid left-closed cells'))
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(''.join(json.dumps(r)+'\n' for r in records))
    store.close()


if __name__=='__main__': main()
