"""Interchange export only; repaired X-CLIP training takes frame triplets directly."""
import argparse
import json
from pathlib import Path
import math
import pandas as pd


def convert_jsonl_to_csv(jsonl_path,csv_path):
    if Path(csv_path).exists(): raise FileExistsError(csv_path)
    rows=[]; seen=set()
    for line in Path(jsonl_path).read_text().splitlines():
        if not line.strip(): continue
        data=json.loads(line); video=data['video']; query=data['query']; duration=float(data['duration'])
        key=(video,query)
        if key in seen: raise ValueError('Duplicate video/query records; group spans first')
        seen.add(key)
        if not math.isfinite(duration) or duration<=0: raise ValueError('Invalid duration')
        for span in data['timestamps']:
            start,end=map(float,span)
            if not 0<=start<end<=duration: raise ValueError(f'Invalid seconds: {span}')
            rows.append(dict(video_id=video,query=query,start_time=start,end_time=end,absent=False,duration=duration))
        if not data['timestamps']:
            rows.append(dict(video_id=video,query=query,start_time=None,end_time=None,absent=True,duration=duration))
    if not rows: raise ValueError('Empty segment manifest')
    Path(csv_path).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(csv_path,index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-jsonl',required=True); p.add_argument('--output-csv',required=True)
    a=p.parse_args(); convert_jsonl_to_csv(a.input_jsonl,a.output_csv)
