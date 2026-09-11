"""Create a reproducible NEW split; never claims to recover dissertation splits."""
import argparse
import json
import random
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata',required=True); p.add_argument('--output',required=True)
    p.add_argument('--train-ratio',type=float,required=True); p.add_argument('--val-ratio',type=float,required=True)
    p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    if not 0<a.train_ratio<1 or not 0<a.val_ratio<1 or a.train_ratio+a.val_ratio>=1: raise ValueError('Invalid split ratios')
    path=Path(a.output)
    if path.exists(): raise FileExistsError(path)
    ids=sorted(json.loads(Path(a.metadata).read_text()))
    random.Random(a.seed).shuffle(ids)
    n=int(len(ids)*a.train_ratio); m=n+int(len(ids)*a.val_ratio)
    result=dict(train=ids[:n],val=ids[n:m],test=ids[m:])
    if any(not v for v in result.values()): raise ValueError('Every split must contain at least one video')
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,indent=2))
    path.with_suffix('.provenance.json').write_text(json.dumps(vars(a),indent=2))


if __name__=='__main__': main()
