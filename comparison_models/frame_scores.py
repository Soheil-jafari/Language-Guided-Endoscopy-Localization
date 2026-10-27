"""Shared plumbing for the frame-level comparison models (CLIP, PeskaVLP).

`work-order`: list the sampled frames (1 fps grid, same frames as our model) and the query sentences of a
    split manifest, as a JSON file a scorer can read without this repository (PeskaVLP runs in its own
    Python environment). With --labels it also writes the per-frame labels of all 14 concepts (for
    fine-tuning): -100 = not observed, 0/1 otherwise.
`assemble`: turn a scorer's score matrix (frames x query sentences) into the standard prediction CSV
    (video_id,text_query,frame_idx,time_sec,score,label) for exactly the video/query pairs of the manifest,
    so that build_reference.py / evaluate.py treat it exactly like our model's predictions.

Score matrix file (.npz): video_ids (V,), offsets (V+1,), frame_idx (N,), queries (Q,), scores (N, Q) in [0, 1].
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_contract import (AnnotationIndex, FrameStore, canonical_query, load_video_metadata, path_identity,  # noqa: E402
                           sample_indices, validate_triplets)

CONCEPTS = [('phase', i) for i in range(7)] + [('tool', i) for i in range(7)]


def manifest_pairs(triplets):
    """{(video, query text): (kind, concept_id)} of a manifest, in a stable order."""
    df = pd.read_csv(triplets)
    specs = validate_triplets(df)
    pairs = {}
    for row, spec in zip(df.to_dict('records'), specs):
        video, _ = path_identity(row['frame_path'])
        pairs[(video, str(row['text_query']))] = spec
    return dict(sorted(pairs.items()))


def write_work_order(triplets, metadata, frames, annotations, output, sample_fps=1.0, labels=None, prompts=None):
    pairs = manifest_pairs(triplets)
    meta = load_video_metadata(metadata)
    store = FrameStore(frames)
    queries = {}
    for (_, text), spec in pairs.items():
        queries.setdefault(text, spec)
    videos = {}
    for video in sorted({v for v, _ in pairs}):
        m = meta[video]
        grid = sample_indices(m['frame_count'], m['source_fps'], sample_fps)
        names = store.names(video)
        if store._zips:                                       # noqa: SLF001 - plain folders only
            raise ValueError('frame archives (.zip) are not supported by the comparison scorers; use loose frames')
        missing = sorted(set(map(int, grid)) - set(names))
        if missing:
            raise FileNotFoundError(f'{video}: {len(missing)} sampled frames missing (first {missing[0]})')
        videos[video] = dict(frame_idx=[int(i) for i in grid],
                             paths=[str(Path(frames) / video / names[int(i)]) for i in grid])
    order = dict(sample_fps=float(sample_fps),
                 queries=[dict(text=t, kind=s[0], concept_id=int(s[1]),
                               prompt=(prompts or {}).get(t, t)) for t, s in queries.items()],
                 videos=videos, pairs=[[v, t] for v, t in pairs])
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(output) + '.tmp')
    tmp.write_text(json.dumps(order))
    tmp.replace(output)
    if labels:
        index = AnnotationIndex(annotations)
        rows, owners = [], []
        for video, v in videos.items():
            for idx in v['frame_idx']:
                rows.append([index.label(video, idx, spec) for spec in CONCEPTS])
                owners.append(video)
        np.savez_compressed(labels, labels=np.asarray(rows, dtype=np.int16), video_ids=np.asarray(owners),
                            concepts=np.asarray([f'{k}:{i}' for k, i in CONCEPTS]),
                            canonical=np.asarray([canonical_query(s) for s in CONCEPTS]))
    store.close()
    return order


def read_scores(path):
    z = np.load(path, allow_pickle=False)
    video_ids = [str(v) for v in z['video_ids']]
    offsets = z['offsets'].astype(np.int64)
    frame_idx = z['frame_idx'].astype(np.int64)
    scores = z['scores'].astype(np.float64)
    queries = [str(q) for q in z['queries']]
    if len(offsets) != len(video_ids) + 1 or offsets[0] != 0 or offsets[-1] != len(frame_idx) or \
            scores.shape != (len(frame_idx), len(queries)) or np.any(np.diff(offsets) <= 0):
        raise ValueError(f'{path}: malformed score file')
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError(f'{path}: scores must be finite probabilities in [0, 1]')
    return video_ids, offsets, frame_idx, scores, queries


def assemble(scores_path, triplets, metadata, annotations, output, sample_fps=1.0, prompts=None):
    """prompts: {manifest query text: sentence that was actually scored} (e.g. the authors' prompt)."""
    if Path(output).exists():
        raise FileExistsError(output)
    video_ids, offsets, frame_idx, scores, queries = read_scores(scores_path)
    position = {v: i for i, v in enumerate(video_ids)}
    column = {q: j for j, q in enumerate(queries)}
    meta = load_video_metadata(metadata)
    index = AnnotationIndex(annotations)
    pairs = manifest_pairs(triplets)
    parts = []
    for (video, text), spec in pairs.items():
        scored = (prompts or {}).get(text, text)
        if video not in position or scored not in column:
            raise ValueError(f'no scores for {video} / {scored!r}')
        lo, hi = offsets[position[video]], offsets[position[video] + 1]
        m = meta[video]
        grid = sample_indices(m['frame_count'], m['source_fps'], sample_fps)
        if not np.array_equal(frame_idx[lo:hi], grid):
            raise ValueError(f'{video}: scored frames differ from the sampling grid')
        parts.append(pd.DataFrame(dict(
            video_id=video, text_query=text, frame_idx=grid.astype(np.int64), time_sec=grid / m['source_fps'],
            score=scores[lo:hi, column[scored]], label=[index.label(video, int(i), spec) for i in grid])))
    df = pd.concat(parts, ignore_index=True)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(output) + '.tmp')
    df.to_csv(tmp, index=False)
    tmp.replace(output)
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    w = sub.add_parser('work-order')
    for name in ['triplets', 'metadata', 'frames', 'annotations', 'output']:
        w.add_argument('--' + name, required=True)
    w.add_argument('--sample-fps', type=float, default=1.0)
    w.add_argument('--labels', help='also write per-frame labels of all 14 concepts to this .npz')
    w.add_argument('--prompts', help='JSON {manifest query: sentence to score instead}')
    a = sub.add_parser('assemble')
    for name in ['scores', 'triplets', 'metadata', 'annotations', 'output']:
        a.add_argument('--' + name, required=True)
    a.add_argument('--sample-fps', type=float, default=1.0)
    a.add_argument('--prompts', help='JSON {manifest query: sentence that was scored}')
    args = p.parse_args()
    prompts = json.loads(Path(args.prompts).read_text()) if args.prompts else None
    if args.command == 'work-order':
        write_work_order(args.triplets, args.metadata, args.frames, args.annotations, args.output, args.sample_fps,
                         args.labels, prompts)
    else:
        assemble(args.scores, args.triplets, args.metadata, args.annotations, args.output, args.sample_fps, prompts)


if __name__ == '__main__':
    main()
