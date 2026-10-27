"""Qualitative explanation maps for the paper (GPU, a few minutes per model).

For two test videos and three clinically meaningful queries, the trained model scores every sampled frame
(the same windows as prediction) and keeps the language-guided head's per-patch relevance logits
(inference.score_frames(raw_patch_maps=True)). Six frames per video/query are shown: four where the concept
is present and two where it is absent, spread over the video. The maps are the relevance head's raw patch
decomposition BEFORE the temporal head: they show where the text-conditioned features respond, not an
attribution of the final temporal score (the figure says so).

Writes <output>/maps_<video>_<concept>.npz, explanation_<video>_<concept>.png/.pdf and done.json (with the
checkpoint hash, so that main.py redoes the maps only when the checkpoint changes).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

QUERIES = ('Calot triangle dissection phase', 'a clip applier is present', 'a hook cautery is present')
N_VIDEOS, N_PRESENT, N_ABSENT = 2, 4, 2


def slug(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


def spread(indices, n):
    """n positions spread evenly over a sorted index list (all of them if there are fewer)."""
    indices = list(indices)
    if len(indices) <= n:
        return indices
    return [indices[int(round(i))] for i in np.linspace(0, len(indices) - 1, n)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ['checkpoint', 'triplets', 'metadata', 'frames', 'annotations', 'output']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--checkpoint-sha', default='')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from torchvision.transforms.v2 import functional as VF
    from data_contract import AnnotationIndex, FrameStore, load_video_metadata, query_spec, sample_indices
    from comparison_models.frame_scores import manifest_pairs
    from inference import load_model, score_frames

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'done.json').unlink(missing_ok=True)
    model, _ = load_model(a.checkpoint, a.device)
    cfg = model.config
    size = cfg.DATA.TRAIN_CROP_SIZE
    meta = load_video_metadata(a.metadata)
    index = AnnotationIndex(a.annotations)
    store = FrameStore(a.frames)
    pairs = manifest_pairs(a.triplets)
    videos = sorted({v for v, _ in pairs}, key=lambda v: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', v)])
    written = []
    for video in videos[:N_VIDEOS]:
        m = meta[video]
        grid = sample_indices(m['frame_count'], m['source_fps'], cfg.DATA.SAMPLE_FPS)
        frames = [VF.resize(VF.pil_to_tensor(store.read(video, int(i))), [size, size], antialias=True) for i in grid]
        for query in QUERIES:
            if (video, query) not in pairs:
                continue
            spec = query_spec(query)
            labels = np.asarray([index.label(video, int(i), spec) for i in grid])
            scores, maps = score_frames(model, frames, query, a.device, raw_patch_maps=True)
            chosen = spread(np.flatnonzero(labels == 1), N_PRESENT) + spread(np.flatnonzero(labels == 0), N_ABSENT)
            if not chosen:
                continue
            name = f'{video}_{slug(query)}'
            np.savez_compressed(out / f'maps_{name}.npz', frame_idx=grid[chosen], maps=maps[chosen],
                                scores=scores[chosen], labels=labels[chosen])
            fig, axes = plt.subplots(1, len(chosen), figsize=(2.4 * len(chosen), 2.9))
            axes = np.atleast_1d(axes)
            for ax, k in zip(axes, chosen):
                image = frames[k].permute(1, 2, 0).numpy()
                heat = maps[k].astype(np.float64)
                heat = (heat - heat.min()) / max(heat.max() - heat.min(), 1e-12)
                ax.imshow(image)
                ax.imshow(heat, cmap='inferno', alpha=0.45, extent=(-0.5, size - 0.5, size - 0.5, -0.5),
                          interpolation='bilinear')
                ax.set_title(f'{grid[k] / m["source_fps"] / 60:.1f} min | {"present" if labels[k] == 1 else "absent"}\n'
                             f'score {scores[k]:.2f}', fontsize=8)
                ax.axis('off')
            fig.suptitle(f'{video}: "{query}" - language-guided patch relevance (before the temporal head)', fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.88))
            for ext in ('png', 'pdf'):
                fig.savefig(out / f'explanation_{name}.{ext}', dpi=150)
            plt.close(fig)
            written.append(name)
            print(f'{name}: {len(chosen)} frames', flush=True)
        del frames
    store.close()
    (out / 'done.json').write_text(json.dumps(dict(checkpoint_sha=a.checkpoint_sha, figures=written), indent=2))


if __name__ == '__main__':
    main()
