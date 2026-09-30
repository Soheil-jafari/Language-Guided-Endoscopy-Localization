#!/usr/bin/env python3
"""Generate a tiny FAKE dataset in the exact Cholec80 layout, to test the whole pipeline in minutes
without the 70 GB download:

    python tools/make_synthetic_cholec80.py --out /tmp/fake_cholec80 --videos 6
    python main.py --preset smoke --root /tmp/lgel_smoke --data-dir /tmp/fake_cholec80 --random-init

Layout written (same as the real release):
    <out>/videos/videoNN.mp4                 25 fps
    <out>/phase_annotations/videoNN-phase.txt   "Frame<TAB>Phase", one row per frame
    <out>/tool_annotations/videoNN-tool.txt     "Frame<TAB>Grasper...SpecimenBag", one row every 25 frames

The background colour encodes the surgical phase and small coloured squares encode the tools, so a
model can genuinely learn the labels. With --phantom-rows the last two videos get the same defect
the real release has in video15/video37 (one extra phase row one frame past the end of the video),
to exercise the annotation repair step.
"""
import argparse
import random
from pathlib import Path

import numpy as np

PHASES = ['Preparation', 'CalotTriangleDissection', 'ClippingCutting', 'GallbladderDissection',
          'GallbladderPackaging', 'CleaningCoagulation', 'GallbladderRetraction']
TOOLS = ['Grasper', 'Bipolar', 'Hook', 'Scissors', 'Clipper', 'Irrigator', 'SpecimenBag']
FPS = 25


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--videos', type=int, default=6)
    ap.add_argument('--seconds', type=int, default=45)
    ap.add_argument('--size', type=int, nargs=2, default=[96, 64], metavar=('W', 'H'))
    ap.add_argument('--phantom-rows', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    import cv2
    rng = random.Random(a.seed)
    nprng = np.random.default_rng(a.seed)
    out = Path(a.out)
    for sub in ('videos', 'phase_annotations', 'tool_annotations'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    w, h = a.size
    palette = [(int(40 + 30 * i), int(200 - 25 * i), int(60 + 20 * ((i * 3) % 7))) for i in range(7)]
    tool_colors = [(255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    for v in range(1, a.videos + 1):
        name = f'video{v:02d}'
        n = a.seconds * FPS
        cuts = sorted(rng.sample(range(FPS * 3, n - FPS * 3), 6))
        bounds = [0] + cuts + [n]
        phase_of = np.zeros(n, dtype=int)
        for i in range(7):
            phase_of[bounds[i]:bounds[i + 1]] = i
        tool_rows = []
        state = [0] * 7
        for f in range(0, n, FPS):
            for t in range(7):
                if rng.random() < 0.25:
                    state[t] = 1 - state[t]
            tool_rows.append((f, list(state)))
        tool_state_at = {}
        for f, s in tool_rows:
            for k in range(f, min(n, f + FPS)):
                tool_state_at[k] = s
        writer = cv2.VideoWriter(str(out / 'videos' / f'{name}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w, h))
        assert writer.isOpened(), 'cv2.VideoWriter could not open (no mp4v codec?)'
        for f in range(n):
            img = np.empty((h, w, 3), np.uint8)
            img[:] = palette[phase_of[f]]
            img += nprng.integers(0, 12, img.shape, dtype=np.uint8)
            for t, on in enumerate(tool_state_at[f]):
                if on:
                    x = 4 + 12 * t
                    img[4:12, x:x + 8] = tool_colors[t]
            writer.write(img)
        writer.release()
        with open(out / 'phase_annotations' / f'{name}-phase.txt', 'w') as fh:
            fh.write('Frame\tPhase\n')
            for f in range(n):
                fh.write(f'{f}\t{PHASES[phase_of[f]]}\n')
            if a.phantom_rows and v > a.videos - 2:
                fh.write(f'{n}\t{PHASES[phase_of[-1]]}\n')       # one row past the end, same phase
        with open(out / 'tool_annotations' / f'{name}-tool.txt', 'w') as fh:
            fh.write('Frame\t' + '\t'.join(TOOLS) + '\n')
            for f, s in tool_rows:
                fh.write(f'{f}\t' + '\t'.join(map(str, s)) + '\n')
        (out / 'videos' / f'{name}-timestamp.txt').write_text('unused\n')
        print(f'{name}: {n} frames')
    print(f'wrote fake dataset to {out}')


if __name__ == '__main__':
    main()
