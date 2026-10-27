"""Zero-shot CLIP frame scores for a work order (see frame_scores.py): every sampled frame against every
query sentence, score = (cosine similarity + 1) / 2 - the same mapping as benchmark.py, so the operating
threshold is chosen on validation like for every other method. Each frame is encoded once."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class Frames(Dataset):
    def __init__(self, paths, image_processor):
        self.paths, self.image_processor = paths, image_processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as image:
            image = image.convert('RGB')
        return self.image_processor(images=image, return_tensors='pt')['pixel_values'][0]


def unique(seq):
    return list(dict.fromkeys(seq))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-order', required=True)
    p.add_argument('--model-name', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    from transformers import CLIPModel, CLIPProcessor
    order = json.loads(Path(a.work_order).read_text())
    model = CLIPModel.from_pretrained(a.model_name).to(a.device).eval()
    processor = CLIPProcessor.from_pretrained(a.model_name)
    prompts = unique(q['prompt'] for q in order['queries'])
    tokens = processor.tokenizer(prompts, padding=True, truncation=True, max_length=77, return_tensors='pt')
    text = torch.nn.functional.normalize(model.get_text_features(**{k: v.to(a.device) for k, v in tokens.items()}).float(), dim=-1)
    video_ids, offsets, frame_idx, blocks = [], [0], [], []
    started = time.time()
    for video, v in order['videos'].items():
        loader = DataLoader(Frames(v['paths'], processor.image_processor), batch_size=a.batch_size,
                            num_workers=a.workers, shuffle=False)
        sims = []
        for pixels in loader:
            image = model.get_image_features(pixel_values=pixels.to(a.device)).float()
            sims.append((torch.nn.functional.normalize(image, dim=-1) @ text.T).cpu())
        sims = torch.cat(sims).numpy().astype(np.float64)
        blocks.append(np.clip((sims + 1) / 2, 0, 1).astype(np.float32))
        video_ids.append(video)
        frame_idx.extend(v['frame_idx'])
        offsets.append(len(frame_idx))
        print(f'{video}: {len(v["frame_idx"])} frames scored', flush=True)
    tmp = Path(a.output).with_name(Path(a.output).name + f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(tmp, video_ids=np.asarray(video_ids), offsets=np.asarray(offsets, dtype=np.int64),
                        frame_idx=np.asarray(frame_idx, dtype=np.int64), queries=np.asarray(prompts),
                        scores=np.concatenate(blocks))
    tmp.replace(a.output)
    n = len(frame_idx)
    Path(a.output + '.manifest.json').write_text(json.dumps(dict(
        model=a.model_name, score='(cosine + 1) / 2, uncalibrated; threshold chosen on validation',
        parameters=sum(t.numel() for t in model.parameters()), frames=n, seconds=round(time.time() - started, 1),
        frames_per_second=round(n / max(time.time() - started, 1e-9), 1)), indent=2))


if __name__ == '__main__':
    main()
