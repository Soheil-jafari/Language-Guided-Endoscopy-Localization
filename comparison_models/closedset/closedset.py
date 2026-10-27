"""Our model without language: a closed-set comparison (paper protocol, run `paper_closedset_s42`).

The same pretrained M2CRL video backbone and the same Transformer temporal head as our baseline, but with a
fixed output per concept (7 phases + 7 tools, a multi-label classifier) instead of the text query. Everything
else is taken from our model's own training configuration (main.py writes it: crop, clip length, window
stride, augmentation, optimiser, learning rate, weight decay, warm-up + cosine schedule, epochs, mixed
precision, temporal smoothness weight, seed) and model selection is the same (lowest validation frame NLL).

    ours        backbone -> patch tokens -> text cross-attention -> mean over patches -> relevance (1)
                                                                                     -> temporal head (1)
    closed set  backbone -> patch tokens --------------------------> mean over patches -> linear (14)
                                                                                     -> temporal head (14)

Loss, as ours without its add-ons: BCE(per-frame output) + BCE(temporal head) over the observed labels,
plus TEMPORAL_LOSS_WEIGHT x mean |p(t+1) - p(t)| over consecutive observed frames. Each training window
carries all 14 concepts at once (ours: one query per window), so an epoch has 14x fewer windows; batch
8 x 2 accumulation keeps the number of optimiser steps per epoch close to ours (8 x 24 windows).

train    resumable at epoch boundaries (latest.pt), deterministic; writes best.pt, training_metrics.jsonl
         and parameter_counts.json.
predict  scores every frame of a work order (comparison_models/frame_scores.py): the temporal head's
         sigmoid, windows as in our validation/prediction (non-overlapping plus the overlapping tail,
         overlaps averaged). A query is answered by the output of its concept (it has no text input).
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

SCHEMA = 'closedset-1'
N_CONCEPTS = 14
IGNORE = -100


def apply_overrides(cfg, values):
    """Our training configuration (as main.py writes it for train.py) onto project_config: existing keys only."""
    for k, v in values.items():
        if not hasattr(cfg, k):
            raise ValueError(f'Unknown configuration field: {k}')
        if isinstance(v, dict):
            apply_overrides(getattr(cfg, k), v)
        else:
            setattr(cfg, k, v)


def column(kind, concept_id):
    """Output column of a concept: 0-6 phases, 7-13 tools (data_contract.PHRASES order)."""
    return int(concept_id) + (0 if kind == 'phase' else 7)


class ClosedSetModel(nn.Module):
    def __init__(self, cfg, initialize_backbone=True):
        super().__init__()
        from models import TemporalHead, build_vision_backbone
        if cfg.MODEL.TEMPORAL_HEAD_TYPE != 'TRANSFORMER' or cfg.MODEL.USE_UNCERTAINTY:
            raise ValueError('the closed-set comparison mirrors the baseline (Transformer temporal head, no add-ons)')
        self.config = cfg
        self.vision_backbone = build_vision_backbone(cfg, initialize_backbone)
        dim = self.vision_backbone.embed_dim
        self.frame_head = nn.Linear(dim, N_CONCEPTS)               # counterpart of our per-frame relevance output
        self.temporal_head = TemporalHead(input_dim=dim, output_dim=N_CONCEPTS,
                                          num_attention_heads=cfg.MODEL.HEAD_NUM_ATTENTION_HEADS,
                                          num_layers=cfg.MODEL.HEAD_NUM_LAYERS, max_length=cfg.DATA.CLIP_LENGTH)

    def forward(self, clip):                                       # clip (B, C, T, H, W)
        b, t = clip.shape[0], clip.shape[2]
        tokens = self.vision_backbone.forward_features(clip, get_all=True)[:, 1:, :]
        frames = tokens.reshape(b, t, -1, tokens.shape[-1]).mean(2)   # (B, T, D): frame-major token order
        return self.frame_head(frames), self.temporal_head(frames)     # logits (B, T, 14) each


def loss_terms(raw, refined, labels, pos_weight, temporal_weight):
    mask = labels != IGNORE
    if not bool(mask.any()):
        zero = raw.sum() * 0 + refined.sum() * 0
        return zero, 0
    target = labels.float()[mask]

    def bce(logits):
        return F.binary_cross_entropy_with_logits(logits.float()[mask], target, pos_weight=pos_weight)
    loss = bce(raw) + bce(refined)
    pairs = mask[:, 1:] & mask[:, :-1]
    if temporal_weight > 0 and bool(pairs.any()):
        probs = refined.float().sigmoid()
        loss = loss + temporal_weight * (probs[:, 1:] - probs[:, :-1]).abs()[pairs].mean()
    return loss, int(mask.sum())


# --------------------------------------------------------------------------- data
def load_order(order_path, labels_path=None):
    """[(video, frame_idx array, paths, labels (n, 14) or None)] in work-order order."""
    order = json.loads(Path(order_path).read_text())
    videos = []
    for video, v in order['videos'].items():
        videos.append([video, np.asarray(v['frame_idx'], dtype=np.int64), list(v['paths']), None])
    if labels_path:
        z = np.load(labels_path, allow_pickle=False)
        labels, owners = z['labels'].astype(np.int64), [str(o) for o in z['video_ids']]
        concepts = [str(c) for c in z['concepts']]
        if concepts != [f'phase:{i}' for i in range(7)] + [f'tool:{i}' for i in range(7)]:
            raise ValueError(f'{labels_path}: unexpected concept order {concepts}')
        start = 0
        for entry in videos:
            n = len(entry[1])
            if owners[start:start + n] != [entry[0]] * n:
                raise ValueError(f'{labels_path} does not match {order_path}')
            entry[3] = labels[start:start + n]
            start += n
        if start != len(labels):
            raise ValueError(f'{labels_path} does not match {order_path}')
    return order, videos


class Windows(Dataset):
    """Clips of clip_length frames over each video's sampling grid (padded with the last frame)."""

    def __init__(self, videos, clip_length, stride, transform, training):
        from data_contract import window_positions
        self.videos, self.length, self.transform = videos, int(clip_length), transform
        self.records = []
        for v, (_, grid, _, labels) in enumerate(videos):
            for start in window_positions(len(grid), self.length, stride):
                if training and not bool((labels[start:start + self.length] != IGNORE).any()):
                    continue
                self.records.append((v, start))
        if not self.records:
            raise ValueError('no usable windows')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        from PIL import Image
        from torchvision.transforms.v2 import functional as VF
        v, start = self.records[i]
        _, grid, paths, labels = self.videos[v]
        n = min(self.length, len(grid) - start)
        frames = []
        for path in paths[start:start + n]:
            with Image.open(path) as image:
                frames.append(VF.pil_to_tensor(image.convert('RGB')))
        frames += [frames[-1]] * (self.length - n)
        clip = self.transform(torch.stack(frames)).permute(1, 0, 2, 3).contiguous()
        target = torch.full((self.length, N_CONCEPTS), IGNORE, dtype=torch.float32)
        if labels is not None:
            target[:n] = torch.from_numpy(labels[start:start + n]).float()
        return clip, target, torch.arange(self.length) < n, v, start


@torch.no_grad()
def predict_videos(model, videos, cfg, device, workers, amp_dtype):
    """{video index: (n, 14) probabilities}, overlaps averaged."""
    from dataset import image_transform
    model.eval()
    length = cfg.DATA.CLIP_LENGTH
    loader = DataLoader(Windows(videos, length, length, image_transform(cfg.DATA.TRAIN_CROP_SIZE), False),
                        batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=workers, shuffle=False)
    sums = {v: np.zeros((len(g), N_CONCEPTS)) for v, (_, g, _, _) in enumerate(videos)}
    counts = {v: np.zeros(len(g)) for v, (_, g, _, _) in enumerate(videos)}
    use_amp = torch.device(device).type == 'cuda'
    for clip, _, valid, vs, starts in loader:
        with torch.autocast(device_type=torch.device(device).type, dtype=amp_dtype, enabled=use_amp):
            _, refined = model(clip.to(device))
        probs = refined.float().sigmoid().cpu().numpy()
        for b in range(len(vs)):
            v, s, n = int(vs[b]), int(starts[b]), int(valid[b].sum())
            sums[v][s:s + n] += probs[b, :n]
            counts[v][s:s + n] += 1
    out = {}
    for v in sums:
        if np.any(counts[v] == 0):
            raise RuntimeError(f'{videos[v][0]}: frames without a prediction')
        out[v] = sums[v] / counts[v][:, None]
    return out


def frame_nll(probs, videos):
    total, n = 0.0, 0
    for v, (_, _, _, labels) in enumerate(videos):
        mask = labels != IGNORE
        y = labels[mask].astype(np.float64)
        p = np.clip(probs[v][mask], 1e-7, 1 - 1e-7)
        total += float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).sum())
        n += int(mask.sum())
    if not n:
        raise ValueError('validation has no observed labels')
    return total / n


# --------------------------------------------------------------------------- train
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha(path):
    from checkpoint_utils import file_hash
    return file_hash(path)


def parameter_counts(model):
    return {name: dict(total=sum(p.numel() for p in m.parameters()),
                       trainable=sum(p.numel() for p in m.parameters() if p.requires_grad))
            for name, m in model.named_children()}


def train(args):
    from checkpoint_utils import atomic_save, config_dict, restore_rng, rng_state
    from dataset import image_transform
    from project_config import config as cfg
    overrides = json.loads(Path(args.config).read_text())
    apply_overrides(cfg, overrides)
    device = args.device
    cfg.TRAIN.DEVICE = device
    if cfg.DATA.NUM_FRAMES != cfg.DATA.CLIP_LENGTH:
        raise ValueError('NUM_FRAMES and CLIP_LENGTH must agree')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    seed_all(cfg.TRAIN.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    settings = config_dict(cfg)
    signature = dict(config={k: v for k, v in settings.items() if k not in ('DATA',)},
                     data={k: v for k, v in settings['DATA'].items() if k != 'NUM_WORKERS'},
                     train=file_sha(args.train_labels), val=file_sha(args.val_labels),
                     weights=os.path.getsize(cfg.MODEL.M2CRL_WEIGHTS_PATH) if cfg.MODEL.M2CRL_WEIGHTS_PATH else None)
    _, train_videos = load_order(args.train_order, args.train_labels)
    _, val_videos = load_order(args.val_order, args.val_labels)
    latest = out / 'latest.pt'
    resume = torch.load(latest, map_location='cpu', weights_only=False) if latest.exists() else None
    if resume is not None and (resume.get('schema') != SCHEMA or resume.get('signature') != signature):
        raise ValueError(f'{latest} was made with different settings or data; use a new output folder')
    model = ClosedSetModel(cfg, initialize_backbone=resume is None).to(device)
    Path(args.parameter_counts or out / 'parameter_counts.json').write_text(json.dumps(parameter_counts(model), indent=2))
    stride = cfg.DATA.TRAIN_WINDOW_STRIDE
    stride = max(1, cfg.DATA.CLIP_LENGTH // 2) if stride is None else int(stride)
    train_set = Windows(train_videos, cfg.DATA.CLIP_LENGTH, stride,
                        image_transform(cfg.DATA.TRAIN_CROP_SIZE, True, cfg.DATA.TRAIN_AUG_SCALE, cfg.DATA.TRAIN_AUG_RATIO),
                        True)
    indices = list(range(len(train_set)))
    if cfg.TRAIN.SUBSET_RATIO < 1:                                 # as dataset.create_dataloaders
        indices = random.Random(cfg.TRAIN.SEED).sample(indices, max(1, int(len(indices) * cfg.TRAIN.SUBSET_RATIO)))
    train_set = torch.utils.data.Subset(train_set, indices)
    batch, accumulation = cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.GRADIENT_ACCUMULATION_STEPS
    batches = math.ceil(len(train_set) / batch)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.TRAIN.LEARNING_RATE,
                                  weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    steps = math.ceil(batches / accumulation)                      # the schedule of train.py
    total = steps * cfg.TRAIN.NUM_EPOCHS
    warm = min(steps * cfg.TRAIN.WARMUP_EPOCHS, total - 1)

    def schedule(step):
        if step < warm:
            return step / max(1, warm)
        return .5 * (1 + math.cos(math.pi * min(1., (step - warm) / max(1, total - warm))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    amp_dtype = torch.bfloat16 if cfg.TRAIN.AMP_DTYPE == 'bf16' else torch.float16
    on_cuda = torch.device(device).type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=on_cuda and amp_dtype == torch.float16)
    pos_weight = torch.tensor(float(cfg.TRAIN.BCE_POS_WEIGHT), device=device)
    start, best = 0, float('inf')
    if resume is not None:
        model.load_state_dict(resume['model'], strict=True)
        optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler'])
        scaler.load_state_dict(resume['scaler'])
        restore_rng(resume['rng'])
        start, best = resume['epoch'], resume['best_val_nll']
        print(f'resuming after epoch {start}', flush=True)
    journal = out / 'training_metrics.jsonl'
    if journal.exists():                    # keep the rows of finished epochs only (an interrupted job is redone)
        rows = {}
        for line in journal.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict) and isinstance(r.get('epoch'), int) and r['epoch'] <= start:
                rows[r['epoch']] = r
        journal.write_text(''.join(json.dumps(rows[e]) + '\n' for e in sorted(rows)))
    workers = args.workers
    for epoch in range(start, cfg.TRAIN.NUM_EPOCHS):
        model.train()
        if on_cuda:
            torch.cuda.reset_peak_memory_stats()
        loader = DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=workers,
                            generator=torch.Generator().manual_seed(cfg.TRAIN.SEED * 1000 + epoch),
                            pin_memory=on_cuda)
        started, weighted, observed, group = time.time(), 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, (clip, labels, _, _, _) in enumerate(loader):
            clip, labels = clip.to(device), labels.to(device)
            with torch.autocast(device_type=torch.device(device).type, dtype=amp_dtype, enabled=on_cuda):
                raw, refined = model(clip)
            loss, n = loss_terms(raw, refined, labels, pos_weight, cfg.TRAIN.TEMPORAL_LOSS_WEIGHT)
            if n:
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'non-finite loss at batch {step}')
                scaler.scale(loss * n).backward()                  # batches weighted by observed labels, as train.py
                weighted += float(loss.detach()) * n
                observed += n
                group += n
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                if group:
                    scaler.unscale_(optimizer)
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.div_(group)
                    old = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= old:
                        scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                group = 0
        if not observed:
            raise ValueError('training epoch has no observed targets')
        trained = time.time()
        val_nll = frame_nll(predict_videos(model, val_videos, cfg, device, workers, amp_dtype), val_videos)
        state = dict(schema=SCHEMA, signature=signature, config=settings, model=model.state_dict(), epoch=epoch + 1,
                     best_val_nll=min(best, val_nll))
        if val_nll < best:
            best = val_nll
            atomic_save(state, out / 'best.pt')
        atomic_save(dict(state, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                         scaler=scaler.state_dict(), rng=rng_state()), latest)
        row = dict(epoch=epoch + 1, train_loss=weighted / observed, val_frame_nll=val_nll,
                   train_seconds=round(trained - started, 1), val_seconds=round(time.time() - trained, 1))
        if on_cuda:
            row['peak_gpu_memory_gb'] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
        with journal.open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)


# --------------------------------------------------------------------------- predict
def predict(args):
    from checkpoint_utils import restore_config
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if ck.get('schema') != SCHEMA:
        raise ValueError(f'{args.checkpoint} is not a closed-set checkpoint')
    cfg = restore_config(ck['config'])
    cfg.TRAIN.DEVICE = args.device
    model = ClosedSetModel(cfg, initialize_backbone=False).to(args.device)
    model.load_state_dict(ck['model'], strict=True)
    order, videos = load_order(args.work_order)
    queries = [q['prompt'] for q in order['queries']]
    if len(set(queries)) != len(queries):
        raise ValueError('duplicate query sentences in the work order')
    columns = [column(q['kind'], q['concept_id']) for q in order['queries']]
    amp_dtype = torch.bfloat16 if cfg.TRAIN.AMP_DTYPE == 'bf16' else torch.float16
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    probs = predict_videos(model, videos, cfg, args.device, args.workers, amp_dtype)
    seconds = time.time() - started
    offsets, frame_idx, blocks = [0], [], []
    for v, (_, grid, _, _) in enumerate(videos):
        frame_idx.extend(grid.tolist())
        offsets.append(len(frame_idx))
        blocks.append(probs[v][:, columns].astype(np.float32))
    tmp = Path(args.output).with_name(Path(args.output).name + f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(tmp, video_ids=np.asarray([v[0] for v in videos]), offsets=np.asarray(offsets, dtype=np.int64),
                        frame_idx=np.asarray(frame_idx, dtype=np.int64), queries=np.asarray(queries),
                        scores=np.clip(np.concatenate(blocks), 0, 1))
    tmp.replace(args.output)
    n = len(frame_idx)
    Path(args.output + '.manifest.json').write_text(json.dumps(dict(
        model='closed-set (our backbone and temporal head, no language)', checkpoint=str(args.checkpoint),
        epoch=ck['epoch'], parameters=sum(t.numel() for t in model.parameters()),
        score='sigmoid of the temporal head, per concept', frames=n, seconds=round(seconds, 1),
        frames_per_second=round(n / max(seconds, 1e-9), 1),
        peak_gpu_memory_gb=round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2) if torch.cuda.is_available() else None),
        indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    for name in ['config', 'train-order', 'train-labels', 'val-order', 'val-labels', 'output']:
        t.add_argument('--' + name, required=True)
    t.add_argument('--parameter-counts', help='where to write the parameter counts (default: in --output)')
    s = sub.add_parser('predict')
    for name in ['checkpoint', 'work-order', 'output']:
        s.add_argument('--' + name, required=True)
    for q in (t, s):
        q.add_argument('--workers', type=int, default=8)
        q.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    dict(train=train, predict=predict)[args.command](args)


if __name__ == '__main__':
    main()
