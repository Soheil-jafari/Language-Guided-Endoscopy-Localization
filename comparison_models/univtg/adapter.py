"""UniVTG comparison: the official UniVTG model and losses (official/, MIT, unmodified) trained on our data.

Commands (all paths explicit; every step can be re-run and continues where it stopped):
  features  per-second visual features of every video, from the pretrained M2CRL backbone - the same
            weights our model starts from, so the comparison is feature-matched. 16-frame windows on the
            1-fps grid (like our model), each frame's feature = mean of its patch tokens (768-d).
  text      CLIP text token features (last hidden state, 512-d; UniVTG's "clip" text features) of every
            query sentence in the given manifests (training sentences and unseen wordings).
  train     official Model + SetCriterion (dense UniVTG formulation), official fine-tuning recipe.
  predict   for every video/query pair of a manifest: frame scores (foreground probability of the clip the
            frame belongs to, standard prediction CSV) and the model's own windows after temporal NMS.

Deviations from the published QVHighlights recipe, and why (our videos are 30-100 minutes long, not 150 s):
  * clips of 4 s (QVHighlights: 2 s), so that the longest video is about 1,500 clips;
  * batch size 8 (QVHighlights: 32), for GPU memory with sequences up to 1,500 clips;
  * up to 100 windows kept after NMS (default 10), because tool queries have many separate occurrences;
  * targets are built in clip units (clip length 1 in UniVTG's formulas). The original adds clip_len/2 to a
    clip index, which only gives the clip centre when clip_len is 1 s; in clip units it always does;
  * windows are converted back with the normalisation used for training (number of clips x clip length)
    and clipped to the video duration (UniVTG assumes the two are equal);
  * video-query pairs without any positive clip are not used for training (UniVTG's data has none) but are
    scored like every other pair at evaluation;
  * every training mini-batch holds different query sentences. Our data has only 14 sentences, and UniVTG's
    in-batch contrastive saliency loss treats the other queries of a batch as negatives, which is only
    valid when they differ (in QVHighlights every query is unique);
  * the best epoch is chosen on the validation videos by window mAP (mean over tIoU 0.3, 0.5, 0.7).
Everything else (architecture, losses and their weights, optimiser, learning-rate schedule, dropout,
feature normalisation, temporal-element features, NMS threshold) follows the published QVHighlights
fine-tuning script (scripts/qvhl_pretrain.sh) and main/config.py.
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE / 'official'))          # the official `model` and `utils` packages

from data_contract import AnnotationIndex, load_video_metadata, sample_indices, window_positions  # noqa: E402
from comparison_models.frame_scores import manifest_pairs  # noqa: E402
from metrics import temporal_ap  # noqa: E402
from checkpoint_utils import commit_best, pending_best_path  # noqa: E402

SCHEMA = 'univtg-adapter-2'
# Official QVHighlights fine-tuning settings (scripts/qvhl_pretrain.sh, main/config.py defaults), except as noted above.
OFFICIAL = dict(dset_type='mr', train_path='ours', model_id='univtg', span_loss_type='l1', hidden_dim=1024,
                enc_layers=4, dec_layers=2, dim_feedforward=1024, nheads=8, input_dropout=0.5, dropout=0.0,
                droppath=0.1, n_input_proj=2, pre_norm=False, position_embedding='sine', use_txt_pos=False,
                max_q_l=75, b_loss_coef=10.0, g_loss_coef=1.0, f_loss_coef=10.0, s_loss_intra_coef=0.05,
                s_loss_inter_coef=0.01, eos_coef=0.1, temperature=0.07, saliency_margin=0.2, set_cost_span=10.0,
                set_cost_giou=1.0, set_cost_class=4.0, wd=1e-4, lr_warmup=10, lr_drop=80, lr_gamma=0.1,
                grad_clip=0.1, nms_thd=0.7, max_before_nms=1000, max_after_nms=100)
VIDEO_DIM, TEXT_DIM = 768, 512


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_save(value, path):
    tmp = Path(path).with_name(Path(path).name + f'.{os.getpid()}.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def l2_normalize(x, eps=1e-5):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


# --------------------------------------------------------------------------- features
def video_features(args):
    """Per-second M2CRL features, one .npy per video (float16, frames x 768); finished videos are skipped."""
    from project_config import config
    from models import LocalizationFramework
    from dataset import image_transform
    from data_contract import FrameStore
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    weights_sha = file_sha(args.weights) if args.weights else 'random-init'      # (--random-init runs)
    info = out / 'features.json'
    if info.exists():
        old = json.loads(info.read_text())
        if old.get('weights_sha256') != weights_sha or old.get('sample_fps') != args.sample_fps:
            raise ValueError(f'{out} holds features of other backbone weights or another frame rate; use a new folder')
    info.write_text(json.dumps(dict(weights_sha256=weights_sha, sample_fps=args.sample_fps, dim=VIDEO_DIM,
                                    description='M2CRL (pretrained, frozen) per-frame mean of patch tokens')))
    config.MODEL.M2CRL_WEIGHTS_PATH = str(args.weights or '')                 # '' = random initialisation
    config.MODEL.TEXT_ENCODER_MODEL = args.text_model
    config.TRAIN.DEVICE = args.device
    backbone = LocalizationFramework(config, initialize_backbone=True).vision_backbone.to(args.device).eval()
    length, size = config.DATA.CLIP_LENGTH, config.DATA.TRAIN_CROP_SIZE
    transform = image_transform(size)
    meta = load_video_metadata(args.metadata)
    store = FrameStore(args.frames)
    videos = json.loads(Path(args.videos).read_text()) if args.videos.endswith('.json') else args.videos.split(',')
    if isinstance(videos, dict):
        videos = [v for vs in videos.values() for v in vs]
    for video in sorted(set(videos)):
        target = out / f'{video}.npy'
        if target.exists():
            continue
        m = meta[video]
        grid = sample_indices(m['frame_count'], m['source_fps'], args.sample_fps)
        sums = np.zeros((len(grid), VIDEO_DIM), dtype=np.float64)
        counts = np.zeros(len(grid))
        starts = window_positions(len(grid), length, length)
        with torch.no_grad():
            for b in range(0, len(starts), args.batch_size):
                clips, spans = [], []
                for start in starts[b:b + args.batch_size]:
                    ids = grid[start:start + length].tolist()
                    n = len(ids)
                    ids += [ids[-1]] * (length - n)
                    frames = torch.stack([torch.from_numpy(np.asarray(store.read(video, i))).permute(2, 0, 1) for i in ids])
                    clips.append(transform(frames).permute(1, 0, 2, 3))
                    spans.append((start, n))
                x = torch.stack(clips).to(args.device)
                with torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16,
                                    enabled=torch.device(args.device).type == 'cuda'):
                    tokens = backbone.forward_features(x, get_all=True)
                per_frame = tokens[:, 1:, :].float().reshape(len(clips), length, -1, VIDEO_DIM).mean(2).cpu().numpy()
                for (start, n), f in zip(spans, per_frame):
                    sums[start:start + n] += f[:n]
                    counts[start:start + n] += 1
        if np.any(counts == 0):
            raise RuntimeError(f'{video}: frames without a feature')
        tmp = target.with_name(target.name + f'.{os.getpid()}.tmp.npy')
        np.save(tmp, (sums / counts[:, None]).astype(np.float16))
        os.replace(tmp, target)
        print(f'{video}: {len(grid)} frame features', flush=True)
    store.close()


def text_features(args):
    """CLIP last-hidden-state token features of every query sentence in the manifests (one file)."""
    import pandas as pd
    from transformers import CLIPTextModel, CLIPTokenizer
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = torch.load(out, map_location='cpu', weights_only=False) if out.exists() else {}
    sentences = sorted({str(q) for t in args.triplets for q in pd.read_csv(t, usecols=['text_query'])['text_query'].unique()})
    todo = [s for s in sentences if s not in existing]
    if todo:
        tokenizer = CLIPTokenizer.from_pretrained(args.text_model)
        model = CLIPTextModel.from_pretrained(args.text_model).to(args.device).eval()
        with torch.no_grad():
            for s in todo:
                t = tokenizer([s], padding=False, truncation=True, max_length=77, return_tensors='pt')
                hidden = model(**{k: v.to(args.device) for k, v in t.items()}).last_hidden_state[0]
                existing[s] = hidden.float().cpu().numpy().astype(np.float16)
        atomic_save(existing, out)
    print(f'text features: {len(sentences)} sentences ({len(todo)} new)', flush=True)


# --------------------------------------------------------------------------- samples and targets
def build_samples(triplets, metadata, annotations, feature_dir, text_file, clip_frames, sample_fps, positive_only):
    meta = load_video_metadata(metadata)
    index = AnnotationIndex(annotations)
    texts = torch.load(text_file, map_location='cpu', weights_only=False)
    cache, samples = {}, []
    for (video, query), spec in manifest_pairs(triplets).items():
        m = meta[video]
        grid = sample_indices(m['frame_count'], m['source_fps'], sample_fps)
        if video not in cache:
            feats = np.load(Path(feature_dir) / f'{video}.npy').astype(np.float32)
            if len(feats) != len(grid):
                raise ValueError(f'{video}: {len(feats)} feature rows for {len(grid)} sampled frames')
            n_clips = math.ceil(len(grid) / clip_frames)
            pooled = np.stack([feats[c * clip_frames:(c + 1) * clip_frames].mean(0) for c in range(n_clips)])
            cache[video] = l2_normalize(pooled)
        labels = np.asarray([index.label(video, int(i), spec) for i in grid])
        clip_labels = []
        for c in range(len(cache[video])):
            part = labels[c * clip_frames:(c + 1) * clip_frames]
            seen = part[part != -100]
            clip_labels.append(int(len(seen) > 0 and seen.mean() >= 0.5))
        windows, start = [], None
        for c, v in enumerate(clip_labels + [0]):              # runs of positive clips, in clip units
            if v and start is None:
                start = c
            if not v and start is not None:
                windows.append([start, c])
                start = None
        if positive_only and not windows:
            continue
        if query not in texts:
            raise ValueError(f'no text features for {query!r}')
        samples.append(dict(video=video, query=query, vfeat=cache[video], tfeat=l2_normalize(texts[query].astype(np.float32)),
                            windows=windows, grid=grid, labels=labels, source_fps=m['source_fps'],
                            duration=m['frame_count'] / m['source_fps']))
    return samples


def targets_for(sample, rng):
    """Official DatasetMR.__getitem__ targets (dense formulation), in clip units."""
    ctx_l = len(sample['vfeat'])
    ts = ((torch.arange(ctx_l) + 0.5) / ctx_l).unsqueeze(1).repeat(1, 2)
    windows = torch.tensor(sample['windows'], dtype=torch.float32) / ctx_l
    nn_window = torch.zeros_like(ts)
    inside = (ts[:, None, 0] - windows[None, :, 0] >= 0) & (windows[None, :, 1] - ts[:, None, 1] >= 0)
    rows, cols = torch.where(inside)
    if len(rows) == 0:
        nn_window = windows[:1].repeat(ctx_l, 1)
    else:
        nn_window[rows] = windows[cols]
    in_window = ((ts[:, 0] >= nn_window[:, 0]) & (ts[:, 1] <= nn_window[:, 1])).float()
    if in_window.sum() < 1:
        in_window[max(0, min(int(sample['windows'][0][0]), ctx_l - 1))] = 1
    positives = torch.where(in_window > 0)[0].tolist()
    return dict(timestamp=ts, timestamp_window=in_window, span_labels_nn=nn_window, saliency_scores=in_window.clone(),
                saliency_pos_labels=[rng.choice(positives)])


def video_input(sample):
    ctx_l = len(sample['vfeat'])
    tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
    tef = torch.stack([tef_st, tef_st + 1.0 / ctx_l], dim=1)
    return torch.cat([torch.from_numpy(sample['vfeat']), tef], dim=1)


def pad(seqs):
    length = max(len(s) for s in seqs)
    out = torch.zeros((len(seqs), length) + tuple(seqs[0].shape[1:]), dtype=torch.float32)
    mask = torch.zeros(len(seqs), length)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
        mask[i, :len(s)] = 1
    return out, mask


def batch_inputs(samples, device, rng=None):
    vid, vid_mask = pad([video_input(s) for s in samples])
    txt, txt_mask = pad([torch.from_numpy(s['tfeat']) for s in samples])
    inputs = dict(src_txt=txt.to(device), src_txt_mask=txt_mask.to(device), src_vid=vid.to(device),
                  src_vid_mask=vid_mask.to(device))
    if rng is None:
        return inputs, None
    parts = [targets_for(s, rng) for s in samples]
    ts, ts_mask = pad([t['timestamp'] for t in parts])
    targets = dict(timestamp=ts.to(device), timestamp_mask=ts_mask.to(device),
                   timestamp_window=pad([t['timestamp_window'] for t in parts])[0].to(device),
                   span_labels_nn=pad([t['span_labels_nn'] for t in parts])[0].to(device),
                   saliency_scores=pad([t['saliency_scores'] for t in parts])[0].to(device),
                   saliency_pos_labels=torch.LongTensor([t['saliency_pos_labels'] for t in parts]).to(device))
    return inputs, targets


def build(args_dict, device):
    from model.univtg import build_model
    opt = SimpleNamespace(**OFFICIAL, **args_dict, device=device, v_feat_dim=VIDEO_DIM + 2, t_feat_dim=TEXT_DIM)
    model, criterion = build_model(opt)
    return model.to(device), criterion.to(device), opt


class WarmupStepLR(torch.optim.lr_scheduler.StepLR):
    """UniVTG's schedule (main/config.py, MIT): linear warm-up over `warmup_steps` epochs, then x gamma every
    `step_size` epochs. Copied verbatim except for the deprecation warning."""
    def __init__(self, optimizer, warmup_steps, step_size, gamma=0.1, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, step_size, gamma=self.gamma, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch == self.warmup_steps or (self.last_epoch % self.step_size != 0 and self.last_epoch > self.warmup_steps):
            return [group['lr'] for group in self.optimizer.param_groups]
        elif self.last_epoch < self.warmup_steps:
            return [group['initial_lr'] * float(self.last_epoch + 1) / float(self.warmup_steps) for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]


def distinct_query_batches(samples, batch_size, rng):
    """Shuffled mini-batches in which no query sentence occurs twice (see the note at the top). Every sample
    is used exactly once per epoch; batches are filled greedily and come out in random order."""
    order = list(range(len(samples)))
    rng.shuffle(order)
    full, open_batches = [], []
    for i in order:
        query = samples[i]['query']
        for batch in open_batches:
            if query not in batch[1]:
                batch[0].append(i)
                batch[1].add(query)
                break
        else:
            batch = ([i], {query})
            open_batches.append(batch)
        if len(batch[0]) == batch_size:
            open_batches.remove(batch)
            full.append(batch[0])
    batches = full + [b[0] for b in open_batches]
    rng.shuffle(batches)
    return batches


# --------------------------------------------------------------------------- inference
@torch.no_grad()
def run_model(model, samples, device, batch_size, opt):
    """Per sample: foreground probability of every clip, and windows [st, ed, score] in seconds after NMS."""
    from utils.temporal_nms import temporal_nms
    model.eval()
    results = []
    for b in range(0, len(samples), batch_size):
        chunk = samples[b:b + batch_size]
        inputs, _ = batch_inputs(chunk, device)
        out = model(**inputs)
        prob = out['pred_logits'][..., 0].float().cpu()
        spans = out['pred_spans'].float().cpu()
        for i, s in enumerate(chunk):
            ctx_l = len(s['vfeat'])
            ts = ((torch.arange(ctx_l) + 0.5) / ctx_l).unsqueeze(1).repeat(1, 2)
            scale = ctx_l * opt.clip_frames / opt.sample_fps                # training normalisation, in seconds
            win = ((ts + spans[i, :ctx_l]) * scale).clamp(0, s['duration'])
            score = prob[i, :ctx_l]
            ranked = sorted(([float(w[0]), float(w[1]), float(p)] for w, p in zip(win, score) if w[1] - w[0] > 1e-6),
                            key=lambda x: x[2], reverse=True)[:opt.max_before_nms]
            kept = temporal_nms(ranked, nms_thd=opt.nms_thd, max_after_nms=opt.max_after_nms) if ranked else []
            results.append((score.numpy(), [[a, max(min(e, s['duration']), a), p] for a, e, p in kept if e > a]))
    return results


def gt_spans(sample):
    """Ground-truth spans in seconds from the 1-fps labels: runs of positive frames, [first, last + 1 step)."""
    times = sample['grid'] / sample['source_fps']
    ends = np.r_[times[1:], sample['duration']]
    spans, start = [], None
    for i, v in enumerate(list(sample['labels'] == 1) + [False]):
        if v and start is None:
            start = i
        if not v and start is not None:
            spans.append([float(times[start]), float(ends[i - 1])])
            start = None
    return spans


def window_map(model, samples, device, batch_size, opt):
    preds, truth = {}, {}
    for s, (_, windows) in zip(samples, run_model(model, samples, device, batch_size, opt)):
        key = (s['video'], s['query'])
        preds[key], truth[key] = windows, gt_spans(s)
    values = [temporal_ap(preds, truth, t) for t in (0.3, 0.5, 0.7)]
    return float(np.mean([v for v in values if v is not None])) if any(v is not None for v in values) else 0.0


# --------------------------------------------------------------------------- train / predict
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def train(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    settings = dict(clip_frames=args.clip_frames, sample_fps=args.sample_fps, epochs=args.epochs, lr=args.lr,
                    batch_size=args.batch_size, eval_every=args.eval_every, seed=args.seed, max_v_l=100000)
    signature = dict(settings, schema=SCHEMA, train=file_sha(args.train), val=file_sha(args.val),
                     text=file_sha(args.text_features))
    train_samples = build_samples(args.train, args.metadata, args.annotations, args.features, args.text_features,
                                  args.clip_frames, args.sample_fps, positive_only=True)
    val_samples = build_samples(args.val, args.metadata, args.annotations, args.features, args.text_features,
                                args.clip_frames, args.sample_fps, positive_only=False)
    if not train_samples:
        raise ValueError('no training video/query pair has a positive clip')
    model, criterion, opt = build(settings, args.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=opt.wd)
    scheduler = WarmupStepLR(optimizer, warmup_steps=opt.lr_warmup, step_size=opt.lr_drop, gamma=opt.lr_gamma)
    start, best, best_epoch = 0, -1.0, None
    latest = out / 'latest.pt'
    if latest.exists():
        ck = torch.load(latest, map_location='cpu', weights_only=False)
        if ck.get('signature') != signature:
            raise ValueError(f'{latest} was made with different settings or data; use a new output folder')
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        restore_rng(ck['rng'])
        start, best, best_epoch = ck['epoch'], ck['best_val_map'], ck.get('best_epoch')
        print(f'resuming after epoch {start}', flush=True)
    commit_best(out / 'best.pt', best_epoch)              # best.pt only holds an epoch that latest.pt committed
    weights = criterion.weight_dict
    for epoch in range(start, args.epochs):
        started = time.time()
        model.train()
        criterion.train()
        rng = random.Random(args.seed * 1000 + epoch)
        total, steps = 0.0, 0
        for batch in distinct_query_batches(train_samples, args.batch_size, rng):
            chunk = [train_samples[i] for i in batch]
            inputs, targets = batch_inputs(chunk, args.device, rng)
            outputs = model(**inputs)
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict[k] * weights[k] for k in loss_dict if k in weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite UniVTG loss at epoch {epoch + 1}')
            optimizer.zero_grad()
            loss.backward()
            if opt.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()
            total += float(loss)
            steps += 1
        scheduler.step()
        row = dict(epoch=epoch + 1, train_loss=total / max(steps, 1), train_seconds=round(time.time() - started, 1))
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            val_map = window_map(model, val_samples, args.device, args.batch_size, opt)
            row['val_window_map'] = val_map
            if val_map > best:
                best, best_epoch = val_map, epoch + 1
                atomic_save(dict(signature=signature, model=model.state_dict(), epoch=epoch + 1, val_window_map=val_map,
                                 settings=settings), pending_best_path(out / 'best.pt', epoch + 1))
        atomic_save(dict(signature=signature, model=model.state_dict(), optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(), rng=rng_state(), epoch=epoch + 1, best_val_map=best,
                         best_epoch=best_epoch), latest)
        commit_best(out / 'best.pt', best_epoch)
        with open(out / 'training_metrics.jsonl', 'a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    if not (out / 'best.pt').exists():
        raise RuntimeError('training finished without a validated checkpoint')


def predict(args):
    import pandas as pd
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    settings = ck['settings']
    model, _, opt = build(settings, args.device)
    model.load_state_dict(ck['model'])
    samples = build_samples(args.triplets, args.metadata, args.annotations, args.features, args.text_features,
                            settings['clip_frames'], settings['sample_fps'], positive_only=False)
    started = time.time()
    results = run_model(model, samples, args.device, args.batch_size, opt)
    parts, native = [], []
    for s, (clip_scores, windows) in zip(samples, results):
        frame_scores = np.repeat(clip_scores, settings['clip_frames'])[:len(s['grid'])]
        parts.append(pd.DataFrame(dict(video_id=s['video'], text_query=s['query'], frame_idx=s['grid'].astype(np.int64),
                                       time_sec=s['grid'] / s['source_fps'], score=np.clip(frame_scores, 0, 1),
                                       label=s['labels'])))
        native.append(dict(video_id=s['video'], text_query=s['query'], duration=s['duration'], predictions=windows))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(args.output + '.tmp')
    pd.concat(parts, ignore_index=True).to_csv(tmp, index=False)
    tmp.replace(args.output)
    Path(args.segments).write_text(''.join(json.dumps(r) + '\n' for r in native))
    n = int(sum(len(s['grid']) for s in samples))
    Path(args.output + '.manifest.json').write_text(json.dumps(dict(
        model='UniVTG (official code) on M2CRL features', checkpoint_epoch=ck['epoch'],
        val_window_map=ck.get('val_window_map'), parameters=sum(p.numel() for p in model.parameters()),
        score='foreground probability of the clip containing the frame', frame_scores=n,
        seconds=round(time.time() - started, 1),
        note='on precomputed M2CRL features: the time to extract them is not included'), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('features')
    for name in ['weights', 'metadata', 'frames', 'videos', 'output']:
        f.add_argument('--' + name, required=True)
    f.add_argument('--text-model', default='openai/clip-vit-base-patch32')
    f.add_argument('--batch-size', type=int, default=4)
    t = sub.add_parser('text')
    t.add_argument('--triplets', nargs='+', required=True); t.add_argument('--output', required=True)
    t.add_argument('--text-model', default='openai/clip-vit-base-patch32')
    r = sub.add_parser('train')
    for name in ['train', 'val', 'metadata', 'annotations', 'features', 'text-features', 'output']:
        r.add_argument('--' + name, required=True)
    r.add_argument('--clip-frames', type=int, default=4); r.add_argument('--epochs', type=int, default=200)
    r.add_argument('--eval-every', type=int, default=5); r.add_argument('--batch-size', type=int, default=8)
    r.add_argument('--lr', type=float, default=1e-4); r.add_argument('--seed', type=int, default=42)
    d = sub.add_parser('predict')
    for name in ['checkpoint', 'triplets', 'metadata', 'annotations', 'features', 'text-features', 'output', 'segments']:
        d.add_argument('--' + name, required=True)
    d.add_argument('--batch-size', type=int, default=8)
    for q in (f, t, r, d):
        q.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    for q in (f, r):
        q.add_argument('--sample-fps', type=float, default=1.0)
    args = p.parse_args()
    dict(features=video_features, text=text_features, train=train, predict=predict)[args.command](args)


if __name__ == '__main__':
    main()
