"""PeskaVLP comparison (runs in its own environment, see setup_env.sh; uses the official SurgVLP package).

check     build the model with the downloaded weights (strict), encode two sentences and one image: proves,
          before any long step, that the environment, the weights and the text encoder all work.
score     every frame of a work order (comparison_models/frame_scores.py) against every query sentence.
          Zero-shot: score = (cosine similarity + 1) / 2, like CLIP. With --checkpoint (fine-tuned):
          score = sigmoid(scale * cosine + bias), as trained.
finetune  train PeskaVLP's image encoder on our training videos with a frame-wise binary objective for the
          14 concepts (the text encoder stays frozen, so unseen query wordings remain meaningful); the best
          epoch is chosen by validation loss. Resumable at epoch boundaries (latest.pt), deterministic.

The model, its weights and its image preprocessing are the authors' (surgvlp.load); only the scoring and
the fine-tuning objective are ours. This file deliberately imports nothing from the main repository.
"""
import argparse
import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# The official PeskaVLP model configuration (SurgVLP repository, tests/config_peskavlp.py, commit 85e8589),
# with one change that does not alter the model: pretrained='random' instead of 'imagenet' for the ResNet-50.
# 'imagenet' only makes the constructor download torchvision's ImageNet weights, which the PeskaVLP checkpoint
# then overwrites completely (strict loading of every parameter and buffer), so that download is skipped.
MODEL_CONFIG = dict(
    type='PeskaVLP',
    backbone_img=dict(type='img_backbones/ImageEncoder', num_classes=768, pretrained='random',
                      backbone_name='resnet_50', img_norm=False),
    backbone_text=dict(type='text_backbones/BertEncoder', text_bert_type='emilyalsentzer/Bio_ClinicalBERT',
                       text_last_n_layers=4, text_aggregate_method='sum', text_norm=False, text_embedding_dim=768,
                       text_freeze_bert=False, text_agg_tokens=True))
SCHEMA = 'peskavlp-finetune-1'


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_torch_save(value, path):
    tmp = Path(path).with_name(Path(path).name + f'.{os.getpid()}.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def load_peskavlp(weights, device):
    """What surgvlp.load() does (build the official model, strict state-dict load, the authors' image
    transform), except that the checkpoint is read onto the CPU first, so it also loads on a machine without
    the GPU it was saved from."""
    import surgvlp.surgvlp as official                  # (importing the package registers its model classes)
    from surgvlp.codes.models import build_algorithm
    config = copy.deepcopy(MODEL_CONFIG)
    model = build_algorithm(config)
    state = torch.load(str(weights), map_location='cpu', weights_only=False)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    return model, official._transform(official._INPUT_RES[MODEL_CONFIG['type']])


@torch.no_grad()
def encode_text(model, sentences, device):
    import surgvlp
    if len(sentences) < 2:
        raise ValueError('need at least two sentences (surgvlp.tokenize squeezes a single one)')
    tokens = surgvlp.tokenize(list(sentences), device=device)
    model.eval()
    emb = model(None, tokens, mode='text')['text_emb'].float()
    return F.normalize(emb, dim=-1)


class Frames(Dataset):
    def __init__(self, paths, preprocess, labels=None):
        self.paths, self.preprocess, self.labels = paths, preprocess, labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as image:
            x = self.preprocess(image.convert('RGB'))
        if self.labels is None:
            return x
        return x, torch.as_tensor(self.labels[i], dtype=torch.float32)


class Head(torch.nn.Module):
    """Learned temperature and bias on the cosine similarity (frame-wise binary objective)."""
    def __init__(self):
        super().__init__()
        self.log_scale = torch.nn.Parameter(torch.tensor(float(np.log(10.0))))
        self.bias = torch.nn.Parameter(torch.tensor(-5.0))

    def forward(self, cosine):
        return self.log_scale.exp() * cosine + self.bias


def image_cosine(model, images, text):
    emb = model(images, None, mode='video')['img_emb'].float()
    return F.normalize(emb, dim=-1) @ text.T


def order_frames(order):
    paths, owners, frame_idx = [], [], []
    for video, v in order['videos'].items():
        paths += v['paths']
        owners += [video] * len(v['paths'])
        frame_idx += v['frame_idx']
    return paths, owners, frame_idx


# --------------------------------------------------------------------------- check
@torch.no_grad()
def check(args):
    model, preprocess = load_peskavlp(args.weights, args.device)
    text = encode_text(model, ['a grasper is present', 'Calot triangle dissection phase'], args.device)
    image = preprocess(Image.new('RGB', (854, 480), (150, 70, 60))).unsqueeze(0).to(args.device)
    cosine = image_cosine(model, image, text)
    if tuple(text.shape) != (2, 768) or tuple(cosine.shape) != (1, 2) or not bool(torch.isfinite(cosine).all()):
        raise SystemExit(f'PeskaVLP CHECK FAILED: text {tuple(text.shape)}, similarity {tuple(cosine.shape)}')
    print(f'PeskaVLP OK: weights {file_sha(args.weights)[:16]}, {sum(t.numel() for t in model.parameters()):,} '
          f'parameters, embeddings {text.shape[1]}-d, test similarities {cosine.squeeze(0).tolist()}', flush=True)


# --------------------------------------------------------------------------- score
@torch.no_grad()
def score(args):
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    device = args.device
    order = json.loads(Path(args.work_order).read_text())
    model, preprocess = load_peskavlp(args.weights, device)
    head = None
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if ck.get('schema') != SCHEMA or ck.get('weights_sha256') != file_sha(args.weights):
            raise ValueError('fine-tuned checkpoint does not match these PeskaVLP weights')
        model.backbone_img.load_state_dict(ck['image_encoder'], strict=True)
        head = Head().to(device)
        head.load_state_dict(ck['head'], strict=True)
    model.eval()
    prompts = list(dict.fromkeys(q['prompt'] for q in order['queries']))
    text = encode_text(model, prompts, device)
    video_ids, offsets, frame_idx, blocks = [], [0], [], []
    started = time.time()
    for video, v in order['videos'].items():
        loader = DataLoader(Frames(v['paths'], preprocess), batch_size=args.batch_size, num_workers=args.workers)
        sims = torch.cat([image_cosine(model, x.to(device), text).cpu() for x in loader])
        values = torch.sigmoid(head(sims.to(device))).cpu() if head is not None else (sims + 1) / 2
        blocks.append(values.clamp(0, 1).numpy().astype(np.float32))
        video_ids.append(video)
        frame_idx += v['frame_idx']
        offsets.append(len(frame_idx))
        print(f'{video}: {len(v["frame_idx"])} frames scored', flush=True)
    tmp = Path(args.output).with_name(Path(args.output).name + f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(tmp, video_ids=np.asarray(video_ids), offsets=np.asarray(offsets, dtype=np.int64),
                        frame_idx=np.asarray(frame_idx, dtype=np.int64), queries=np.asarray(prompts),
                        scores=np.concatenate(blocks))
    os.replace(tmp, args.output)
    n = len(frame_idx)
    Path(args.output + '.manifest.json').write_text(json.dumps(dict(
        model='PeskaVLP' + (' fine-tuned' if head is not None else ' zero-shot'), weights_sha256=file_sha(args.weights),
        checkpoint=args.checkpoint, parameters=sum(t.numel() for t in model.parameters()),
        score='sigmoid(scale * cosine + bias)' if head is not None else '(cosine + 1) / 2, uncalibrated',
        frames=n, seconds=round(time.time() - started, 1),
        frames_per_second=round(n / max(time.time() - started, 1e-9), 1)), indent=2))


# --------------------------------------------------------------------------- finetune
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


def frame_set(order_path, labels_path):
    order = json.loads(Path(order_path).read_text())
    z = np.load(labels_path, allow_pickle=False)
    paths, owners, _ = order_frames(order)
    labels = z['labels'].astype(np.float32)
    if len(labels) != len(paths) or list(map(str, z['video_ids'])) != owners:
        raise ValueError(f'{labels_path} does not match {order_path}')
    return paths, labels, [str(c) for c in z['canonical']]


def bce(logits, labels):
    mask = labels != -100
    if not bool(mask.any()):
        return logits.sum() * 0, 0
    return F.binary_cross_entropy_with_logits(logits[mask], labels[mask]), int(mask.sum())


def finetune(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device
    seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    signature = dict(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed,
                     weights_sha256=file_sha(args.weights), train=file_sha(args.train_labels), val=file_sha(args.val_labels))
    train_paths, train_labels, canonical = frame_set(args.train_order, args.train_labels)
    val_paths, val_labels, canonical_val = frame_set(args.val_order, args.val_labels)
    if canonical != canonical_val:
        raise ValueError('train/validation concept lists differ')
    model, preprocess = load_peskavlp(args.weights, device)
    text = encode_text(model, canonical, device)               # frozen text encoder: fixed concept embeddings
    for p in model.backbone_text.parameters():
        p.requires_grad = False
    head = Head().to(device)
    params = [dict(params=[p for p in model.backbone_img.parameters()], lr=args.lr),
              dict(params=list(head.parameters()), lr=args.lr * 100)]
    optimizer = torch.optim.AdamW(params, weight_decay=0.01)
    start, best = 0, float('inf')
    latest = out / 'latest.pt'
    if latest.exists():
        ck = torch.load(latest, map_location='cpu', weights_only=False)
        if ck.get('schema') != SCHEMA or ck.get('signature') != signature:
            raise ValueError(f'{latest} was made with different settings or data; use a new output folder')
        model.backbone_img.load_state_dict(ck['image_encoder'], strict=True)
        head.load_state_dict(ck['head'], strict=True)
        optimizer.load_state_dict(ck['optimizer'])
        restore_rng(ck['rng'])
        start, best = ck['epoch'], ck['best_val_loss']
        print(f'resuming after epoch {start}', flush=True)
    train_set = Frames(train_paths, preprocess, train_labels)
    val_loader = DataLoader(Frames(val_paths, preprocess, val_labels), batch_size=args.batch_size,
                            num_workers=args.workers, shuffle=False)
    for epoch in range(start, args.epochs):
        model.backbone_img.train()
        order = torch.randperm(len(train_set), generator=torch.Generator().manual_seed(args.seed * 1000 + epoch))
        loader = DataLoader(torch.utils.data.Subset(train_set, order.tolist()), batch_size=args.batch_size,
                            num_workers=args.workers, shuffle=False, drop_last=len(train_set) > args.batch_size)
        started, total, count = time.time(), 0.0, 0
        for images, labels in loader:
            logits = head(image_cosine(model, images.to(device), text))
            loss, n = bce(logits, labels.to(device))
            if n == 0:
                continue
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss) * n
            count += n
        trained = time.time()
        model.backbone_img.eval()
        vtotal, vcount = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                loss, n = bce(head(image_cosine(model, images.to(device), text)), labels.to(device))
                vtotal += float(loss) * n
                vcount += n
        val_loss = vtotal / max(vcount, 1)
        state = dict(schema=SCHEMA, signature=signature, weights_sha256=signature['weights_sha256'],
                     image_encoder=model.backbone_img.state_dict(), head=head.state_dict(), epoch=epoch + 1,
                     best_val_loss=min(best, val_loss), canonical=canonical)
        if val_loss < best:
            best = val_loss
            atomic_torch_save(state, out / 'best.pt')
        atomic_torch_save(dict(state, optimizer=optimizer.state_dict(), rng=rng_state()), latest)
        row = dict(epoch=epoch + 1, train_loss=total / max(count, 1), val_loss=val_loss,
                   train_seconds=round(trained - started, 1), val_seconds=round(time.time() - trained, 1))
        with open(out / 'training_metrics.jsonl', 'a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('check')
    c.add_argument('--weights', required=True)
    c.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    s = sub.add_parser('score')
    s.add_argument('--work-order', required=True); s.add_argument('--weights', required=True)
    s.add_argument('--output', required=True); s.add_argument('--checkpoint')
    f = sub.add_parser('finetune')
    for name in ['train-order', 'train-labels', 'val-order', 'val-labels', 'weights', 'output']:
        f.add_argument('--' + name, required=True)
    f.add_argument('--epochs', type=int, default=5); f.add_argument('--lr', type=float, default=1e-5)
    f.add_argument('--seed', type=int, default=42)
    for q in (s, f):
        q.add_argument('--batch-size', type=int, default=64); q.add_argument('--workers', type=int, default=8)
        q.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    dict(check=check, score=score, finetune=finetune)[args.command](args)


if __name__ == '__main__':
    main()
