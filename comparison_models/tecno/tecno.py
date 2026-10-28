"""TeCNO (Czempiel et al., MICCAI 2020): a specialised Cholec80 phase-recognition model, re-implemented.

Re-implemented from the paper and the authors' code and configuration (github.com/tobiascz/TeCNO:
modules/cnn/config/config_feature_extract.yml, modules/mstcn/config/config_tcn.yml, models/*.py). Their code
is GPL-3.0 and needs pytorch-lightning 0.8 / albumentations, so it is not vendored. The architecture, losses,
optimisers, batch size, epochs and early-stopping rules follow it; the differences are listed at the end.

Stage 1 (`cnn`): ImageNet ResNet-50 with two heads on the 2048-d pooled feature - phase (7, cross-entropy with
    the authors' Cholec80 class weights) and tool presence (7, binary cross-entropy) - balanced by learned
    uncertainty weights (Kendall et al.; TeCNO's log_vars). Adam, lr 5e-4, batch 80 (their configuration asks
    for 1000, which their code caps at the model's maximum of 80), frames at 1 fps resized to 224 x 224,
    shift/scale/rotate augmentation with p = 0.7, the authors' Cholec80 normalisation. At most 15 epochs, at
    least 6, early stopping with patience 3 on validation phase accuracy; the best epoch is kept.
`extract`: the best stage-1 network's features (2048-d) and phase/tool logits for every sampled frame.
Stage 2 (`tcn`): causal multi-stage temporal convolutional network on the features (2 stages, 8 dilated
    residual layers, 32 feature maps, dropout 0.5; later stages refine the softmax of the previous one), one
    video per batch, Adam lr 7e-5, weighted cross-entropy averaged over the stages, at most 16 / at least 12
    epochs, early stopping with patience 3 on validation accuracy of the last stage; the best epoch is kept.
`predict`: phase queries are answered by the last stage's softmax (online/causal, as TeCNO); tool queries by
    the sigmoid of the stage-1 CNN's auxiliary tool head (TeCNO's temporal network predicts phases only).
    Writes a score matrix for comparison_models/frame_scores.py.

Same split as TeCNO's (videos 1-40 train, 41-48 validation, 49-80 test).

Differences from the authors' code (deliberate, so this is a re-implementation, not a reproduction):
  * stage-2 loss: their code applies the cross-entropy to softmax outputs; here it is applied to the logits,
    as the paper describes;
  * stage-2 model selection: their monitored "val_acc" is a stateful metric that is never reset - the running
    accuracy after each validation video, accumulated over all epochs so far, averaged over the videos. Here the
    best epoch (and early stopping) uses the validation frame accuracy of that epoch; their quantity is logged
    next to it in every row (`val_acc_reference`) so the two selections can be compared;
  * augmentation: torchvision RandomAffine with the parameters of their albumentations ShiftScaleRotate
    (shift 0.1, scale 0.8-1.5, rotation 15 degrees, black border, p = 0.7); interpolation details differ;
  * frames without a phase label (-100) are left out of the phase loss, unobserved tools out of the tool loss
    (their dataframe has labels for every frame).

Every stage resumes at epoch boundaries (latest.pt; the best model is only replaced after latest.pt has
recorded it). Augmentation is drawn per sample from (seed, epoch, index), so it does not depend on the number
of loader processes; on a GPU, replaying an interrupted epoch can still differ in the last bits.
"""
import argparse
import json
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

SCHEMA = 'tecno-1'
IGNORE = -100
# TeCNO's Cholec80 phase weights (median frequency balancing on videos 1-40, i.e. our training split).
CLASS_WEIGHTS = (1.6411019141231247, 0.19090963801041133, 1.0, 0.2502662616859295, 1.9176363911137977,
                 0.9840248158200853, 2.174635818337618)
NORM_MEAN, NORM_STD = (0.3456, 0.2281, 0.2233), (0.2528, 0.2135, 0.2104)   # TeCNO's Cholec80 statistics
TCN = dict(stages=2, layers=8, f_maps=32, dim=2048, classes=7)


# --------------------------------------------------------------------------- models
class TwoHeadResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision import models
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.model.fc = nn.Identity()
        self.fc_phase = nn.Linear(2048, 7)
        self.fc_tool = nn.Linear(2048, 7)
        self.log_vars = nn.Parameter(torch.zeros(2))           # uncertainty balancing of the two losses

    def forward(self, x):
        stem = self.model(x)
        return stem, self.fc_phase(stem), self.fc_tool(stem)


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, channels, causal=True, kernel_size=3):
        super().__init__()
        self.causal, self.dilation = causal, dilation
        padding = dilation * (kernel_size - 1) if causal else dilation
        self.conv_dilated = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        if self.causal:
            out = out[:, :, :-(self.dilation * 2)]             # keep outputs that see only the past
        return x + self.dropout(self.conv_1x1(out))


class SingleStage(nn.Module):
    def __init__(self, layers, f_maps, dim, classes, causal=True):
        super().__init__()
        self.conv_1x1 = nn.Conv1d(dim, f_maps, 1)
        self.layers = nn.ModuleList([DilatedResidualLayer(2 ** i, f_maps, causal) for i in range(layers)])
        self.conv_out_classes = nn.Conv1d(f_maps, classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        return self.conv_out_classes(out)


class MultiStageTCN(nn.Module):
    def __init__(self, stages=2, layers=8, f_maps=32, dim=2048, classes=7, causal=True):
        super().__init__()
        self.stage1 = SingleStage(layers, f_maps, dim, classes, causal)
        self.stages = nn.ModuleList([SingleStage(layers, f_maps, classes, classes, causal) for _ in range(stages - 1)])

    def forward(self, x):                                       # x (B, dim, T) -> logits (stages, B, classes, T)
        out = self.stage1(x)
        outputs = [out]
        for stage in self.stages:
            out = stage(F.softmax(out, dim=1))
            outputs.append(out)
        return torch.stack(outputs)


# --------------------------------------------------------------------------- data
def load_order(order_path, labels_path=None):
    """[(video, frame_idx, paths, labels (n, 14) or None)] in work-order order."""
    order = json.loads(Path(order_path).read_text())
    videos = [[video, np.asarray(v['frame_idx'], dtype=np.int64), list(v['paths']), None]
              for video, v in order['videos'].items()]
    if labels_path:
        z = np.load(labels_path, allow_pickle=False)
        labels, owners = z['labels'].astype(np.int64), [str(o) for o in z['video_ids']]
        if [str(c) for c in z['concepts']] != [f'phase:{i}' for i in range(7)] + [f'tool:{i}' for i in range(7)]:
            raise ValueError(f'{labels_path}: unexpected concept order')
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


def phase_targets(labels):
    """(n, 14) concept labels -> (n,) phase class (-100 where the phase is unknown)."""
    phases = labels[:, :7]
    known = (phases != IGNORE).all(1)
    if np.any((phases != IGNORE).any(1) & ~known) or np.any((phases[known] == 1).sum(1) != 1):
        raise ValueError('phase labels must be one-hot over the 7 phases (or all unknown)')
    out = np.full(len(labels), IGNORE, dtype=np.int64)
    out[known] = phases[known].argmax(1)
    return out


def image_transform(size, training):
    from torchvision.transforms import v2 as T
    ops = [T.Resize((size, size), antialias=True)]
    if training:                                                # albumentations ShiftScaleRotate(0.1, (-0.2, 0.5), 15, p=0.7)
        ops.append(T.RandomApply([T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.5),
                                                 interpolation=T.InterpolationMode.BILINEAR, fill=0)], p=0.7))
    return T.Compose(ops + [T.ToDtype(torch.float32, scale=True), T.Normalize(NORM_MEAN, NORM_STD)])


def sample_seed(seed, epoch, index):
    """The random augmentation of one sample in one epoch: a function of these three numbers only."""
    return (int(seed) * 1_000_003 + int(epoch) * 10_007 + int(index)) % (2 ** 63)


class FrameSet(Dataset):
    def __init__(self, paths, phase, tools, transform, seed=None):
        self.paths, self.phase, self.tools, self.transform = paths, phase, tools, transform
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        from torchvision.transforms.v2 import functional as VF
        with Image.open(self.paths[i]) as image:
            x = VF.pil_to_tensor(image.convert('RGB'))
        if self.seed is None:
            x = self.transform(x)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(sample_seed(self.seed, self.epoch, i))
                x = self.transform(x)
        return x, int(self.phase[i]), torch.from_numpy(self.tools[i].astype(np.float32))


def frame_arrays(videos, observed_only):
    paths, phase, tools = [], [], []
    for _, _, p, labels in videos:
        ph = phase_targets(labels)
        tl = labels[:, 7:]
        keep = (ph != IGNORE) | (tl != IGNORE).any(1) if observed_only else np.ones(len(p), bool)
        paths += [x for x, k in zip(p, keep) if k]
        phase.append(ph[keep])
        tools.append(tl[keep])
    return paths, np.concatenate(phase), np.concatenate(tools)


# --------------------------------------------------------------------------- utilities
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


def file_sha(path):
    from checkpoint_utils import file_hash
    return file_hash(path)


def atomic_save(value, path):
    tmp = Path(path).with_name(Path(path).name + f'.{os.getpid()}.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def commit_best(best, epoch):
    from checkpoint_utils import commit_best as commit
    return commit(best, epoch)


def pending_best(best, epoch):
    from checkpoint_utils import pending_best_path
    return pending_best_path(best, epoch)


def write_json(path, value):
    tmp = Path(path).with_name(Path(path).name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    os.replace(tmp, path)


def clean_journal(journal, finished):
    """Keep the rows of finished epochs only (an interrupted epoch is redone after a resume)."""
    if not journal.exists():
        return
    rows = {}
    for line in journal.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict) and isinstance(r.get('epoch'), int) and r['epoch'] <= finished:
            rows[r['epoch']] = r
    from checkpoint_utils import atomic_write_text
    atomic_write_text(journal, ''.join(json.dumps(rows[e]) + '\n' for e in sorted(rows)))


class EarlyStopping:
    """pytorch-lightning's EarlyStopping(mode='max', min_delta=0, patience) with the trainer's min/max epochs."""

    def __init__(self, patience, min_epochs, max_epochs, state=None):
        self.patience, self.min_epochs, self.max_epochs = patience, min_epochs, max_epochs
        self.best, self.best_epoch, self.wait, self.stopped = -float('inf'), 0, 0, False
        if state:
            self.__dict__.update(state)

    def update(self, epoch, value):
        """After `epoch` (1-based) with validation `value`; returns True when this epoch is the best so far."""
        improved = value > self.best
        if improved:
            self.best, self.best_epoch, self.wait = value, epoch, 0
        else:
            self.wait += 1
        if epoch >= self.max_epochs or (self.wait >= self.patience and epoch >= self.min_epochs):
            self.stopped = True
        return improved

    def state(self):
        return dict(best=self.best, best_epoch=self.best_epoch, wait=self.wait, stopped=self.stopped)


def counts(module):
    return dict(total=sum(p.numel() for p in module.parameters()),
                trainable=sum(p.numel() for p in module.parameters() if p.requires_grad))


# --------------------------------------------------------------------------- stage 1
def cnn_loss(model, phase_logits, tool_logits, phase, tools, ce):
    terms = []
    if bool((phase != IGNORE).any()):
        loss_phase = ce(phase_logits.float(), phase)
        terms.append(torch.exp(-model.log_vars[0]) * loss_phase + model.log_vars[0])
    mask = tools != IGNORE
    if bool(mask.any()):
        loss_tools = F.binary_cross_entropy_with_logits(tool_logits.float()[mask], tools[mask])
        terms.append(torch.exp(-model.log_vars[1]) * loss_tools + model.log_vars[1])
    return sum(terms) if terms else None


@torch.no_grad()
def phase_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, phase, _ in loader:
        _, logits, _ = model(x.to(device))
        keep = phase != IGNORE
        correct += int((logits.argmax(1).cpu()[keep] == phase[keep]).sum())
        total += int(keep.sum())
    if not total:
        raise ValueError('validation frames have no phase labels')
    return correct / total


def train_cnn(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device
    seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    signature = dict(epochs=args.epochs, min_epochs=args.min_epochs, patience=args.patience, lr=args.lr,
                     batch_size=args.batch_size, input_size=args.input_size, seed=args.seed,
                     train=file_sha(args.train_labels), val=file_sha(args.val_labels),
                     train_order=file_sha(args.train_order), val_order=file_sha(args.val_order))
    latest = out / 'latest.pt'
    resume = torch.load(latest, map_location='cpu', weights_only=False) if latest.exists() else None
    if resume is not None and (resume.get('schema') != SCHEMA or resume.get('signature') != signature):
        raise ValueError(f'{latest} was made with different settings or data; use a new output folder')
    commit_best(out / 'best.pt', resume['early_stopping']['best_epoch'] if resume is not None else None)
    if resume is not None and resume['early_stopping']['stopped']:
        print(f'stage 1 already finished (best epoch {resume["early_stopping"]["best_epoch"]})', flush=True)
        return
    _, train_videos = load_order(args.train_order, args.train_labels)
    _, val_videos = load_order(args.val_order, args.val_labels)
    model = TwoHeadResNet50(pretrained=resume is None).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    stopper = EarlyStopping(args.patience, args.min_epochs, args.epochs)
    start = 0
    if resume is not None:
        model.load_state_dict(resume['model'], strict=True)
        optimizer.load_state_dict(resume['optimizer'])
        restore_rng(resume['rng'])
        stopper = EarlyStopping(args.patience, args.min_epochs, args.epochs, resume['early_stopping'])
        start = resume['epoch']
        print(f'resuming stage 1 after epoch {start}', flush=True)
    clean_journal(out / 'training_metrics.jsonl', start)
    ce = nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS, device=device), ignore_index=IGNORE)
    train_set = FrameSet(*frame_arrays(train_videos, True), image_transform(args.input_size, True), seed=args.seed)
    val_loader = DataLoader(FrameSet(*frame_arrays(val_videos, True), image_transform(args.input_size, False)),
                            batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    on_cuda = torch.device(device).type == 'cuda'
    for epoch in range(start, args.epochs):
        model.train()
        if on_cuda:
            torch.cuda.reset_peak_memory_stats()
        train_set.epoch = epoch                                 # (the loader's processes copy the dataset)
        loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                            generator=torch.Generator().manual_seed(args.seed * 1000 + epoch), pin_memory=on_cuda)
        started, total, batches = time.time(), 0.0, 0
        for x, phase, tools in loader:
            _, phase_logits, tool_logits = model(x.to(device))
            loss = cnn_loss(model, phase_logits, tool_logits, phase.to(device), tools.to(device), ce)
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss)
            batches += 1
        trained = time.time()
        acc = phase_accuracy(model, val_loader, device)
        improved = stopper.update(epoch + 1, acc)
        state = dict(schema=SCHEMA, signature=signature, model=model.state_dict(), epoch=epoch + 1)
        if improved:
            atomic_save(dict(state, val_acc_phase=acc), pending_best(out / 'best.pt', epoch + 1))
        atomic_save(dict(state, optimizer=optimizer.state_dict(), rng=rng_state(), early_stopping=stopper.state()),
                    latest)
        commit_best(out / 'best.pt', stopper.best_epoch)
        row = dict(epoch=epoch + 1, train_loss=total / max(batches, 1), val_acc_phase=acc,
                   train_seconds=round(trained - started, 1), val_seconds=round(time.time() - trained, 1),
                   train_frames=len(train_set), optimizer_steps=batches)
        if on_cuda:
            row['peak_gpu_memory_gb'] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
        with open(out / 'training_metrics.jsonl', 'a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        if stopper.stopped:
            print(f'stage 1 stopped after epoch {epoch + 1}; best epoch {stopper.best_epoch} '
                  f'(validation phase accuracy {stopper.best:.4f})', flush=True)
            break


@torch.no_grad()
def extract(args):
    """Features and logits of the best stage-1 network for every video of the work orders (resumable)."""
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device
    sha = file_sha(args.checkpoint)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if ck.get('schema') != SCHEMA:
        raise ValueError(f'{args.checkpoint} is not a TeCNO stage-1 checkpoint')
    input_size = ck['signature']['input_size']
    model = TwoHeadResNet50(pretrained=False).to(device)
    model.load_state_dict(ck['model'], strict=True)
    model.eval()
    transform = image_transform(input_size, False)
    videos = {}
    for order_path in args.work_order:
        for video, grid, paths, _ in load_order(order_path)[1]:
            if video in videos and not np.array_equal(videos[video][0], grid):
                raise ValueError(f'{video}: different sampling grids in the work orders')
            videos[video] = (grid, paths)
    frames, seconds = 0, 0.0
    for video, (grid, paths) in sorted(videos.items()):
        target = out / f'{video}.npz'
        if target.exists():
            z = np.load(target, allow_pickle=False)
            if str(z['cnn_sha']) == sha and np.array_equal(z['frame_idx'], grid):
                frames, seconds = frames + len(grid), seconds + float(z['seconds'])
                continue
        started = time.time()
        loader = DataLoader(FrameSet(paths, np.zeros(len(paths), np.int64), np.zeros((len(paths), 7)), transform),
                            batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
        feats, phase, tools = [], [], []
        for x, _, _ in loader:
            stem, p, t = model(x.to(device))
            feats.append(stem.float().cpu().numpy())
            phase.append(p.float().cpu().numpy())
            tools.append(t.float().cpu().numpy())
        tmp = target.with_name(target.name + f'.{os.getpid()}.tmp.npz')
        np.savez(tmp, frame_idx=grid, features=np.concatenate(feats).astype(np.float32),
                 phase_logits=np.concatenate(phase).astype(np.float32),
                 tool_logits=np.concatenate(tools).astype(np.float32), cnn_sha=np.asarray(sha),
                 seconds=np.asarray(time.time() - started))
        os.replace(tmp, target)
        frames, seconds = frames + len(grid), seconds + time.time() - started
        print(f'{video}: {len(grid)} frames', flush=True)
    write_json(out / 'done.json', dict(cnn_sha=sha, videos=sorted(videos), extracted_frames=frames,
                                       seconds=round(seconds, 3)))


def load_features(folder, video, grid, sha=None):
    path = Path(folder) / f'{video}.npz'
    if not path.exists():
        raise FileNotFoundError(f'no TeCNO features for {video} ({path}); the train stage extracts them')
    z = np.load(path, allow_pickle=False)
    if not np.array_equal(z['frame_idx'], grid):
        raise ValueError(f'{video}: TeCNO features were extracted on a different frame grid')
    if sha is not None and str(z['cnn_sha']) != sha:
        raise ValueError(f'{video}: TeCNO features come from a different stage-1 network')
    return z


# --------------------------------------------------------------------------- stage 2
def tcn_loss(outputs, target, ce):
    """Weighted cross-entropy averaged over the stages (outputs: stages, 1, classes, T; target: T)."""
    if not bool((target != IGNORE).any()):
        return None
    return sum(ce(outputs[s, 0].transpose(0, 1).float(), target) for s in range(outputs.shape[0])) / outputs.shape[0]


@torch.no_grad()
def tcn_accuracy(model, sequences, device, reference=None):
    """Validation frame accuracy of the last stage (used for model selection here), and TeCNO's own monitored
    quantity: their AccuracyStages state is never reset, so after each validation video they log the running
    accuracy over every validation pass so far (all epochs), and the epoch value is the mean of those logs.
    `reference` ({'correct', 'total'}) carries that state across epochs; it is updated in place."""
    model.eval()
    correct, total, logged = 0, 0, []
    for feats, target in sequences:
        pred = model(torch.from_numpy(feats).T.unsqueeze(0).to(device))[-1, 0].argmax(0).cpu().numpy()
        keep = target != IGNORE
        c, n = int((pred[keep] == target[keep]).sum()), int(keep.sum())
        correct += c
        total += n
        if reference is not None:
            reference['correct'] += c
            reference['total'] += n
            logged.append(reference['correct'] / max(reference['total'], 1))
    if not total:
        raise ValueError('validation videos have no phase labels')
    return correct / total, (float(np.mean(logged)) if logged else None)


def sequences(folder, videos, sha):
    out = []
    for video, grid, _, labels in videos:
        z = load_features(folder, video, grid, sha)
        out.append((z['features'].astype(np.float32), phase_targets(labels)))
    return out


def train_tcn(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device
    seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    sha = json.loads((Path(args.features) / 'done.json').read_text())['cnn_sha']
    signature = dict(epochs=args.epochs, min_epochs=args.min_epochs, patience=args.patience, lr=args.lr,
                     seed=args.seed, cnn_sha=sha, train=file_sha(args.train_labels), val=file_sha(args.val_labels),
                     train_order=file_sha(args.train_order), val_order=file_sha(args.val_order), **TCN)
    latest = out / 'latest.pt'
    resume = torch.load(latest, map_location='cpu', weights_only=False) if latest.exists() else None
    if resume is not None and (resume.get('schema') != SCHEMA or resume.get('signature') != signature):
        raise ValueError(f'{latest} was made with different settings or data; use a new output folder')
    commit_best(out / 'best.pt', resume['early_stopping']['best_epoch'] if resume is not None else None)
    model = MultiStageTCN(**TCN).to(device)
    if args.parameter_counts:
        write_json(args.parameter_counts, dict(resnet50_two_heads=counts(TwoHeadResNet50(pretrained=False)),
                                               mstcn=counts(model)))
    if resume is not None and resume['early_stopping']['stopped']:
        print(f'stage 2 already finished (best epoch {resume["early_stopping"]["best_epoch"]})', flush=True)
        return
    train_seq = sequences(args.features, load_order(args.train_order, args.train_labels)[1], sha)
    val_seq = sequences(args.features, load_order(args.val_order, args.val_labels)[1], sha)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    stopper = EarlyStopping(args.patience, args.min_epochs, args.epochs)
    start, reference = 0, dict(correct=0, total=0)
    if resume is not None:
        model.load_state_dict(resume['model'], strict=True)
        optimizer.load_state_dict(resume['optimizer'])
        restore_rng(resume['rng'])
        stopper = EarlyStopping(args.patience, args.min_epochs, args.epochs, resume['early_stopping'])
        start, reference = resume['epoch'], dict(resume['reference_metric_state'])
        print(f'resuming stage 2 after epoch {start}', flush=True)
    clean_journal(out / 'training_metrics.jsonl', start)
    ce = nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS, device=device), ignore_index=IGNORE)
    for epoch in range(start, args.epochs):
        model.train()
        started, total, n = time.time(), 0.0, 0
        for i in torch.randperm(len(train_seq), generator=torch.Generator().manual_seed(args.seed * 1000 + epoch)).tolist():
            feats, target = train_seq[i]
            outputs = model(torch.from_numpy(feats).T.unsqueeze(0).to(device))
            loss = tcn_loss(outputs, torch.from_numpy(target).to(device), ce)
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss)
            n += 1
        trained = time.time()
        acc, acc_reference = tcn_accuracy(model, val_seq, device, reference)
        improved = stopper.update(epoch + 1, acc)
        state = dict(schema=SCHEMA, signature=signature, model=model.state_dict(), epoch=epoch + 1, cnn_sha=sha)
        if improved:
            atomic_save(dict(state, val_acc=acc), pending_best(out / 'best.pt', epoch + 1))
        atomic_save(dict(state, optimizer=optimizer.state_dict(), rng=rng_state(), early_stopping=stopper.state(),
                         reference_metric_state=dict(reference)), latest)
        commit_best(out / 'best.pt', stopper.best_epoch)
        row = dict(epoch=epoch + 1, train_loss=total / max(n, 1), val_acc=acc, val_acc_reference=acc_reference,
                   train_seconds=round(trained - started, 1), val_seconds=round(time.time() - trained, 1),
                   train_videos=len(train_seq), optimizer_steps=n)
        with open(out / 'training_metrics.jsonl', 'a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        if stopper.stopped:
            print(f'stage 2 stopped after epoch {epoch + 1}; best epoch {stopper.best_epoch} '
                  f'(validation accuracy {stopper.best:.4f})', flush=True)
            break


# --------------------------------------------------------------------------- predict
@torch.no_grad()
def predict(args):
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    device = args.device
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if ck.get('schema') != SCHEMA or 'cnn_sha' not in ck:
        raise ValueError(f'{args.checkpoint} is not a TeCNO stage-2 checkpoint')
    model = MultiStageTCN(**TCN).to(device)
    model.load_state_dict(ck['model'], strict=True)
    model.eval()
    order, videos = load_order(args.work_order)
    queries = [q['prompt'] for q in order['queries']]
    if len(set(queries)) != len(queries):
        raise ValueError('duplicate query sentences in the work order')
    started = time.time()
    offsets, frame_idx, blocks = [0], [], []
    for video, grid, _, _ in videos:
        z = load_features(args.features, video, grid, ck['cnn_sha'])
        phase = F.softmax(model(torch.from_numpy(z['features'].astype(np.float32)).T.unsqueeze(0).to(device))[-1, 0],
                          dim=0).T.float().cpu().numpy()                       # (T, 7), causal
        tools = torch.sigmoid(torch.from_numpy(z['tool_logits'].astype(np.float64))).numpy()  # (T, 7), stage-1 head
        both = np.concatenate([phase, tools], axis=1)
        blocks.append(np.stack([both[:, int(q['concept_id']) + (0 if q['kind'] == 'phase' else 7)]
                                for q in order['queries']], axis=1).astype(np.float32))
        frame_idx.extend(grid.tolist())
        offsets.append(len(frame_idx))
    seconds = time.time() - started
    tmp = Path(args.output).with_name(Path(args.output).name + f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(tmp, video_ids=np.asarray([v[0] for v in videos]), offsets=np.asarray(offsets, dtype=np.int64),
                        frame_idx=np.asarray(frame_idx, dtype=np.int64), queries=np.asarray(queries),
                        scores=np.clip(np.concatenate(blocks), 0, 1))
    tmp.replace(args.output)
    extraction = json.loads((Path(args.features) / 'done.json').read_text())
    n = len(frame_idx)
    # Throughput of the whole model: the ResNet-50 pass (measured during extraction) plus the TCN.
    cnn_rate = extraction['extracted_frames'] / extraction['seconds'] if extraction.get('extracted_frames') and \
        extraction.get('seconds') else None
    total_seconds = seconds + (n / cnn_rate if cnn_rate else 0)
    write_json(args.output + '.manifest.json', dict(
        model='TeCNO (re-implementation)', checkpoint=str(args.checkpoint), epoch=ck['epoch'],
        parameters=sum(t.numel() for t in model.parameters()) + counts(TwoHeadResNet50(pretrained=False))['total'],
        score='phases: softmax of the last TCN stage (causal); tools: sigmoid of the stage-1 tool head',
        frames=n, seconds=round(total_seconds, 1) if cnn_rate else None,
        frames_per_second=round(n / max(total_seconds, 1e-9), 1) if cnn_rate else None))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('cnn')
    t = sub.add_parser('tcn')
    for q in (c, t):
        for name in ['train-order', 'train-labels', 'val-order', 'val-labels', 'output']:
            q.add_argument('--' + name, required=True)
        q.add_argument('--epochs', type=int, required=True); q.add_argument('--min-epochs', type=int, required=True)
        q.add_argument('--patience', type=int, default=3); q.add_argument('--lr', type=float, required=True)
        q.add_argument('--seed', type=int, default=42)
    c.add_argument('--batch-size', type=int, default=80); c.add_argument('--input-size', type=int, default=224)
    t.add_argument('--features', required=True); t.add_argument('--parameter-counts')
    e = sub.add_parser('extract')
    e.add_argument('--checkpoint', required=True); e.add_argument('--work-order', required=True, nargs='+')
    e.add_argument('--output', required=True); e.add_argument('--batch-size', type=int, default=256)
    s = sub.add_parser('predict')
    for name in ['checkpoint', 'features', 'work-order', 'output']:
        s.add_argument('--' + name, required=True)
    for q in (c, e):
        q.add_argument('--workers', type=int, default=8)
    for q in (c, t, e, s):
        q.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    dict(cnn=train_cnn, extract=extract, tcn=train_tcn, predict=predict)[args.command](args)


if __name__ == '__main__':
    main()
