"""Source-frame sampling with explicit supervision and disjoint-video validation."""
import math
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision.transforms import v2 as T
from project_config import config
from data_contract import (AnnotationIndex, FrameStore, load_video_metadata,
    path_identity, query_spec, sample_indices, validate_triplets, window_positions, IGNORE_INDEX)

parse_query_kind = query_spec


def image_transform(size, training=False, scale=(0.8, 1.0), ratio=(0.9, 1.1)):
    """Evaluation: the whole frame is resized to (size, size).

    Training applies the same full-frame squish first (to a slightly larger square
    so the crop never upsamples), then a random crop covering `scale` of it with
    aspect jitter `ratio`, plus a horizontal flip. Train and evaluation therefore
    see the same field of view and the same aspect distortion, up to augmentation.
    One stacked clip (T, C, H, W) shares one geometric draw across its frames.
    """
    scale, ratio = tuple(float(x) for x in scale), tuple(float(x) for x in ratio)
    if not (0 < scale[0] <= scale[1] <= 1) or not (0 < ratio[0] <= ratio[1]):
        raise ValueError('TRAIN_AUG_SCALE must lie in (0,1] and TRAIN_AUG_RATIO must be positive')
    if training:
        pre = int(math.ceil(size / math.sqrt(scale[0])))
        ops = [T.Resize((pre, pre), antialias=True),
               T.RandomResizedCrop((size, size), scale=scale, ratio=ratio, antialias=True),
               T.RandomHorizontalFlip()]
    else:
        ops = [T.Resize((size, size), antialias=True)]
    return T.Compose(ops + [T.ToDtype(torch.float32, scale=True),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class EndoscopyLocalizationDataset(Dataset):
    def __init__(self, triplets_csv_path, tokenizer, clip_length=16, is_training=True, settings=None):
        self.settings = settings or config
        cfg = self.settings
        self.triplets_df = pd.read_csv(triplets_csv_path).drop_duplicates().reset_index(drop=True)
        specs = validate_triplets(self.triplets_df)
        self.metadata = load_video_metadata(cfg.VIDEO_METADATA_PATH)
        self.annotations = AnnotationIndex(cfg.CHOLEC80_PARSED_ANNOTATIONS)
        self.store = FrameStore(cfg.EXTRACTED_FRAMES_DIR)
        self.tokenizer, self.clip_length = tokenizer, int(clip_length)
        self.transform = image_transform(cfg.DATA.TRAIN_CROP_SIZE, is_training,
            getattr(cfg.DATA, 'TRAIN_AUG_SCALE', (0.8, 1.0)), getattr(cfg.DATA, 'TRAIN_AUG_RATIO', (0.9, 1.1)))
        self.videos, self.grids, self.records = set(), {}, []
        # Per-record flag: window contains at least one positive observed frame.
        self.positive_windows = []
        pairs = {}
        for (_, row), spec in zip(self.triplets_df.iterrows(), specs):
            video, anchor = path_identity(row.frame_path)
            if video not in self.metadata:
                raise ValueError(f'Missing source-video metadata: {video}')
            self.videos.add(video)
            meta = self.metadata[video]
            if self.annotations.maximum_frame.get(video,-1)>=meta['frame_count']:
                raise ValueError(f'Annotation frame IDs exceed source duration: {video}')
            if not 0 <= anchor < meta['frame_count']:
                raise ValueError(f'Anchor outside video: {video}/{anchor}')
            grid = self.grids.setdefault(video, sample_indices(meta['frame_count'], meta['source_fps'], cfg.DATA.SAMPLE_FPS))
            actual = self.annotations.label(video, anchor, spec)
            if 'relevance_label' in row and pd.notna(row.relevance_label) and int(row.relevance_label) != actual:
                raise ValueError(f'Triplet disagrees with annotations: {video}/{anchor}/{row.text_query}')
            pairs[(video, str(row.text_query))] = spec
        # Windows are cut on a fixed stride over each (video, query) grid. Training
        # uses an overlapping stride (default half a window) and keeps only windows
        # with an observed target; validation uses non-overlapping windows plus the
        # overlapping tail so every source frame is scored.
        if is_training:
            stride = getattr(cfg.DATA, 'TRAIN_WINDOW_STRIDE', None)
            stride = max(1, self.clip_length // 2) if stride is None else int(stride)
            if not 0 < stride <= self.clip_length:
                raise ValueError('TRAIN_WINDOW_STRIDE must be in (0, CLIP_LENGTH]')
        else:
            stride = self.clip_length
        for (video, query), spec in sorted(pairs.items()):
            grid = self.grids[video]
            usable = 0
            for start in window_positions(len(grid), self.clip_length, stride):
                labels = [self.annotations.label(video, int(i), spec) for i in grid[start:start + self.clip_length]]
                observed = any(l != IGNORE_INDEX for l in labels)
                if is_training and not observed:
                    continue
                self.records.append((video, query, start, spec))
                self.positive_windows.append(any(l == 1 for l in labels))
                usable += observed
            if is_training and not usable:
                raise ValueError(f'{video}/{query}: no training window has an observed target on the declared grid')
        for video in self.videos:
            missing = set(map(int, self.grids[video])) - set(self.store.names(video))
            if missing:
                raise FileNotFoundError(f'{video}: missing {len(missing)} sampled source frames; first={min(missing)}')
        if not self.records:
            raise ValueError('No usable clips')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        video, query, start, spec = self.records[idx]
        ids = self.grids[video][start:start + self.clip_length].tolist()
        n = len(ids)
        labels = [self.annotations.label(video, i, spec) for i in ids]
        ids += [ids[-1]] * (self.clip_length - n)
        labels += [IGNORE_INDEX] * (self.clip_length - n)
        frames = torch.stack([T.functional.pil_to_tensor(self.store.read(video, i)) for i in ids])
        clip = self.transform(frames).permute(1, 0, 2, 3).contiguous()
        text = self.tokenizer(query, padding='max_length', truncation=True,
            max_length=self.settings.DATA.MAX_TEXT_LENGTH, return_tensors='pt')
        return dict(video_clip=clip, input_ids=text['input_ids'][0], attention_mask=text['attention_mask'][0],
            labels=torch.tensor(labels, dtype=torch.float32), video_id=video, text_query=query,
            frame_indices=torch.tensor(ids), valid_frames=torch.arange(self.clip_length) < n)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def create_dataloaders(train_csv_path, val_csv_path, tokenizer, clip_length=16, subset_ratio=1.0, settings=None):
    cfg = settings or config
    if not 0 < subset_ratio <= 1:
        raise ValueError('subset_ratio must be in (0, 1]')
    train = EndoscopyLocalizationDataset(train_csv_path, tokenizer, clip_length, True, cfg)
    val = EndoscopyLocalizationDataset(val_csv_path, tokenizer, clip_length, False, cfg)
    if train.videos & val.videos:
        raise ValueError(f'Train/validation video leakage: {sorted(train.videos & val.videos)}')
    indices = list(range(len(train)))
    if subset_ratio < 1:
        indices = random.Random(cfg.TRAIN.SEED).sample(indices, max(1, int(len(train)*subset_ratio)))
    positive_weight = float(getattr(cfg.TRAIN, 'POSITIVE_WINDOW_WEIGHT', 1.0))
    if positive_weight <= 0:
        raise ValueError('POSITIVE_WINDOW_WEIGHT must be positive')
    weights = [positive_weight if train.positive_windows[i] else 1.0 for i in indices]
    train = Subset(train, indices)
    kw = dict(batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.DATA.NUM_WORKERS,
              pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker, persistent_workers=False)
    if positive_weight != 1.0:
        # generator=None -> global torch RNG, which checkpoints save and restore.
        sampler = WeightedRandomSampler(weights, num_samples=len(indices), replacement=True)
        return DataLoader(train, sampler=sampler, **kw), DataLoader(val, shuffle=False, **kw)
    return DataLoader(train, shuffle=True, **kw), DataLoader(val, shuffle=False, **kw)
