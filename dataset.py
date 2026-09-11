"""Source-frame sampling with explicit supervision and disjoint-video validation."""
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import v2 as T
from project_config import config
from data_contract import (AnnotationIndex, FrameStore, load_video_metadata,
    path_identity, query_spec, sample_indices, validate_triplets, window_positions, IGNORE_INDEX)

parse_query_kind = query_spec


def image_transform(size, training=False):
    # One stacked clip shares geometric augmentation across its frames.
    ops = [T.RandomResizedCrop((size, size), scale=(0.8, 1.0)), T.RandomHorizontalFlip()] if training else [T.Resize((size, size))]
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
        self.transform = image_transform(cfg.DATA.TRAIN_CROP_SIZE, is_training)
        self.videos, self.grids, self.records = set(), {}, []
        pairs, seen = {}, set()
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
            if is_training:
                center = int(np.abs(grid - anchor).argmin())
                start = max(0, min(center - self.clip_length // 2, len(grid) - self.clip_length))
                key = (video, str(row.text_query), start)
                if key not in seen:
                    self.records.append((*key, spec))
                    seen.add(key)
        if not is_training:
            for (video, query), spec in sorted(pairs.items()):
                for start in window_positions(len(self.grids[video]), self.clip_length, self.clip_length):
                    self.records.append((video, query, start, spec))
        for video in self.videos:
            missing = set(map(int, self.grids[video])) - set(self.store.names(video))
            if missing:
                raise FileNotFoundError(f'{video}: missing {len(missing)} sampled source frames; first={min(missing)}')
        if not self.records:
            raise ValueError('No usable clips')
        if is_training:
            for video,query,start,spec in self.records:
                if not any(self.annotations.label(video,int(i),spec)!=IGNORE_INDEX for i in self.grids[video][start:start+self.clip_length]):
                    raise ValueError(f'{video}/{query}: a training clip has no observed targets on the declared grid')

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
    if subset_ratio < 1:
        train = Subset(train, random.Random(cfg.TRAIN.SEED).sample(range(len(train)), max(1, int(len(train)*subset_ratio))))
    kw = dict(batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.DATA.NUM_WORKERS,
              pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker, persistent_workers=False)
    return DataLoader(train, shuffle=True, **kw), DataLoader(val, shuffle=False, **kw)
