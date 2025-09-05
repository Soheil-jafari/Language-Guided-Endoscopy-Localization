import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

class MomentDETRDataset(Dataset):
    def __init__(self, cfg, split):
        super().__init__()
        self.split = split
        self.cfg = cfg
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        ann_path = os.path.join(cfg.ann_path, f"{split}.jsonl")
        self.annotations = [json.loads(line) for line in open(ann_path, 'r')]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        feature_path = os.path.join(self.cfg.feature_path, f"{ann['video']}.npz")
        features = np.load(feature_path)['features'].astype(np.float32)

        if features.shape[0] > self.cfg.max_v_len:
            indices = np.linspace(0, features.shape[0] - 1, self.cfg.max_v_len).astype(int)
            features = features[indices]

        word_ids = self.tokenizer(
            ann['query'],
            add_special_tokens=True,
            max_length=self.cfg.max_q_len,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )

        # Get timestamps [[start,end]] in seconds
        timestamps = torch.tensor(ann['timestamps'], dtype=torch.float32)  # shape [N,2] or [2]
        if timestamps.ndim == 1:
            timestamps = timestamps.unsqueeze(0)  # [1,2]
        dur = float(ann['duration'])
        assert dur > 0, f"Bad duration for video {ann['video']}: {dur}"
        spans_start_end = timestamps / dur  # normalized [0,1]
        spans_start_end = torch.clamp(spans_start_end, 0.0, 1.0)

        # Sanity: well-formed
        s, e = float(spans_start_end[0, 0]), float(spans_start_end[0, 1])
        assert s < e, f"Bad normalized span order for {ann['video']}: {(s, e)}"

        return {
            'video_feats': torch.from_numpy(features).contiguous(),
            'query': word_ids['input_ids'].squeeze(0).long(),
            'query_mask': word_ids['attention_mask'].squeeze(0).long(),
            'raw_spans': spans_start_end,  # Pass raw spans to collate_fn
            'duration': ann['duration'],
            'video_id': ann['video'],
            'query_str': ann['query']
        }


def collate_fn(batch):
    # Pad video features to the max length in the batch
    v_feats = torch.zeros(len(batch), max(x['video_feats'].shape[0] for x in batch), batch[0]['video_feats'].shape[1])
    v_mask = torch.zeros(len(batch), v_feats.shape[1], dtype=torch.bool)
    for i, item in enumerate(batch):
        len_v = item['video_feats'].shape[0]
        v_feats[i, :len_v] = item['video_feats']
        v_mask[i, :len_v] = True

    queries = torch.stack([x['query'] for x in batch])
    query_masks = torch.stack([x['query_mask'] for x in batch])

    targets = []
    final_num_tokens = v_feats.shape[1]
    grid_size = 1.0 / float(final_num_tokens)

    for item in batch:
        raw_spans = item['raw_spans']  # normalized [N,2] start,end
        raw_spans = torch.as_tensor(raw_spans, dtype=torch.float32)
        if raw_spans.ndim == 1:
            raw_spans = raw_spans.unsqueeze(0)  # [1,2]

        # Snap to the FINAL grid based on the padded tensor length
        # Start/end indices on the grid
        start_indices = torch.floor(raw_spans[:, 0] / grid_size)
        end_indices = torch.ceil(raw_spans[:, 1] / grid_size)

        # Back to normalized [0,1] start,end aligned to grid
        snapped_x1 = start_indices * grid_size
        snapped_x2 = end_indices * grid_size

        # Clamp to [0,1] and enforce length >= one grid
        snapped_x1 = torch.clamp(snapped_x1, 0.0, 1.0)
        snapped_x2 = torch.clamp(snapped_x2, 0.0, 1.0)
        snapped_x2 = torch.maximum(snapped_x2, snapped_x1 + grid_size)

        # -------------- Provide BOTH target formats --------------
        # A) "segments": start,end in [0,1]  (most losses/eval use this)
        segments = torch.stack([snapped_x1, snapped_x2], dim=-1)  # [N,2]

        # B) "spans": center,width in [0,1] (many DETR heads predict this)
        centers = (snapped_x1 + snapped_x2) / 2.0
        widths = (snapped_x2 - snapped_x1)
        widths = torch.clamp(widths, min=grid_size)
        spans_cw = torch.stack([centers, widths], dim=-1)  # [N,2]

        # -------------- Hard sanity checks --------------
        assert segments.numel() > 0, "Empty GT segments after snapping."
        s0, e0 = float(segments[0, 0]), float(segments[0, 1])
        assert 0.0 <= s0 < e0 <= 1.0, f"Invalid segment range after snap: {(s0, e0)}"
        c0, w0 = float(spans_cw[0, 0]), float(spans_cw[0, 1])
        assert 0.0 <= c0 <= 1.0 and w0 > 0.0, f"Invalid span cw after snap: {(c0, w0)}"

        # Labels: single class (0) for MR; length = N
        labels = torch.zeros(segments.shape[0], dtype=torch.long)

        # Put BOTH keys into targets so the model/criterion can pick what it needs
        targets.append({
            'segments': segments,  # normalized start,end  [N,2]
            'spans': spans_cw,  # normalized center,width [N,2]
            'labels': labels
        })

    return {
        'video_feats': v_feats, 'video_mask': v_mask,
        'query': queries, 'query_mask': query_masks,
        'targets': targets,
        'meta': {
            'video_id': [x['video_id'] for x in batch],
            'duration': [x['duration'] for x in batch],
            'query': [x['query_str'] for x in batch]
        }
    }
