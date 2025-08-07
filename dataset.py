import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import os
import cv2
import torchvision.transforms.v2 as T
from transformers import AutoTokenizer
import sys
import random
import re
import numpy as np
from project_config import config


class EndoscopyLocalizationDataset(Dataset):
    """
    Custom PyTorch dataset that loads short video CLIPS.
    It uses the triplets CSV to find an "anchor" frame and then loads a sequence of
    consecutive frames around it.
    """

    def __init__(self, triplets_csv_path, tokenizer, clip_length=16, is_training=True):
        self.triplets_df = pd.read_csv(triplets_csv_path)
        self.tokenizer = tokenizer
        self.clip_length = clip_length  # Number of frames per clip
        self.is_training = is_training

        # === UNIFIED, TEMPORALLY CONSISTENT IMAGE TRANSFORMS ===
        # Unified training augmentations, tensor conversion, and normalization
        self.train_transforms = T.Compose([
            # This is now a mandatory operation, ensuring all frames are 224x224
            T.RandomResizedCrop(
                (config.DATA.TRAIN_CROP_SIZE, config.DATA.TRAIN_CROP_SIZE),
                scale=(0.8, 1.0),
                ratio=(0.75, 1.33)
            ),
            T.RandomApply([
                T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                )], p=0.5),
            T.RandomHorizontalFlip(p=config.DATA.AUGMENT_PROB),
            T.RandomApply([T.RandomRotation(degrees=10)], p=0.3),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
            # --- ADD THIS STEP for explicit PIL-to-Tensor conversion ---
            T.ToImage(),
            # --- The following steps now correctly receive a Tensor ---
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Unified validation transforms, tensor conversion, and normalization
        self.val_transforms = T.Compose([
            T.Resize(size=config.DATA.TRAIN_CROP_SIZE + 32),
            T.CenterCrop(size=config.DATA.TRAIN_CROP_SIZE),
            # --- ADD THIS STEP for explicit PIL-to-Tensor conversion ---
            T.ToImage(),
            # --- The following steps now correctly receive a Tensor ---
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # The separate `final_transforms` is no longer needed.
        self.final_transforms = None  # Or just delete the attribute

    def __len__(self):
        return len(self.triplets_df)

    def _load_frame(self, frame_path):
        """Load a single frame with proper error handling"""
        frame = cv2.imread(frame_path)
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        triplet = self.triplets_df.iloc[idx]
        anchor_frame_path = triplet['frame_path']
        text_query = str(triplet['text_query'])
        relevance = float(triplet['relevance_label'])

        # Extract video directory and frame number
        video_dir = os.path.dirname(anchor_frame_path)
        frame_name = os.path.basename(anchor_frame_path)

        # Parse frame number with robust regex
        match = re.search(r'frame_(\d+)\.jpg$', frame_name)
        if not match:
            print(f"Warning: Could not parse frame number from {frame_name}. Using frame 1.")
            anchor_frame_num = 1
        else:
            anchor_frame_num = int(match.group(1))

        # Calculate frame range
        start_frame_num = max(1, anchor_frame_num - self.clip_length // 2)
        frames = []
        last_valid_frame = None

        # Load frames with error handling
        for i in range(self.clip_length):
            current_frame_num = start_frame_num + i
            frame_path = os.path.join(video_dir, f"frame_{current_frame_num:07d}.jpg")

            frame = self._load_frame(frame_path)
            if frame is None:
                if last_valid_frame is not None:
                    # Use last valid frame if available
                    frames.append(last_valid_frame.copy())
                else:
                    # Create black frame as fallback
                    frames.append(np.zeros((config.DATA.HEIGHT, config.DATA.WIDTH, 3), dtype=np.uint8))
            else:
                frames.append(frame)
                last_valid_frame = frame

        # Convert to PIL Images for augmentation
        pil_frames = [T.ToPILImage()(frame) for frame in frames]

        # Apply the single, unified transform pipeline.
        # This will return a single tensor of shape (T, C, H, W).
        if self.is_training:
            transformed_frames = self.train_transforms(pil_frames)
        else:
            transformed_frames = self.val_transforms(pil_frames)

        # 💡 Manually stack the list of tensors into a single tensor.
        # The shape will become (T, C, H, W).
        video_clip = torch.stack(transformed_frames)

        # Now that `video_clip` is a single tensor, we can permute it.
        # The shape becomes (C, T, H, W).
        video_clip = video_clip.permute(1, 0, 2, 3)

        # --- Text and Label Preparation ---
        text_inputs = self.tokenizer(
            text_query, padding='max_length', truncation=True,
            max_length=config.DATA.MAX_TEXT_LENGTH, return_tensors="pt"
        )
        input_ids = text_inputs['input_ids'].squeeze(0)
        attention_mask = text_inputs['attention_mask'].squeeze(0)
        relevance_tensor = torch.full((self.clip_length,), relevance, dtype=torch.float32)

        # Ensure contiguous memory layout
        video_clip = video_clip.contiguous()

        # The DataLoader will handle pinning memory if pin_memory=True is set.
        # We remove the manual call from the dataset worker.

        return {
            'video_clip': video_clip,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': relevance_tensor
        }


def create_dataloaders(train_csv_path, val_csv_path, tokenizer, clip_length=16, subset_ratio=1.0):
    print(f"Loading training data from: {train_csv_path}")
    train_dataset = EndoscopyLocalizationDataset(
        train_csv_path, tokenizer, clip_length, is_training=True
    )

    print(f"Loading validation data from: {val_csv_path}")
    val_dataset = EndoscopyLocalizationDataset(
        val_csv_path, tokenizer, clip_length, is_training=False
    )

    # Apply subsetting if needed
    if subset_ratio < 1.0:
        random.seed(42)
        torch.manual_seed(42)

        train_size = int(len(train_dataset) * subset_ratio)
        val_size = int(len(val_dataset) * subset_ratio)

        train_size = max(1, train_size)
        val_size = max(1, val_size)

        train_indices = random.sample(range(len(train_dataset)), train_size)
        val_indices = random.sample(range(len(val_dataset)), val_size)

        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

        print(f"Using {train_size} samples ({subset_ratio * 100:.1f}%) for training subset.")
        print(f"Using {val_size} samples ({subset_ratio * 100:.1f}%) for validation subset.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if config.DATA.NUM_WORKERS > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if config.DATA.NUM_WORKERS > 0 else False
    )

    return train_loader, val_loader