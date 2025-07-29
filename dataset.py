import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import pandas as pd
import os
import cv2
# Import torchvision.transforms.v2 for temporally consistent transforms
import torchvision.transforms.v2 as T  # <--- NEW IMPORT
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

    def __init__(self, triplets_csv_path, tokenizer, clip_length=16):
        self.triplets_df = pd.read_csv(triplets_csv_path)
        self.tokenizer = tokenizer
        self.clip_length = clip_length  # Number of frames per clip

        # === TEMPORALLY CONSISTENT IMAGE TRANSFORMS ===
        # Define the augmentations. T.Compose is used, but the application will be per-clip.
        # Note: ToTensor and Normalize are applied *after* the clip is formed.
        self.spatial_augmentations = T.Compose([
            T.RandomResizedCrop(
                (config.DATA.TRAIN_CROP_SIZE, config.DATA.TRAIN_CROP_SIZE),
                scale=(0.8, 1.0),
                ratio=(0.75, 1.33)
            ),
            T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            T.RandomHorizontalFlip(p=config.DATA.AUGMENT_PROB),
            T.RandomRotation(degrees=10),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])

        # Normalization and ToTensor are applied after augmentation and clip formation
        self.final_transforms = T.Compose([
            T.ToDtype(torch.float32, scale=True),  # Convert to float and scale to [0,1]
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # === END TEMPORALLY CONSISTENT IMAGE TRANSFORMS ===

    def __len__(self):
        return len(self.triplets_df)

    def __getitem__(self, idx):
        triplet = self.triplets_df.iloc[idx]
        anchor_frame_path = triplet['frame_path']
        text_query = str(triplet['text_query'])
        relevance = float(triplet['relevance_label'])

        frames = []
        video_subdir_name = os.path.basename(os.path.dirname(anchor_frame_path))
        match = re.search(r'frame_(\d+)\.jpg$', os.path.basename(anchor_frame_path))

        if not match:
            print(f"Warning: Could not parse frame number from {anchor_frame_path}. Returning random sample.",
                  file=sys.stderr)
            return self.__getitem__(random.randint(0, len(self) - 1))

        anchor_frame_num = int(match.group(1))
        start_frame_num = max(1, anchor_frame_num - self.clip_length // 2)

        for i in range(self.clip_length):
            current_frame_num = start_frame_num + i
            current_frame_filename = f"frame_{current_frame_num:07d}.jpg"
            full_frame_path = os.path.join(config.VIDEO_ROOT_PATH, video_subdir_name, current_frame_filename)

            frame = cv2.imread(full_frame_path)
            if frame is None:
                if frames:
                    # Ensure the dummy frame is compatible with PIL Image (H, W, C) for spatial_augmentations
                    frames.append(
                        np.array(frames[-1].permute(1, 2, 0).cpu()) if isinstance(frames[-1], torch.Tensor) else frames[
                            -1])
                else:
                    # Start with a black image as a NumPy array (H, W, C)
                    frames.append(
                        np.zeros((config.DATA.TRAIN_CROP_SIZE, config.DATA.TRAIN_CROP_SIZE, 3), dtype=np.uint8))
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image here, before appending to list for spatial_augmentations
            frames.append(T.ToPILImage()(frame))  # <--- Convert to PIL Image here

        # Convert list of PIL Images to a single tensor (T, H, W, C)
        # T.Stack is now part of torchvision.transforms.v2
        video_clip_pil_list = frames

        # Apply spatial augmentations consistently across all frames in the clip
        # T.Compose can now take a list of PIL Images and apply transforms consistently
        video_clip_augmented = self.spatial_augmentations(video_clip_pil_list)  # This is a list of PIL images

        # Convert list of augmented PIL Images to a single tensor (T, C, H, W)
        # T.ToTensor() is implicitly handled by T.ToDtype(..., scale=True) when input is PIL Image
        video_clip_tensor = torch.stack([T.ToDtype(torch.float32, scale=True)(img) for img in video_clip_augmented])  # Stack individual tensors

        # Apply final transforms (normalize)
        video_clip = self.final_transforms(video_clip_tensor)  # (T, C, H, W)

        # Permute to (C, T, H, W) as expected by the model
        video_clip = video_clip.permute(1, 0, 2, 3)

        # --- Text and Label Preparation (this part is correct) ---
        text_inputs = self.tokenizer(
            text_query, padding='max_length', truncation=True,
            max_length=config.DATA.MAX_TEXT_LENGTH, return_tensors="pt"
        )
        input_ids = text_inputs['input_ids'].squeeze(0)
        attention_mask = text_inputs['attention_mask'].squeeze(0)
        relevance_tensor = torch.full((self.clip_length,), relevance, dtype=torch.float32)

        return {
            'video_clip': video_clip,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': relevance_tensor
        }


def create_dataloaders(train_csv_path, val_csv_path, tokenizer, clip_length=16, subset_ratio=1.0):
    # ... (rest of create_dataloaders, no changes needed here) ...
    print(f"Loading training data from: {train_csv_path}")
    train_dataset = EndoscopyLocalizationDataset(train_csv_path, tokenizer, clip_length)

    print(f"Loading validation data from: {val_csv_path}")
    val_dataset = EndoscopyLocalizationDataset(val_csv_path, tokenizer, clip_length)

    # --- Apply subsetting if subset_ratio < 1.0 ---
    if subset_ratio < 1.0:
        random.seed(42)
        torch.manual_seed(42)

        train_size = int(len(train_dataset) * subset_ratio)
        val_size = int(len(val_dataset) * subset_ratio)

        train_size = max(1, train_size)
        val_size = max(1, val_size)

        train_dataset, _ = random_split(train_dataset, [train_size, len(train_dataset) - train_size])
        val_dataset, _ = random_split(val_dataset, [val_size, len(val_dataset) - val_size])

        print(f"Using {train_size} samples ({subset_ratio * 100:.1f}%) for training subset.")
        print(f"Using {val_size} samples ({subset_ratio * 100:.1f}%) for validation subset.")

    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=True,
                              num_workers=config.DATA.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=False,
                            num_workers=config.DATA.NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader
