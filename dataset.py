# dataset.py
# CORRECTED VERSION: This script now loads short video clips, not single frames.

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import os
import cv2
from torchvision import transforms
from transformers import AutoTokenizer
import sys
import random
import re

import config


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

        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.triplets_df)

    def __getitem__(self, idx):
        triplet = self.triplets_df.iloc[idx]
        anchor_frame_path = triplet['frame_path']
        text_query = str(triplet['text_query'])
        relevance = float(triplet['relevance_label'])

        # --- Clip Loading Logic ---
        frames = []
        # Extract the directory and the frame number from the anchor path
        # e.g., from 'CHOLEC80_video01/frame_0000125.jpg'
        directory = os.path.dirname(anchor_frame_path)

        match = re.search(r'(\d+)\.jpg$', anchor_frame_path)
        if not match:
            # Fallback if regex fails, return a random valid sample
            return self.__getitem__(random.randint(0, len(self) - 1))

        anchor_frame_num = int(match.group(1))

        # Load a sequence of 'clip_length' frames starting from the anchor
        for i in range(self.clip_length):
            current_frame_num = anchor_frame_num + i
            # Construct the path for the current frame in the sequence
            current_frame_filename = f"frame_{current_frame_num:07d}.jpg"  # Assumes 7-digit padding
            full_frame_path = os.path.join(config.EXTRACTED_FRAMES_DIR, directory, current_frame_filename)

            try:
                frame = cv2.imread(full_frame_path)
                if frame is None:
                    # If a frame is missing (e.g., at the end of a video), repeat the last valid frame
                    frame = cv2.imread(os.path.join(config.EXTRACTED_FRAMES_DIR, anchor_frame_path))

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.image_transform(frame)
                frames.append(frame)
            except Exception as e:
                # If there's an error, just append the anchor frame again as a fallback
                print(f"Warning: Error loading frame {full_frame_path}: {e}. Re-using anchor frame.", file=sys.stderr)
                anchor_img = cv2.imread(os.path.join(config.EXTRACTED_FRAMES_DIR, anchor_frame_path))
                anchor_img = cv2.cvtColor(anchor_img, cv2.COLOR_BGR2RGB)
                anchor_img = self.image_transform(anchor_img)
                frames.append(anchor_img)

        # Stack the list of frames into a single tensor
        video_clip = torch.stack(frames)  # Shape: [T, C, H, W]

        # --- Text and Label Preparation ---
        text_inputs = self.tokenizer(
            text_query, padding='max_length', truncation=True,
            max_length=config.MAX_TEXT_LENGTH, return_tensors="pt"
        )
        input_ids = text_inputs['input_ids'].squeeze(0)
        attention_mask = text_inputs['attention_mask'].squeeze(0)

        # The relevance label applies to the entire clip
        relevance_tensor = torch.full((self.clip_length,), relevance, dtype=torch.float32)

        return video_clip, input_ids, attention_mask, relevance_tensor


def create_dataloaders(train_csv_path, val_csv_path, clip_length=16):
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    print(f"Loading training data from: {train_csv_path}")
    train_dataset = EndoscopyLocalizationDataset(train_csv_path, tokenizer, clip_length)

    print(f"Loading validation data from: {val_csv_path}")
    val_dataset = EndoscopyLocalizationDataset(val_csv_path, tokenizer, clip_length)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS,
                            pin_memory=True)

    return train_loader, val_loader
