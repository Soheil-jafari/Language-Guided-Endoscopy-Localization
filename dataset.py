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

        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.DATA.TRAIN_CROP_SIZE, config.DATA.TRAIN_CROP_SIZE)), # Use config.DATA.TRAIN_CROP_SIZE
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

        # --- Clip Loading Logic (this part is correct) ---
        frames = []
        directory = os.path.dirname(anchor_frame_path)
        match = re.search(r'frame_(\d+)\.jpg$', os.path.basename(anchor_frame_path))

        if not match:
            print(f"Warning: Could not parse frame number from {anchor_frame_path}. Returning random sample.",
                  file=sys.stderr)
            return self.__getitem__(random.randint(0, len(self) - 1))

        anchor_frame_num = int(match.group(1))
        start_frame_num = max(1, anchor_frame_num - self.clip_length // 2)

        for i in range(self.clip_length):
            current_frame_num = start_frame_num + i
            video_id = os.path.basename(directory)
            current_frame_filename = f"frame_{current_frame_num:07d}.jpg"
            full_frame_path = os.path.join(config.VIDEO_ROOT_PATH, video_id, current_frame_filename)

            frame = cv2.imread(full_frame_path)
            if frame is None:
                if frames:
                    frames.append(frames[-1].clone())
                else:
                    frames.append(torch.zeros(3, config.DATA.TRAIN_CROP_SIZE, config.DATA.TRAIN_CROP_SIZE))
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.image_transform(frame)
            frames.append(frame)

        video_clip = torch.stack(frames).permute(1, 0, 2, 3)

        # --- Text and Label Preparation (this part is correct) ---
        text_inputs = self.tokenizer(
            text_query, padding='max_length', truncation=True,
            max_length=config.DATA.MAX_TEXT_LENGTH, return_tensors="pt"
        )
        input_ids = text_inputs['input_ids'].squeeze(0)
        attention_mask = text_inputs['attention_mask'].squeeze(0)
        relevance_tensor = torch.full((self.clip_length,), relevance, dtype=torch.float32)

        # === THIS IS THE FIX: Return a dictionary instead of a tuple ===
        return {
            'video_clip': video_clip,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': relevance_tensor
        }

def create_dataloaders(train_csv_path, val_csv_path, clip_length, batch_size, num_workers):
    """
    Creates training and validation dataloaders.
    Accepts batch_size and num_workers as arguments to avoid errors.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL.TEXT_ENCODER_MODEL)

    print(f"--- Creating Dataloaders ---")
    print(f"Loading training data from: {train_csv_path}")
    train_dataset = EndoscopyLocalizationDataset(train_csv_path, tokenizer, clip_length)

    print(f"Loading validation data from: {val_csv_path}")
    val_dataset = EndoscopyLocalizationDataset(val_csv_path, tokenizer, clip_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Training dataloader created with batch size {batch_size} and {num_workers} workers.")
    print(f"Validation dataloader created with batch size {batch_size} and {num_workers} workers.")

    return train_loader, val_loader