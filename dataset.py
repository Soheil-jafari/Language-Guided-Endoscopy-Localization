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

        # --- Clip Loading Logic ---
        frames = []

        directory = os.path.dirname(anchor_frame_path)

        match = re.search(r'(\d+)\.jpg$', anchor_frame_path)
        if not match:
            # Fallback if regex fails, return a random valid sample
            print(f"Warning: Could not parse frame number from {anchor_frame_path}. Returning random sample.", file=sys.stderr)
            return self.__getitem__(random.randint(0, len(self) - 1))

        anchor_frame_num = int(match.group(1))

        # Determine the start frame for the clip to center it around the anchor, if possible
        # This prevents going out of bounds at the beginning of the video
        start_frame_num = max(1, anchor_frame_num - self.clip_length // 2)

        # Loop to collect frames for the clip
        for i in range(self.clip_length):
            current_frame_num = start_frame_num + i
            # Construct the path for the current frame in the sequence
            current_frame_filename = f"frame_{current_frame_num:07d}.jpg"  # Assumes 7-digit padding
            full_frame_path = os.path.join(config.EXTRACTED_FRAMES_DIR, directory, current_frame_filename)

            try:
                frame = cv2.imread(full_frame_path)
                if frame is None:

                    if frames:
                        frame = frames[-1]
                    else:
                        frame = np.zeros((config.DATA.TRAIN_CROP_SIZE, config.DATA.TRAIN_CROP_SIZE, 3), dtype=np.uint8)
                        print(f"Warning: Could not load initial frame {full_frame_path}. Using black image.", file=sys.stderr)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.image_transform(frame)
                frames.append(frame)
            except Exception as e:
                # Fallback if I/O or processing error
                print(f"Error loading/processing frame {full_frame_path}: {e}. Skipping.", file=sys.stderr)
                # If an error occurs for a frame, fill with a dummy (e.g., black) frame
                dummy_frame = torch.zeros(3, config.DATA.TRAIN_CROP_SIZE, config.DATA.TRAIN_CROP_SIZE)
                frames.append(dummy_frame)


        # Stack the list of frames into a single tensor
        video_clip = torch.stack(frames)  # Original Shape: [T, C, H, W]

        video_clip = video_clip.permute(1, 0, 2, 3) # New Shape: [C, T, H, W]

        # --- Text and Label Preparation ---
        text_inputs = self.tokenizer(
            text_query, padding='max_length', truncation=True,
            max_length=config.DATA.MAX_TEXT_LENGTH, return_tensors="pt" # Use config.DATA.MAX_TEXT_LENGTH
        )
        input_ids = text_inputs['input_ids'].squeeze(0)
        attention_mask = text_inputs['attention_mask'].squeeze(0)

        # The relevance label applies to the entire clip
        relevance_tensor = torch.full((self.clip_length,), relevance, dtype=torch.float32)

        return video_clip, input_ids, attention_mask, relevance_tensor


def create_dataloaders(train_csv_path, val_csv_path, clip_length=16):
    # Use config.MODEL.TEXT_ENCODER_MODEL for consistency
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL.TEXT_ENCODER_MODEL)

    print(f"Loading training data from: {train_csv_path}")
    train_dataset = EndoscopyLocalizationDataset(train_csv_path, tokenizer, clip_length)

    print(f"Loading validation data from: {val_csv_path}")
    val_dataset = EndoscopyLocalizationDataset(val_csv_path, tokenizer, clip_length)

    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=True, # Use config.TRAIN.BATCH_SIZE
                              num_workers=config.DATA.NUM_WORKERS, pin_memory=True) # Use config.DATA.NUM_WORKERS
    val_loader = DataLoader(val_dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=False, # Use config.TRAIN.BATCH_SIZE
                            num_workers=config.DATA.NUM_WORKERS, pin_memory=True) # Use config.DATA.NUM_WORKERS

    return train_loader, val_loader