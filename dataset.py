# dataset.py
# This script now correctly handles relative paths from the annotation CSV.

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import os
import cv2
from torchvision import transforms
from transformers import AutoTokenizer
import sys
import random
import config


class EndoscopyLocalizationDataset(Dataset):
    """
    Custom PyTorch dataset for loading pre-generated (frame_path, text_query, relevance_label) triplets.
    """

    def __init__(self, triplets_df, tokenizer):
        self.triplets_df = triplets_df
        self.tokenizer = tokenizer
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
        relative_frame_path = triplet['frame_path']
        text_query = str(triplet['text_query'])
        relevance = float(triplet['relevance_label'])

        # --- CORRECTED PATH LOGIC ---
        # Construct the full, absolute path by joining the base directory from config
        # with the relative path from the CSV file.
        # It also removes the leading './' from the relative path if it exists.
        full_frame_path = os.path.join(config.EXTRACTED_FRAMES_DIR, relative_frame_path.lstrip('./'))

        try:
            frame = cv2.imread(full_frame_path)
            if frame is None:
                raise FileNotFoundError(f"Could not read frame: {full_frame_path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.image_transform(frame)
        except Exception as e:
            print(f"Warning: Error with frame {full_frame_path}: {e}. Returning dummy sample.", file=sys.stderr)
            return self.__getitem__(random.randint(0, len(self) - 1))

        text_inputs = self.tokenizer(
            text_query,
            padding='max_length',
            truncation=True,
            max_length=config.MAX_TEXT_LENGTH,
            return_tensors="pt"
        )
        input_ids = text_inputs['input_ids'].squeeze(0)
        attention_mask = text_inputs['attention_mask'].squeeze(0)
        relevance = torch.tensor(relevance, dtype=torch.float32)

        return frame, input_ids, attention_mask, relevance


def create_dataloaders(triplets_csv_path):
    """
    Sets up the datasets and dataloaders for training and validation.
    """
    print(f"Loading and splitting triplets from {triplets_csv_path}...")

    try:
        full_df = pd.read_csv(triplets_csv_path)
    except FileNotFoundError:
        print(f"Error: Triplets CSV not found at {triplets_csv_path}", file=sys.stderr)
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    full_dataset = EndoscopyLocalizationDataset(full_df, tokenizer)

    val_size = int(config.VALIDATION_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Dataset split complete: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    print("Training and validation dataloaders created successfully.")
    return train_loader, val_loader
