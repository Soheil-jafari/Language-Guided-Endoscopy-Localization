# train.py
# CORRECTED VERSION: This script now works with batches of video clips.

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import argparse

import config
from models import LocalizationFramework
from dataset import create_dataloaders  # This will now use the clip-based dataset


class DualObjectiveLoss(nn.Module):
    def __init__(self, temporal_weight=config.TEMPORAL_LOSS_WEIGHT):
        super().__init__()
        self.relevance_loss = nn.BCEWithLogitsLoss()
        self.temporal_weight = temporal_weight

    def forward(self, refined_scores, raw_scores, ground_truth_relevance):
        # refined_scores and ground_truth_relevance are now [B, T]
        primary_loss = self.relevance_loss(refined_scores, ground_truth_relevance)

        temporal_loss = 0.0
        if self.temporal_weight > 0 and refined_scores.shape[1] > 1:
            scores_t = refined_scores[:, 1:]
            scores_t_minus_1 = refined_scores[:, :-1]
            temporal_loss = torch.mean(torch.abs(scores_t - scores_t_minus_1))

        total_loss = primary_loss + (self.temporal_weight * temporal_loss)
        return total_loss, primary_loss, temporal_loss


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", unit="batch", leave=False)

    for video_clip, input_ids, attention_mask, relevance in progress_bar:
        # Move all tensors to the GPU
        video_clip, input_ids, attention_mask, relevance = video_clip.to(device), input_ids.to(
            device), attention_mask.to(device), relevance.to(device)

        optimizer.zero_grad()

        # The model now directly accepts the video clip
        raw_scores, refined_scores = model(video_clip, input_ids, attention_mask)

        # Squeeze the last dimension of scores to match relevance shape [B, T]
        loss, _, _ = criterion(refined_scores.squeeze(-1), raw_scores.squeeze(-1), relevance)

        if isinstance(loss, torch.Tensor) and loss.numel() > 1:
            loss = loss.mean()

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(dataloader)


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss, all_preds, all_labels = 0.0, [], []
    progress_bar = tqdm(dataloader, desc="Validating", unit="batch", leave=False)

    with torch.no_grad():
        for video_clip, input_ids, attention_mask, relevance in progress_bar:
            video_clip, input_ids, attention_mask, relevance = video_clip.to(device), input_ids.to(
                device), attention_mask.to(device), relevance.to(device)

            raw_scores, refined_scores = model(video_clip, input_ids, attention_mask)
            loss, _, _ = criterion(refined_scores.squeeze(-1), raw_scores.squeeze(-1), relevance)

            if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                loss = loss.mean()

            running_loss += loss.item()

            preds = torch.sigmoid(refined_scores.squeeze(-1)) > 0.5
            all_preds.extend(preds.flatten().cpu().numpy())
            all_labels.extend(relevance.flatten().cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    return avg_loss, accuracy


def main(args):
    # (This main function logic remains largely the same)
    print("--- Starting Training Pipeline ---")
    device = torch.device(config.DEVICE)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    model = LocalizationFramework().to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    if args.debug:
        print("--- RUNNING IN DEBUG MODE ---")
        train_csv = config.TRAIN_TRIPLETS_CSV_PATH.replace(".csv", "_DEBUG.csv")
        val_csv = config.VAL_TRIPLETS_CSV_PATH.replace(".csv", "_DEBUG.csv")
        epochs = 1
    else:
        train_csv = config.TRAIN_TRIPLETS_CSV_PATH
        val_csv = config.VAL_TRIPLETS_CSV_PATH
        epochs = config.EPOCHS

    train_loader, val_loader = create_dataloaders(
        train_csv_path=train_csv,
        val_csv_path=val_csv,
        clip_length=config.CLIP_LENGTH  # Pass clip length from config
    )

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE)
    criterion = DualObjectiveLoss()

    best_val_loss = float('inf')
    print("\n--- Beginning Training and Validation Epochs ---")

    for epoch in range(epochs):
        print(f"\n===== Epoch {epoch + 1}/{epochs} =====")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate_one_epoch(model, val_loader, criterion, device)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"\tTraining Loss: {train_loss:.4f}")
        print(f"\tValidation Loss: {val_loss:.4f}")
        print(f"\tValidation Accuracy: {val_accuracy:.4f} ({val_accuracy:.2%})")

        if not args.debug:
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"model_epoch_{epoch + 1}.pth")
            torch.save(model_state, checkpoint_path)
            print(f"\tCheckpoint saved to {checkpoint_path}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
                torch.save(model_state, best_model_path)
                print(f"\t*** New best model saved to {best_model_path} ***")

    print("\n--- Training Complete ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the Language-Guided Localization model.")
    parser.add_argument('--debug', action='store_true', help="Run in debug mode on a small subset of data.")
    args = parser.parse_args()
    main(args)
