# train.py
# The professional-grade training script for the Language-Guided Localization Framework.
# Includes training and validation loops, accuracy metrics, and robust checkpointing.

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

import config
from models import LocalizationFramework
from dataset import create_dataloaders


class DualObjectiveLoss(nn.Module):
    def __init__(self, temporal_weight=config.TEMPORAL_LOSS_WEIGHT):
        super().__init__()
        self.relevance_loss = nn.BCEWithLogitsLoss()
        self.temporal_weight = temporal_weight

    def forward(self, refined_scores, raw_scores, ground_truth_relevance):
        primary_loss = self.relevance_loss(refined_scores.squeeze(-1), ground_truth_relevance)

        temporal_loss = 0.0
        if self.temporal_weight > 0 and refined_scores.shape[1] > 1:
            scores_t = refined_scores[:, 1:, :]
            scores_t_minus_1 = refined_scores[:, :-1, :]
            temporal_loss = torch.mean(torch.abs(scores_t - scores_t_minus_1))

        total_loss = primary_loss + (self.temporal_weight * temporal_loss)
        return total_loss, primary_loss, temporal_loss


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", unit="batch")

    for batch in progress_bar:
        frames, input_ids, attention_mask, relevance = [b.to(device) for b in batch]
        # Our dataset creates single frames, but model expects clips. We unsqueeze to add a time dimension of 1.
        frames = frames.unsqueeze(1)  # [B, 1, C, H, W]
        relevance = relevance.unsqueeze(1)  # [B, 1]

        optimizer.zero_grad()
        raw_scores, refined_scores = model(frames, input_ids, attention_mask)
        loss, _, _ = criterion(refined_scores, raw_scores, relevance)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(dataloader)


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    progress_bar = tqdm(dataloader, desc="Validating", unit="batch")

    with torch.no_grad():
        for batch in progress_bar:
            frames, input_ids, attention_mask, relevance = [b.to(device) for b in batch]
            frames = frames.unsqueeze(1)
            relevance = relevance.unsqueeze(1)

            raw_scores, refined_scores = model(frames, input_ids, attention_mask)
            loss, _, _ = criterion(refined_scores, raw_scores, relevance)

            running_loss += loss.item()

            # For accuracy calculation
            preds = torch.sigmoid(refined_scores.squeeze(-1)) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(relevance.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    return avg_loss, accuracy


def main():
    print("--- Starting Training Pipeline ---")
    device = torch.device(config.DEVICE)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    print("Initializing model...")
    model = LocalizationFramework().to(device)

    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config.TRIPLETS_CSV_PATH)

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = DualObjectiveLoss()

    best_val_loss = float('inf')
    print("\n--- Beginning Training and Validation Epochs ---")

    for epoch in range(config.EPOCHS):
        print(f"\n===== Epoch {epoch + 1}/{config.EPOCHS} =====")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate_one_epoch(model, val_loader, criterion, device)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"\tTraining Loss: {train_loss:.4f}")
        print(f"\tValidation Loss: {val_loss:.4f}")
        print(f"\tValidation Accuracy: {val_accuracy:.4f} ({val_accuracy:.2%})")

        # --- Checkpointing ---
        # Save a checkpoint after every epoch
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"\tCheckpoint saved to {checkpoint_path}")

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"\t*** New best model saved to {best_model_path} ***")

    print("\n--- Training Complete ---")


if __name__ == '__main__':
    if not hasattr(config, 'TRIPLETS_CSV_PATH') or not config.TRIPLETS_CSV_PATH:
        raise ValueError("Please define TRIPLETS_CSV_PATH in config.py.")
    main()
