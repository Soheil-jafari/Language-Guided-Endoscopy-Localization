import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch.cuda.amp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import numpy as np
import math
import argparse
from project_config import config  # Keep this line - this is the instance you need
from models import LocalizationFramework
from dataset import create_dataloaders  # This will now use the clip-based dataset


class DualObjectiveLoss(nn.Module):
    # Access temporal_weight from the config object
    def __init__(self, temporal_weight=config.TRAIN.TEMPORAL_LOSS_WEIGHT):
        super().__init__()
        self.relevance_loss = nn.BCEWithLogitsLoss()
        self.temporal_weight = temporal_weight

    def forward(self, refined_scores, raw_relevance_scores, ground_truth_relevance):
        """
        Calculates a dual objective loss combining BCE on both refined and raw scores,
        plus an optional temporal consistency loss.

        Args:
            refined_scores (torch.Tensor): Output from the TemporalHead (B, T, 1).
            raw_relevance_scores (torch.Tensor): Output from the LanguageGuidedHead (B, T, 1).
            ground_truth_relevance (torch.Tensor): Ground truth relevance labels (B, T).

        Returns:
            tuple: (total_loss, primary_loss, temporal_loss)
        """
        # Ensure scores are squeezed to (B, T) to match ground_truth_relevance
        # BCEWithLogitsLoss expects inputs and targets of the same shape.
        refined_scores_squeezed = refined_scores.squeeze(-1)
        raw_relevance_scores_squeezed = raw_relevance_scores.squeeze(-1)

        primary_loss_refined = self.relevance_loss(refined_scores_squeezed, ground_truth_relevance)
        primary_loss_raw = self.relevance_loss(raw_relevance_scores_squeezed, ground_truth_relevance)

        # Combine the two primary loss components.
        # You can sum them or average them. Summing gives equal weighting by default.
        primary_loss = primary_loss_refined + primary_loss_raw

        temporal_loss = 0.0
        # Calculate temporal consistency loss if weight > 0 and clip has more than one frame
        if self.temporal_weight > 0 and refined_scores_squeezed.shape[1] > 1:
            # Calculate absolute difference between consecutive refined scores
            scores_t = refined_scores_squeezed[:, 1:]
            scores_t_minus_1 = refined_scores_squeezed[:, :-1]
            temporal_loss = torch.mean(torch.abs(scores_t - scores_t_minus_1))

        # Total loss is the sum of primary and weighted temporal loss
        total_loss = primary_loss + (self.temporal_weight * temporal_loss)

        return total_loss, primary_loss, temporal_loss


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """
    Create a schedule with a learning rate that linearly increases during
    `num_warmup_steps` and then decreases following a cosine curve.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", unit="batch", leave=False)

    scaler = torch.cuda.amp.GradScaler()  # Initialize GradScaler for AMP

    for video_clip, input_ids, attention_mask, relevance in progress_bar:
        # Move all tensors to the GPU
        video_clip, input_ids, attention_mask, relevance = video_clip.to(device), input_ids.to(
            device), attention_mask.to(device), relevance.to(device)

        optimizer.zero_grad()

        # Wrap the forward pass with autocast for mixed precision
        with torch.cuda.amp.autocast():
            # IMPORTANT: The model's forward method now returns 3 values.
            # It's refined_scores, raw_relevance_scores, and attention_weights_for_xai.
            refined_scores, raw_relevance_scores, _ = model(video_clip, input_ids, attention_mask)

            # Squeeze the last dimension of scores to match relevance shape [B, T]
            # Pass raw_relevance_scores to criterion's forward method
            # Adjust this if your criterion expects the raw_relevance_scores to be different.
            loss, _, _ = criterion(refined_scores.squeeze(-1), raw_relevance_scores.squeeze(-1), relevance)

            if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                loss = loss.mean()

        # Perform backward pass with scaler
        scaler.scale(loss).backward()

        # Optimizer step with scaler
        scaler.step(optimizer)

        # Update the scaler for the next iteration
        scaler.update()

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

            # IMPORTANT: The model's forward method now returns 3 values.
            refined_scores, raw_relevance_scores, _ = model(video_clip, input_ids, attention_mask)
            # Pass raw_relevance_scores to criterion's forward method
            loss, _, _ = criterion(refined_scores.squeeze(-1), raw_relevance_scores.squeeze(-1), relevance)

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
    print("--- Starting Training Pipeline ---")
    # Access device from config.TRAIN
    device = torch.device(config.TRAIN.DEVICE)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # IMPORTANT: Pass the config object to LocalizationFramework constructor
    model = LocalizationFramework(config=config).to(device)  # Keep this as 'config'
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    if args.debug:
        print("--- RUNNING IN DEBUG MODE ---")
        train_csv = config.TRAIN_TRIPLETS_CSV_PATH.replace(".csv", "_DEBUG.csv")
        val_csv = config.VAL_TRIPLETS_CSV_PATH.replace(".csv", "_DEBUG.csv")
        epochs = 1
    else:
        # Access these paths and epochs directly from the 'config' object
        train_csv = config.TRAIN_TRIPLETS_CSV_PATH  # CHANGED from project_config.TRAIN_TRIPLETS_CSV_PATH
        val_csv = config.VAL_TRIPLETS_CSV_PATH
        epochs = config.TRAIN.NUM_EPOCHS  # CHANGED from project_config.TRAIN.NUM_EPOCHS

    train_loader, val_loader = create_dataloaders(
        train_csv_path=train_csv,
        val_csv_path=val_csv,
        # Pass clip_length from config.DATA
        clip_length=config.DATA.CLIP_LENGTH
    )

    # Access learning_rate from config.TRAIN
    # CHANGED from project_config.TRAIN.LEARNING_RATE
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.TRAIN.LEARNING_RATE,
        weight_decay=config.TRAIN.WEIGHT_DECAY
    )
    # --- Learning Rate Scheduler Setup ---
    num_training_epochs = config.TRAIN.NUM_EPOCHS
    num_warmup_epochs = config.TRAIN.WARMUP_EPOCHS

    if num_warmup_epochs >= num_training_epochs:
        print(f"Warning: Warmup epochs ({num_warmup_epochs}) is >= total epochs ({num_training_epochs}).")
        print(f"Setting warmup to 10% of total epochs for effective cosine decay.")
        num_warmup_epochs = max(1, int(num_training_epochs * 0.1))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_epochs,
        num_training_steps=num_training_epochs
    )
    print(
        f"Learning rate scheduler initialized: Warmup for {num_warmup_epochs} epochs, then cosine decay over {num_training_epochs} epochs.")

    criterion = DualObjectiveLoss()  # temporal_weight is now fetched from config inside DualObjectiveLoss __init__

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

        scheduler.step()
        print(f"\tLearning rate for next epoch: {optimizer.param_groups[0]['lr']:.6f}")

    print("\n--- Training Complete ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the Language-Guided Localization model.")
    parser.add_argument('--debug', action='store_true', help="Run in debug mode on a small subset of data.")
    args = parser.parse_args()
    main(args)
