import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch.cuda.amp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import math
import argparse
from project_config import config  # Keep this line - this is the instance you need
from models import LocalizationFramework
from dataset import create_dataloaders  # This will now use the clip-based dataset


class EvidentialLoss(nn.Module):
    """
    Evidential Loss for Uncertainty Quantification.
    Modified to return three values (total_loss, primary_loss, reg_loss)
    to match the signature of the original DualObjectiveLoss.
    """

    def __init__(self, regularizer_weight=config.TRAIN.EVIDENTIAL_LAMBDA):
        super().__init__()
        self.regularizer_weight = regularizer_weight

    def forward(self, evidential_output, target):
        evidence = F.softplus(evidential_output)
        alpha = evidence[..., 2:3] + 1
        beta = evidence[..., 3:4] + 1
        S = alpha + beta

        log_likelihood_positive = torch.log(S) - torch.log(alpha)
        log_likelihood_negative = torch.log(S) - torch.log(beta)

        primary_loss = target * log_likelihood_positive + (1 - target) * log_likelihood_negative

        regularizer_term = (2.0 + alpha + beta) / S
        reg_loss = (target - (alpha / S)).abs() * regularizer_term

        total_loss = (primary_loss + self.regularizer_weight * reg_loss).mean()

        # Return three values to match the expected unpacking in the training loop
        return total_loss, primary_loss.mean(), reg_loss.mean()

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


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss_tracker = 0.0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress_bar:
        video_clip = batch['video_clip'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device).float()

        optimizer.zero_grad()
        outputs = model(video_clip, input_ids, attention_mask)

        if config.MODEL.USE_UNCERTAINTY:
            evidential_output, _, _, _ = outputs
            loss, _, _ = criterion(evidential_output, labels)
        else:
            refined_scores, raw_scores, _ = outputs
            loss, _, _ = criterion(refined_scores, raw_scores, labels)

        loss.backward()
        optimizer.step()
        total_loss_tracker += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss_tracker / len(dataloader)


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss, all_preds, all_labels = 0.0, [], []
    progress_bar = tqdm(dataloader, desc="Validating", unit="batch", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            video_clip = batch['video_clip'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device).float()

            outputs = model(video_clip, input_ids, attention_mask)

            if config.MODEL.USE_UNCERTAINTY:
                evidential_output, refined_scores, _, _ = outputs
                loss, _, _ = criterion(evidential_output, labels)
                # Predictions are the refined scores calculated from alpha/beta in the model
                preds = refined_scores.squeeze(-1) > 0.5
            else:
                refined_scores, raw_scores, _ = outputs
                loss, _, _ = criterion(refined_scores, raw_scores, labels)
                # Predictions are based on sigmoid of refined scores
                preds = torch.sigmoid(refined_scores.squeeze(-1)) > 0.5

            if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                loss = loss.mean()

            running_loss += loss.item()
            all_preds.extend(preds.flatten().cpu().numpy())
            all_labels.extend(labels.flatten().cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return avg_loss, accuracy


def main(args):
    print("--- Starting Training Pipeline ---")
    device = torch.device(config.TRAIN.DEVICE)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    model = LocalizationFramework(config=config).to(device)
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
        epochs = config.TRAIN.NUM_EPOCHS

    train_loader, val_loader = create_dataloaders(
        train_csv_path=train_csv,
        val_csv_path=val_csv,
        clip_length=config.DATA.CLIP_LENGTH,
        batch_size=config.TRAIN.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS
    )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.TRAIN.LEARNING_RATE,
        weight_decay=config.TRAIN.WEIGHT_DECAY
    )

    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = len(train_loader) * config.TRAIN.WARMUP_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    if config.MODEL.USE_UNCERTAINTY:
        print("\n=== Training with UNCERTAINTY (Evidential Loss) ===")
        criterion = EvidentialLoss()
    else:
        print("\n=== Training without uncertainty (Dual Objective Loss) ===")
        criterion = DualObjectiveLoss()

    best_val_loss = float('inf')
    print("\n--- Beginning Training and Validation Epochs ---")

    # Your training loop was already correct and does not need to be changed.
    for epoch in range(epochs):
        print(f"\n===== Epoch {epoch + 1}/{epochs} =====")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = validate_one_epoch(model, val_loader, criterion, device)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"\tTraining Loss: {train_loss:.4f}")
        print(f"\tValidation Loss: {val_loss:.4f}")
        print(f"\tValidation Accuracy: {val_accuracy:.4f} ({val_accuracy:.2%})")

        if not args.debug:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
                model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
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