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
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.transforms.functional import normalize
from tqdm import tqdm
import numpy as np
import math
import argparse
from project_config import config
from models import LocalizationFramework
from dataset import create_dataloaders


class EvidentialLoss(nn.Module):
    """
    Evidential Loss for Uncertainty Quantification.
    """

    def __init__(self, regularizer_weight=config.TRAIN.EVIDENTIAL_LAMBDA):
        super().__init__()
        self.regularizer_weight = regularizer_weight

    def forward(self, evidential_output, target):

        target = target.unsqueeze(-1)

        evidence = F.softplus(evidential_output)
        alpha = evidence[..., 0:1] + 1
        beta = evidence[..., 1:2] + 1

        S = alpha + beta

        log_likelihood_positive = torch.log(S) - torch.log(alpha)
        log_likelihood_negative = torch.log(S) - torch.log(beta)

        # This multiplication now works correctly
        loss = target * log_likelihood_positive + (1 - target) * log_likelihood_negative

        regularizer = (2.0 + alpha + beta) / S
        loss += self.regularizer_weight * (target - (alpha / S)).abs() * regularizer

        # Return three values to match the signature of the other loss functions
        total_loss = loss.mean()
        return total_loss, total_loss, torch.tensor(0.0)

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
        refined_scores_squeezed = refined_scores.squeeze(-1)
        raw_relevance_scores_squeezed = raw_relevance_scores.squeeze(-1)

        primary_loss_refined = self.relevance_loss(refined_scores_squeezed, ground_truth_relevance)
        primary_loss_raw = self.relevance_loss(raw_relevance_scores_squeezed, ground_truth_relevance)

        # Combine the two primary loss components.
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


class BilevelConsistencyLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.relevance_loss = nn.BCEWithLogitsLoss()
        self.consistency_loss = nn.L1Loss()

        self.optical_flow_model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(config.TRAIN.DEVICE)
        self.optical_flow_model.eval()

    def warp_features(self, features, flow):
        """Warps features from frame t to t+1 using the optical flow field."""
        B, T, H_feat, W_feat, C = features.shape  # Note: H_feat and W_feat are feature map dimensions (e.g., 14x14)

        # === THIS IS THE FIX: Downsample the flow field ===
        # The flow has shape [B*(T-1), 2, H_img, W_img]. We need to downsample it to [B*(T-1), 2, H_feat, W_feat]
        downsampled_flow = F.interpolate(flow, size=(H_feat, W_feat), mode='bilinear', align_corners=False)

        # Scale the flow vectors to match the new, smaller resolution
        scale_factor_h = H_feat / flow.shape[2]
        scale_factor_w = W_feat / flow.shape[3]
        downsampled_flow[:, 0, :, :] *= scale_factor_w
        downsampled_flow[:, 1, :, :] *= scale_factor_h

        # 1. Create the base grid of pixel coordinates for the FEATURE MAP
        grid_y, grid_x = torch.meshgrid(torch.arange(H_feat, device=flow.device),
                                        torch.arange(W_feat, device=flow.device), indexing="ij")
        grid = torch.stack((grid_x, grid_y), 2).float()  # Shape: [H_feat, W_feat, 2]

        # 2. Expand the grid to match the batch size
        grid = grid.unsqueeze(0).expand(B * (T - 1), -1, -1, -1)  # Shape: [B*(T-1), H_feat, W_feat, 2]

        # 3. Permute the downsampled flow to match grid dimensions
        downsampled_flow = downsampled_flow.permute(0, 2, 3, 1)  # Shape: [B*(T-1), H_feat, W_feat, 2]

        # 4. Add the grid and flow to get the new sampling coordinates. This will now work.
        new_grid = grid + downsampled_flow

        # 5. Normalize grid values for grid_sample
        new_grid[..., 0] = 2.0 * new_grid[..., 0] / max(W_feat - 1, 1) - 1.0
        new_grid[..., 1] = 2.0 * new_grid[..., 1] / max(H_feat - 1, 1) - 1.0

        # 6. Reshape features and warp them
        features_to_warp = features[:, :-1].reshape(B * (T - 1), H_feat, W_feat, C).permute(0, 3, 1, 2)
        warped_features = F.grid_sample(features_to_warp, new_grid, padding_mode='border', align_corners=True)

        # 7. Reshape back to the original format
        warped_features = warped_features.permute(0, 2, 3, 1).reshape(B, T - 1, H_feat, W_feat, C)
        return warped_features

    def forward(self, model_outputs, video_clip, ground_truth_relevance):
        refined_scores, raw_scores, _, semantic_features, spatial_features = model_outputs

        primary_loss = self.relevance_loss(raw_scores.squeeze(-1), ground_truth_relevance)

        semantic_loss = torch.tensor(0.0, device=refined_scores.device)
        if self.config.TRAIN.SEMANTIC_LOSS_WEIGHT > 0 and semantic_features.shape[1] > 1:
            semantic_loss = self.consistency_loss(semantic_features[:, 1:], semantic_features[:, :-1])

        flow_loss = torch.tensor(0.0, device=refined_scores.device)
        if self.config.TRAIN.OPTICAL_FLOW_LOSS_WEIGHT > 0 and video_clip.shape[2] > 1:
            with torch.no_grad():
                B, C, T, H, W = video_clip.shape
                video_permuted = video_clip.permute(0, 2, 1, 3, 4)
                video_t = video_permuted[:, :-1].reshape(-1, C, H, W)
                video_t_plus_1 = video_permuted[:, 1:].reshape(-1, C, H, W)

                video_t = normalize(video_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                video_t_plus_1 = normalize(video_t_plus_1, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

                flow = self.optical_flow_model(video_t, video_t_plus_1)[-1]

            warped_spatial_features = self.warp_features(spatial_features, flow)
            actual_next_features = spatial_features[:, 1:]
            flow_loss = self.consistency_loss(warped_spatial_features, actual_next_features)

        total_loss = (primary_loss +
                      self.config.TRAIN.SEMANTIC_LOSS_WEIGHT * semantic_loss +
                      self.config.TRAIN.OPTICAL_FLOW_LOSS_WEIGHT * flow_loss)

        return total_loss, primary_loss, semantic_loss + flow_loss

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=1, last_epoch=-1):
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

        # --- THIS IS THE FIX ---
        if config.MODEL.USE_UNCERTAINTY:
            evidential_output, _, _, _, _, _ = outputs
            loss, _, _ = criterion(evidential_output, labels)
        elif config.TRAIN.USE_BILEVEL_CONSISTENCY:
            # The bilevel loss function expects the entire outputs tuple
            loss, _, _ = criterion(outputs, video_clip, labels)
        else:
            # The baseline DualObjectiveLoss only needs the first two tensors
            refined_scores, raw_scores, _, _, _ = outputs
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

            # --- THIS IS THE FIX ---
            if config.MODEL.USE_UNCERTAINTY:
                evidential_output, refined_scores, _, _, _, _ = outputs
                loss, _, _ = criterion(evidential_output, labels)
                preds = refined_scores.squeeze(-1) > 0.5
            elif config.TRAIN.USE_BILEVEL_CONSISTENCY:
                refined_scores, _, _, _, _ = outputs
                loss, _, _ = criterion(outputs, video_clip, labels)
                preds = torch.sigmoid(refined_scores.squeeze(-1)) > 0.5
            else:
                refined_scores, raw_scores, _, _, _ = outputs
                loss, _, _ = criterion(refined_scores, raw_scores, labels)
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
    elif config.TRAIN.USE_BILEVEL_CONSISTENCY:
        print("\n=== Training with BILEVEL CONSISTENCY Loss ===")
        criterion = BilevelConsistencyLoss(config)
    else:
        print("\n=== Training with Dual Objective Loss ===")
        criterion = DualObjectiveLoss()

    best_val_loss = float('inf')
    print("\n--- Beginning Training and Validation Epochs ---")

    # This training loop is correct and does not need to be changed.
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
