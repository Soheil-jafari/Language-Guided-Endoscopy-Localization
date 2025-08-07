import os
import math
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast

from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.transforms.functional import normalize

from project_config import config
from models import LocalizationFramework
from dataset import create_dataloaders

# Set environment variables for performance
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


class EvidentialLoss(nn.Module):
    """
    Calculates the evidential loss for a single output. It computes the Negative
    Log-Likelihood and a KL divergence regularizer term.
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
        loss_nll = target * (torch.log(S) - torch.log(alpha)) + \
                   (1 - target) * (torch.log(S) - torch.log(beta))
        regularizer = (2.0 + alpha + beta) / S
        loss_kl_reg = (target - (alpha / S)).abs() * regularizer
        return (loss_nll + self.regularizer_weight * loss_kl_reg).mean()


class MasterLoss(nn.Module):
    """
    A definitive, unified loss function that safely unpacks all potential model
    outputs and correctly applies the appropriate loss based on the configuration,
    guaranteeing dual supervision in all scenarios.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_uncertainty = config.MODEL.USE_UNCERTAINTY
        self.use_bilevel = config.TRAIN.USE_BILEVEL_CONSISTENCY

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        if self.use_uncertainty:
            self.evidential_loss = EvidentialLoss()
        if self.use_bilevel:
            self.optical_flow_model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(
                config.TRAIN.DEVICE)

            if torch.cuda.device_count() > 1:
                print("Parallelizing RAFT model for Bi-Level Consistency Loss.")
                self.optical_flow_model = nn.DataParallel(self.optical_flow_model)

            self.optical_flow_model.eval()

    def _get_bilevel_consistency_loss(self, semantic_features, spatial_features, video_clip):
        """Calculates the add-on bi-level consistency regularizer."""
        loss_semantic = torch.tensor(0.0, device=video_clip.device)
        if self.config.TRAIN.SEMANTIC_LOSS_WEIGHT > 0 and semantic_features is not None and semantic_features.shape[
            1] > 1:
            loss_semantic = self.l1_loss(semantic_features[:, 1:], semantic_features[:, :-1])

        loss_flow = torch.tensor(0.0, device=video_clip.device)
        if self.config.TRAIN.OPTICAL_FLOW_LOSS_WEIGHT > 0 and spatial_features is not None and video_clip.shape[2] > 1:
            with torch.no_grad(), autocast(enabled=False):
                video_clip_fp32 = video_clip.float()
                B, C, T, H, W = video_clip_fp32.shape

                # Downsample to 50% resolution before flow computation
                video_permuted = video_clip_fp32.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
                video_t = video_permuted[:, :-1].reshape(-1, C, H, W)  # (B*(T-1), C, H, W)
                video_t_plus_1 = video_permuted[:, 1:].reshape(-1, C, H, W)

                # The RAFT model's internal pyramid downsamples by 8. To get a feature map of at least 16x16,
                # the input image must be at least 128x128 (128 / 8 = 16).
                # Our previous 224 * 0.5 = 112px was too small. We now use a safe target size.
                target_raft_size = (128, 128)

                # Downsample frames to the required minimum size for RAFT
                video_t_small = F.interpolate(video_t, size=target_raft_size, mode='bilinear', align_corners=False)
                video_t_plus_1_small = F.interpolate(video_t_plus_1, size=target_raft_size, mode='bilinear',
                                                     align_corners=False)
                video_t_norm = normalize(video_t_small, mean=[0.5] * 3, std=[0.5] * 3)
                video_t_plus_1_norm = normalize(video_t_plus_1_small, mean=[0.5] * 3, std=[0.5] * 3)

                # Compute flow on downsampled frames
                flow_small = self.optical_flow_model(video_t_norm, video_t_plus_1_norm)[-1]

                # Upscale flow to original feature resolution
                feature_res = (H // 16, W // 16)  # For patch size 16
                flow = F.interpolate(flow_small, size=feature_res, mode='bilinear') * (feature_res[0] / target_raft_size[0])

            warped_spatial = self.warp_features(spatial_features, flow)
            actual_next = spatial_features[:, 1:]
            loss_flow = self.l1_loss(warped_spatial, actual_next)

        return (self.config.TRAIN.SEMANTIC_LOSS_WEIGHT * loss_semantic +
                self.config.TRAIN.OPTICAL_FLOW_LOSS_WEIGHT * loss_flow)

    def _get_baseline_temporal_loss(self, refined_scores):
        """Calculates the simple L1 temporal smoothing loss."""
        if refined_scores.shape[1] <= 1:
            return torch.tensor(0.0, device=refined_scores.device)
        temporal_diff = refined_scores[:, 1:] - refined_scores[:, :-1]
        return self.l1_loss(temporal_diff, torch.zeros_like(temporal_diff))

    def forward(self, model_outputs, video_clip, ground_truth_relevance):
        refined_scores, raw_scores, _, semantic_features, spatial_features, evidential_output = model_outputs

        loss_raw = self.bce_loss(raw_scores.squeeze(-1), ground_truth_relevance.float())
        if self.use_uncertainty:
            loss_refined = self.evidential_loss(evidential_output, ground_truth_relevance.float())
        else:
            loss_refined = self.bce_loss(refined_scores.squeeze(-1), ground_truth_relevance.float())
        primary_loss = loss_raw + loss_refined

        temporal_regularizer_loss = torch.tensor(0.0, device=primary_loss.device)
        if self.use_bilevel:
            temporal_regularizer_loss = self._get_bilevel_consistency_loss(semantic_features, spatial_features,
                                                                           video_clip)
        elif self.config.TRAIN.TEMPORAL_LOSS_WEIGHT > 0:
            temporal_regularizer_loss = self.config.TRAIN.TEMPORAL_LOSS_WEIGHT * self._get_baseline_temporal_loss(
                refined_scores)

        total_loss = primary_loss + temporal_regularizer_loss

        return total_loss, primary_loss, temporal_regularizer_loss

    def warp_features(self, features, flow):
        B, T, H_feat, W_feat, C_feat = features.shape
        features_to_warp = features[:, :-1].reshape(B * (T - 1), H_feat, W_feat, C_feat).permute(0, 3, 1, 2)
        downsampled_flow = F.interpolate(flow.to(features.dtype), size=(H_feat, W_feat), mode='bilinear',
                                         align_corners=False).permute(0, 2, 3, 1)
        grid_y, grid_x = torch.meshgrid(torch.arange(H_feat, device=flow.device),
                                        torch.arange(W_feat, device=flow.device), indexing="ij")
        grid = torch.stack((grid_x, grid_y), 2).float().unsqueeze(0).expand(B * (T - 1), -1, -1, -1)
        new_grid = grid + downsampled_flow
        new_grid[..., 0] = 2.0 * new_grid[..., 0] / max(W_feat - 1, 1) - 1.0
        new_grid[..., 1] = 2.0 * new_grid[..., 1] / max(H_feat - 1, 1) - 1.0
        warped_features_flat = F.grid_sample(features_to_warp, new_grid, padding_mode='border', align_corners=False)
        warped_features = warped_features_flat.permute(0, 2, 3, 1).view(B, T - 1, H_feat, W_feat, C_feat)
        return warped_features


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def train_one_epoch(model, dataloader, optimizer, criterion, scheduler, device):
    """
    Trains the model for one epoch with optional gradient accumulation.
    """
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    scaler = GradScaler()

    # Zero the gradients once before the loop begins.
    optimizer.zero_grad()
    torch.backends.cudnn.benchmark = True

    for i, batch in enumerate(progress_bar):
        video_clip = batch['video_clip'].to(device, non_blocking=True)
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        relevance = batch['labels'].to(device, non_blocking=True)

        # Use autocast for mixed-precision training to save memory and speed up training.
        with autocast():
            with torch.backends.cuda.sdp_kernel(enable_flash=True):  # <-- ADD THIS
                outputs = model(video_clip, input_ids, attention_mask)
            loss, _, _ = criterion(outputs, video_clip, relevance)
            if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                loss = loss.mean()

            # --- GRADIENT ACCUMULATION STEP 1 ---
            # Scale the loss down by the number of accumulation steps.
            # This is crucial because loss.backward() sums gradients. By scaling,
            # we ensure the final accumulated gradient is the average over the steps,
            # not the sum, preventing excessively large updates.
            # If GRADIENT_ACCUMULATION_STEPS is 1, this has no effect.
            loss = loss / config.TRAIN.GRADIENT_ACCUMULATION_STEPS

        # --- GRADIENT ACCUMULATION STEP 2 ---
        # Calculate gradients for the current mini-batch. The scaler handles
        # mixed-precision scaling. These gradients are ADDED to any existing
        # gradients from previous steps in this accumulation cycle.
        scaler.scale(loss).backward()

        # --- GRADIENT ACCUMULATION STEP 3 ---
        # This is the core logic. We only update the model's weights
        # after processing a specified number of batches.
        if (i + 1) % config.TRAIN.GRADIENT_ACCUMULATION_STEPS == 0:
            # Perform the optimizer step to update model weights using the
            # accumulated gradients from the past few steps.
            scaler.step(optimizer)

            # Update the gradient scaler for the next cycle.
            scaler.update()

            # Update the learning rate scheduler.
            scheduler.step()

            # Reset gradients to zero to start a new accumulation cycle.
            optimizer.zero_grad()

        # To log the correct loss, we multiply the scaled loss back up.
        running_loss += loss.item() * config.TRAIN.GRADIENT_ACCUMULATION_STEPS
        progress_bar.set_postfix(loss=f"{running_loss / (i + 1):.4f}")

    return running_loss / len(dataloader)


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss, all_preds, all_labels = 0.0, [], []
    progress_bar = tqdm(dataloader, desc="Validating", unit="batch", leave=False)

    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            video_clip = batch['video_clip'].to(device, non_blocking=True)
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            relevance = batch['labels'].to(device, non_blocking=True)

            with autocast():
                outputs = model(video_clip, input_ids, attention_mask)
                loss, _, _ = criterion(outputs, video_clip, relevance)

            if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                loss = loss.mean()

            refined_scores, _, _, _, _, _ = outputs
            preds = torch.sigmoid(refined_scores.squeeze(-1)) > 0.5

            running_loss += loss.item()
            all_preds.extend(preds.flatten().cpu().numpy())
            all_labels.extend(relevance.flatten().cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    if not all_labels:
        accuracy = 0.0
    else:
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return avg_loss, accuracy


def save_model(model_state, save_path, epoch=None):
    """Robust model saving with verification and error handling"""
    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save model
        torch.save(model_state, save_path)

        # Verify save was successful
        if os.path.exists(save_path) and os.path.getsize(save_path) > 1024:  # 1KB minimum
            msg = f"Successfully saved {'best' if epoch is None else f'epoch {epoch}'} model to {save_path}"
            print(f"\t--- {msg} ---")
            return True
        else:
            print(f"\tXXX ERROR: Saved file is too small or missing at {save_path} XXX")
            return False
    except Exception as e:
        print(f"\tXXX CRITICAL SAVE ERROR: {str(e)} XXX")
        print(f"\tAttempted path: {save_path}")
        return False


def main(args):
    print("--- Starting Training Pipeline ---")
    device = torch.device(config.TRAIN.DEVICE)

    # Enhanced checkpoint directory handling
    checkpoint_dir = os.path.abspath(config.CHECKPOINT_DIR)
    print(f"Checkpoint directory: {checkpoint_dir}")

    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Created checkpoint directory: {checkpoint_dir}")
        # Test write access
        test_file = os.path.join(checkpoint_dir, "write_test.txt")
        with open(test_file, "w") as f:
            f.write("Write test successful")
        os.remove(test_file)
        print("Verified write access to checkpoint directory")
    except Exception as e:
        print(f"XXX CRITICAL DIRECTORY ERROR: {str(e)} XXX")
        print(f"Failed to create/access checkpoint directory at {checkpoint_dir}")
        print("Please check permissions or specify a different location in project_config.py")
        return

    model = LocalizationFramework(config=config).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    tokenizer_for_dataloaders = model.module.text_encoder.tokenizer if isinstance(model,
                                                                                  nn.DataParallel) else model.text_encoder.tokenizer

    if args.debug:
        print("--- RUNNING IN DEBUG MODE ---")
        train_csv = os.path.join(config.OUTPUT_TRIPLETS_DIR, "cholec80_train_triplets_DEBUG.csv")
        val_csv = os.path.join(config.OUTPUT_TRIPLETS_DIR, "cholec80_val_triplets_DEBUG.csv")
        epochs = 3
        current_subset_ratio = 1.0
    else:
        train_csv = config.TRAIN_TRIPLETS_CSV_PATH
        val_csv = config.VAL_TRIPLETS_CSV_PATH
        epochs = config.TRAIN.NUM_EPOCHS
        current_subset_ratio = 0.3

    train_loader, val_loader = create_dataloaders(
        train_csv_path=train_csv,
        val_csv_path=val_csv,
        tokenizer=tokenizer_for_dataloaders,
        clip_length=config.DATA.CLIP_LENGTH,
        subset_ratio=current_subset_ratio
    )

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.TRAIN.LEARNING_RATE,
                            weight_decay=config.TRAIN.WEIGHT_DECAY)

    total_training_steps = len(train_loader) // config.TRAIN.GRADIENT_ACCUMULATION_STEPS * epochs
    num_warmup_steps = len(train_loader) // config.TRAIN.GRADIENT_ACCUMULATION_STEPS * config.TRAIN.WARMUP_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_training_steps)

    criterion = MasterLoss(config)
    print(f"\n=== Training with MasterLoss ===")
    print(f"Uncertainty Mode: {criterion.use_uncertainty}")
    print(f"Bilevel Consistency Mode: {criterion.use_bilevel}")
    best_val_loss = float('inf')
    print("\n--- Beginning Training and Validation Epochs ---")

    for epoch in range(epochs):
        print(f"\n===== Epoch {epoch + 1}/{epochs} =====")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, device)
        val_loss, val_accuracy = validate_one_epoch(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"\tTraining Loss: {train_loss:.4f}")
        print(f"\tValidation Loss: {val_loss:.4f}")
        print(f"\tValidation Accuracy: {val_accuracy:.4f} ({val_accuracy:.2%})")
        print(f"\tCurrent Learning Rate: {current_lr:.6f}")

        model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

        # Save latest model with verification
        latest_model_path = os.path.join(checkpoint_dir, f"latest_model_epoch_{epoch + 1}.pth")
        save_success = save_model(model_state, latest_model_path, epoch=epoch + 1)

        # Save best model if validation loss improves
        if math.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            print(f"\t*** New best model found! Validation loss: {val_loss:.4f} ***")
            if save_model(model_state, best_model_path):
                print(f"\t--- New best model saved ---")
            else:
                print(f"\tXXX Failed to save best model XXX")

    print("\n--- Training Complete ---")
    print(f"Final models saved in: {checkpoint_dir}")
    print("Use: `ls -lh \"{}\"` to verify files".format(checkpoint_dir.replace('"', '\\"')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the Language-Guided Localization model.")
    parser.add_argument('--debug', action='store_true', help="Run in debug mode on a small subset of data.")
    args = parser.parse_args()
    main(args)