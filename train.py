import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from torch.cuda.amp import GradScaler, autocast
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

        loss = target * log_likelihood_positive + (1 - target) * log_likelihood_negative

        regularizer = (2.0 + alpha + beta) / S
        loss += self.regularizer_weight * (target - (alpha / S)).abs() * regularizer

        total_loss = loss.mean()
        return total_loss, total_loss, torch.tensor(0.0) # Return three values to match signature


class MasterLoss(nn.Module):
    """
    A unified loss function that provides full, independent control over
    using evidential loss for uncertainty and bilevel consistency for regularization.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_uncertainty = config.MODEL.USE_UNCERTAINTY
        self.use_bilevel_consistency = config.TRAIN.USE_BILEVEL_CONSISTENCY

        # Initialize all potential loss components
        self.bce_loss = nn.BCEWithLogitsLoss()
        if self.use_uncertainty:
            self.evidential_loss = EvidentialLoss()

        if self.use_bilevel_consistency:
            self.consistency_loss_l1 = nn.L1Loss()
            self.optical_flow_model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(
                config.TRAIN.DEVICE)
            self.optical_flow_model.eval()

    def forward(self, model_outputs, video_clip, ground_truth_relevance):
        # --- 1. Primary Relevance Loss Calculation ---
        primary_loss = torch.tensor(0.0, device=self.config.TRAIN.DEVICE)

        if self.use_uncertainty:
            evidential_output, *_ = model_outputs
            primary_loss, _, _ = self.evidential_loss(evidential_output, ground_truth_relevance)
        else:
            refined_scores, raw_scores, *_ = model_outputs
            loss_refined = self.bce_loss(refined_scores.squeeze(-1), ground_truth_relevance)
            loss_raw = self.bce_loss(raw_scores.squeeze(-1), ground_truth_relevance)
            primary_loss = loss_refined + loss_raw

        # --- 2. Consistency Regularization Calculation ---
        consistency_loss = torch.tensor(0.0, device=self.config.TRAIN.DEVICE)

        if self.use_bilevel_consistency:
            _, _, _, semantic_features, spatial_features, _ = model_outputs

            if semantic_features is not None and spatial_features is not None:
                # Semantic Consistency
                semantic_loss = torch.tensor(0.0, device=primary_loss.device)
                if self.config.TRAIN.SEMANTIC_LOSS_WEIGHT > 0 and semantic_features.shape[1] > 1:
                    semantic_loss = self.consistency_loss_l1(semantic_features[:, 1:], semantic_features[:, :-1])

                # Optical Flow Consistency
                flow_loss = torch.tensor(0.0, device=primary_loss.device)
                if self.config.TRAIN.OPTICAL_FLOW_LOSS_WEIGHT > 0 and video_clip.shape[2] > 1:
                    with torch.no_grad():
                        B, C, T, H, W = video_clip.shape
                        video_permuted = video_clip.permute(0, 2, 1, 3, 4)

                        # === THIS IS THE FINAL, CORRECT FIX ===
                        # The reshape operation on a permuted tensor can create a non-contiguous tensor.
                        # We force it to be contiguous right before it's used.
                        video_t = video_permuted[:, :-1].reshape(-1, C, H, W).contiguous()
                        video_t_plus_1 = video_permuted[:, 1:].reshape(-1, C, H, W).contiguous()

                        flow = self.optical_flow_model(normalize(video_t, [0.5] * 3, [0.5] * 3),
                                                       normalize(video_t_plus_1, [0.5] * 3, [0.5] * 3))[-1]

                    warped_spatial = self.warp_features(spatial_features, flow)
                    actual_next = spatial_features[:, 1:]
                    flow_loss = self.consistency_loss_l1(warped_spatial, actual_next)

                consistency_loss = (self.config.TRAIN.SEMANTIC_LOSS_WEIGHT * semantic_loss +
                                    self.config.TRAIN.OPTICAL_FLOW_LOSS_WEIGHT * flow_loss)
            else:
                print(
                    "\nWARNING: Bilevel consistency is ON, but model did not return consistency features. Skipping consistency loss for this batch.\n",
                    flush=True)

        # --- 3. Final Loss Combination ---
        total_loss = primary_loss + consistency_loss
        return total_loss, primary_loss, consistency_loss

    def warp_features(self, features, flow):
        # This helper function is correct and needs no changes.
        B, T, H_feat, W_feat, C = features.shape
        downsampled_flow = F.interpolate(flow, size=(H_feat, W_feat), mode='bilinear', align_corners=False)
        scale_factor_h, scale_factor_w = H_feat / flow.shape[2], W_feat / flow.shape[3]
        downsampled_flow[:, 0, :, :] *= scale_factor_w
        downsampled_flow[:, 1, :, :] *= scale_factor_h
        grid_y, grid_x = torch.meshgrid(torch.arange(H_feat, device=flow.device),
                                        torch.arange(W_feat, device=flow.device), indexing="ij")
        grid = torch.stack((grid_x, grid_y), 2).float().unsqueeze(0).expand(B * (T - 1), -1, -1, -1)
        new_grid = grid + downsampled_flow.permute(0, 2, 3, 1)
        new_grid[..., 0] = 2.0 * new_grid[..., 0] / max(W_feat - 1, 1) - 1.0
        new_grid[..., 1] = 2.0 * new_grid[..., 1] / max(H_feat - 1, 1) - 1.0
        features_to_warp = features[:, :-1].reshape(B * (T - 1), H_feat, W_feat, C).permute(0, 3, 1, 2)
        warped_features = F.grid_sample(features_to_warp, new_grid, padding_mode='border', align_corners=True)
        return warped_features.permute(0, 2, 3, 1).reshape(B, T - 1, H_feat, W_feat, C)

    def warp_features(self, features, flow):
        # Helper function for optical flow - no changes needed
        B, T, H_feat, W_feat, C = features.shape
        downsampled_flow = F.interpolate(flow, size=(H_feat, W_feat), mode='bilinear', align_corners=False)
        scale_factor_h, scale_factor_w = H_feat / flow.shape[2], W_feat / flow.shape[3]
        downsampled_flow[:, 0, :, :] *= scale_factor_w
        downsampled_flow[:, 1, :, :] *= scale_factor_h
        grid_y, grid_x = torch.meshgrid(torch.arange(H_feat, device=flow.device),
                                        torch.arange(W_feat, device=flow.device), indexing="ij")
        grid = torch.stack((grid_x, grid_y), 2).float().unsqueeze(0).expand(B * (T - 1), -1, -1, -1)
        new_grid = grid + downsampled_flow.permute(0, 2, 3, 1)
        new_grid[..., 0] = 2.0 * new_grid[..., 0] / max(W_feat - 1, 1) - 1.0
        new_grid[..., 1] = 2.0 * new_grid[..., 1] / max(H_feat - 1, 1) - 1.0
        features_to_warp = features[:, :-1].reshape(B * (T - 1), H_feat, W_feat, C).permute(0, 3, 1, 2)
        warped_features = F.grid_sample(features_to_warp, new_grid, padding_mode='border', align_corners=True)
        return warped_features.permute(0, 2, 3, 1).reshape(B, T - 1, H_feat, W_feat, C)

# Corrected get_cosine_schedule_with_warmup: num_cycles=1.0
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=1.0, last_epoch=-1):
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


# Corrected train_one_epoch signature for optimizer and criterion order
def train_one_epoch(model, dataloader, optimizer, criterion, scheduler, device):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    # Correctly initialized GradScaler
    scaler = GradScaler()

    for i, batch in enumerate(progress_bar):
        # === THIS IS THE FIX: Ensure tensor is in contiguous memory ===
        video_clip = batch['video_clip'].to(device).contiguous()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        relevance = batch['labels'].to(device)

        # Correctly uses the imported autocast
        with autocast():
            outputs = model(video_clip, input_ids, attention_mask)
            loss, _, _ = criterion(outputs, video_clip, relevance)
            if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                loss = loss.mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        progress_bar.set_postfix(loss=f"{running_loss / (i + 1):.4f}")

    return running_loss / len(dataloader)

# Corrected validate_one_epoch signature for optimizer and criterion order
def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss, all_preds, all_labels = 0.0, [], []
    progress_bar = tqdm(dataloader, desc="Validating", unit="batch", leave=False)

    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            # === THIS IS THE FIX: Ensure tensor is in contiguous memory ===
            video_clip = batch['video_clip'].to(device).contiguous()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            relevance = batch['labels'].to(device)
            outputs = model(video_clip, input_ids, attention_mask)

            # The call to criterion is now simple and direct.
            loss, _, _ = criterion(outputs, video_clip, relevance)
            if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                loss = loss.mean()

            # Prediction logic must now handle both uncertainty and standard modes
            if config.MODEL.USE_UNCERTAINTY:
                _, refined_scores, *_ = outputs
            else:
                refined_scores, *_ = outputs
            preds = torch.sigmoid(refined_scores.squeeze(-1)) > 0.5

            running_loss += loss.item()
            all_preds.extend(preds.flatten().cpu().numpy())
            all_labels.extend(relevance.flatten().cpu().numpy())

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

    # --- Get the tokenizer instance from the model's text encoder ---
    tokenizer_for_dataloaders = model.module.text_encoder.tokenizer if isinstance(model,
                                                                                  nn.DataParallel) else model.text_encoder.tokenizer

    if args.debug:
        print("--- RUNNING IN DEBUG MODE ---")
        train_csv = config.TRAIN_TRIPLETS_CSV_PATH.replace(".csv", "_DEBUG.csv")
        val_csv = config.VAL_TRIPLETS_CSV_PATH.replace(".csv", "_DEBUG.csv")
        epochs = 1
        current_subset_ratio = 0.01
    else:
        train_csv = config.TRAIN_TRIPLETS_CSV_PATH
        val_csv = config.VAL_TRIPLETS_CSV_PATH
        epochs = config.TRAIN.NUM_EPOCHS
        current_subset_ratio = 0.3

    train_loader, val_loader = create_dataloaders(
        train_csv_path=train_csv,
        val_csv_path=val_csv,
        tokenizer=tokenizer_for_dataloaders,  # Pass the tokenizer
        clip_length=config.DATA.CLIP_LENGTH,
        subset_ratio=current_subset_ratio  # Pass the subset ratio
    )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.TRAIN.LEARNING_RATE,
        weight_decay=config.TRAIN.WEIGHT_DECAY
    )

    # Scheduler setup is now per-epoch
    num_training_steps_for_scheduler = epochs # Total epochs for scheduler
    num_warmup_steps_for_scheduler = config.TRAIN.WARMUP_EPOCHS # Warmup epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps_for_scheduler, num_training_steps=num_training_steps_for_scheduler)

    criterion = MasterLoss(config)

    print(f"\n=== Training with MasterLoss ===")
    print(f"Uncertainty Mode: {criterion.use_uncertainty}")
    print(f"Bilevel Consistency Mode: {criterion.use_bilevel_consistency}")

    best_val_loss = float('inf')
    print("\n--- Beginning Training and Validation Epochs ---")

    for epoch in range(epochs):
        print(f"\n===== Epoch {epoch + 1}/{epochs} =====")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, device)
        val_loss, val_accuracy = validate_one_epoch(model, val_loader, criterion, device)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"\tTraining Loss: {train_loss:.4f}")
        print(f"\tValidation Loss: {val_loss:.4f}")
        print(f"\tValidation Accuracy: {val_accuracy:.4f} ({val_accuracy:.2%})")

        # === START: ROBUST CHECKPOINT SAVING LOGIC ===

        # 1. Skip saving if debug mode is on
        if args.debug:
            print("\tDEBUG mode is on. Skipping checkpoint save.")
            scheduler.step()
            continue  # Go to the next epoch

        # 2. Check for invalid validation loss values (NaN or Infinity)
        if not math.isfinite(val_loss):
            print(
                f"\tWARNING: Validation loss is {val_loss}. Cannot save model. Check for exploding gradients or other issues.")
            scheduler.step()
            continue  # Go to the next epoch

        # 3. Compare with the best validation loss so far with explicit logging
        print(f"\tCurrent Best Validation Loss: {best_val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

            print(f"\t*** New best model found! Attempting to save to {best_model_path} ***")

            try:
                # Get the model's state dictionary, handling DataParallel correctly
                model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save(model_state, best_model_path)
                print(f"\t--- Successfully saved new best model. ---")
            except Exception as e:
                print(f"\tXXX ERROR: FAILED TO SAVE MODEL. Reason: {e} XXX")
                print(f"\tPlease check file permissions for the directory: {config.CHECKPOINT_DIR}")
        else:
            print(f"\tValidation loss did not improve. Not saving model.")
        # === END: ROBUST CHECKPOINT SAVING LOGIC ===

        # Step the learning rate scheduler
        scheduler.step()
        print(f"\tLearning rate for next epoch: {optimizer.param_groups[0]['lr']:.6f}")

    print("\n--- Training Complete ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the Language-Guided Localization model.")
    parser.add_argument('--debug', action='store_true', help="Run in debug mode on a small subset of data.")
    args = parser.parse_args()
    main(args)
