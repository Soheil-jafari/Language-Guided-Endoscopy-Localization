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


# ===== Robust checkpoint utilities =====
def _strip_module_prefix(state_dict):
    return { (k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items() }

def _add_module_prefix(state_dict):
    return { (k if k.startswith('module.') else f'module.{k}'): v for k, v in state_dict.items() }

def atomic_torch_save(obj, path):
    import tempfile, os, torch
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=d, prefix='.tmp_ckpt_', suffix='.pt')
    os.close(fd)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)  # atomic on POSIX
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
def load_weights_for_finetune(model, checkpoint_path, device):
    import os, torch, builtins
    import torch.serialization as ts

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    print(f"Loading weights for fine-tuning from: {checkpoint_path}")

    # 1) Try safe (tensors-only) load on PyTorch 2.6+
    ckpt = None
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        pass  # older torch

    # 2) If that failed because old ckpt has pickled objects, allowlist and retry
    if ckpt is None:
        try:
            if hasattr(ts, "add_safe_globals"):
                ts.add_safe_globals([getattr, builtins.getattr])
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Could not load checkpoint: {e}")

    # Extract state_dict
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Strip/normalize DP prefix
    def strip_module(d):
        return { (k[7:] if k.startswith("module.") else k): v for k, v in d.items() }
    state_dict = strip_module(state_dict)

    # Filter to matching keys & shapes
    model_sd = model.state_dict()
    filtered = {}
    skipped_bad_shape = []
    skipped_missing_key = []

    for k, v in state_dict.items():
        if k in model_sd:
            if model_sd[k].shape == v.shape:
                filtered[k] = v
            else:
                skipped_bad_shape.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
        else:
            skipped_missing_key.append(k)

    # Load non-strict
    missing, unexpected = model.load_state_dict(filtered, strict=False)

    # Summary
    print(f"✅ Loaded {len(filtered)} tensors into model.")
    if missing:
        print(f"⚠️  Missing in checkpoint but required by model: {len(missing)} (e.g. {missing[:5]})")
    if unexpected:
        print(f"⚠️  Present in checkpoint but not in model: {len(unexpected)} (e.g. {unexpected[:5]})")
    if skipped_bad_shape:
        ex = skipped_bad_shape[:3]
        print(f"⚠️  Skipped {len(skipped_bad_shape)} due to shape mismatch (e.g. {ex})")

    print("➡️  Fine-tuning will proceed with the overlapping weights only.")
    return model

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_loss, path, extra=None):

    # Get model weights (strip DP prefix if present)
    if isinstance(model, nn.DataParallel):
        model_state = { (k[7:] if k.startswith('module.') else k): v.cpu()
                        for k, v in model.state_dict().items() }
    else:
        model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # Keep optimizer/scheduler/scaler optional (useful for exact resume),
    # but they’re tensors only; avoid custom objects.
    ckpt = {
        'epoch': int(epoch),
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'best_val_loss': float(best_val_loss) if best_val_loss is not None else None,
        # only simple, JSON-like values here
        'extra': {
            **(extra or {}),
            'lr': float(getattr(config.TRAIN, 'LEARNING_RATE', 0.0)),
            'wd': float(getattr(config.TRAIN, 'WEIGHT_DECAY', 0.0)),
            'batch_size': int(getattr(config.TRAIN, 'BATCH_SIZE', 0)),
            'subset': float(getattr(config.TRAIN, 'SUBSET_RATIO', 1.0)) if hasattr(config.TRAIN, 'SUBSET_RATIO') else 1.0,
        },
    }

    # Atomic save
    atomic_torch_save(ckpt, path)
    ok = os.path.exists(path) and os.path.getsize(path) > 4096
    print(f"\t--- Saved checkpoint to {path} ({'OK' if ok else 'SMALL/ERROR'}) ---")
    return ok


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, map_location=None, strict=False):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    ckpt = torch.load(path, map_location=map_location)
    state_dict = ckpt.get('model_state_dict', ckpt)
    # Try stripping/adding module prefix to match
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as e:
        try:
            model.load_state_dict(_add_module_prefix(state_dict), strict=strict)
        except Exception:
            model.load_state_dict(_strip_module_prefix(state_dict), strict=strict)
    if optimizer and ckpt.get('optimizer_state_dict'):
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and ckpt.get('scheduler_state_dict'):
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if scaler and ckpt.get('scaler_state_dict'):
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    return ckpt
# ======================================
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
            try:
                with torch.backends.cuda.sdp_kernel(enable_flash=True):
                    outputs = model(video_clip, input_ids, attention_mask)
            except Exception:
                outputs = model(video_clip, input_ids, attention_mask)
            # (flash SDP guarded)
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


def save_model_DEPRECATED(*args, **kwargs):
    raise RuntimeError('save_model is deprecated; use save_checkpoint instead.')

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
    torch.backends.cudnn.benchmark = True
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
        # read from config, allow CLI override
        current_subset_ratio = getattr(config.TRAIN, "SUBSET_RATIO", 1.0)
        if args.subset is not None:
            current_subset_ratio = float(args.subset)
        current_subset_ratio = max(0.0, min(1.0, current_subset_ratio))

    print(f"\tUsing training subset ratio: {current_subset_ratio:.2f}")

    train_loader, val_loader = create_dataloaders(
        train_csv_path=train_csv,
        val_csv_path=val_csv,
        tokenizer=tokenizer_for_dataloaders,
        clip_length=config.DATA.CLIP_LENGTH,
        subset_ratio=current_subset_ratio
    )

    if args.finetune_from:
        model = load_weights_for_finetune(model, args.finetune_from, device)
        print(f"Fine-tuning FROM: {args.finetune_from}")
    else:
        print("Fine-tuning FROM: (none) — training from scratch weights")

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
    print("\n===== Fine-tune / Training Settings =====")
    print(f"Subset ratio (effective): {current_subset_ratio:.2f}")
    print(f"Subset ratio (config):    {getattr(config.TRAIN, 'SUBSET_RATIO', 'N/A')}")
    print(f"Learning rate:         {config.TRAIN.LEARNING_RATE}")
    print(f"Weight decay:          {config.TRAIN.WEIGHT_DECAY}")
    print(f"Batch size:            {config.TRAIN.BATCH_SIZE}")
    print(f"Num epochs:            {config.TRAIN.NUM_EPOCHS}")
    print(f"Gradient accumulation: {config.TRAIN.GRADIENT_ACCUMULATION_STEPS}")
    print(f"Semantic loss weight:  {config.TRAIN.SEMANTIC_LOSS_WEIGHT}")
    print(f"Temporal loss weight:  {config.TRAIN.TEMPORAL_LOSS_WEIGHT}")
    print(f"Optical flow weight:   {config.TRAIN.OPTICAL_FLOW_LOSS_WEIGHT}")
    print(f"Evidential lambda:     {config.TRAIN.EVIDENTIAL_LAMBDA}")
    print("=========================================\n")

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
        # --- Save checkpoints (latest and best) ---
        latest_model_path = os.path.join(checkpoint_dir, f"latest_model_epoch_{epoch + 1}.pth")
        save_checkpoint(model, optimizer, scheduler, None, epoch + 1, best_val_loss, latest_model_path)

        if math.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            print(f"	*** New best model found! Validation loss: {val_loss:.4f} ***")
            save_checkpoint(model, optimizer, scheduler, None, epoch + 1, best_val_loss, best_model_path)
    print("\n--- Training Complete ---")
    print(f"Final models saved in: {checkpoint_dir}")
    print("Use: `ls -lh \"{}\"` to verify files".format(checkpoint_dir.replace('"', '\\"')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the Language-Guided Localization model.")
    parser.add_argument('--debug', action='store_true', help="Run in debug mode on a small subset of data.")
    parser.add_argument('--subset', type=float, default=None,
                        help="Fraction of TRAIN set to use (0..1). Overrides config if provided.")
    parser.add_argument('--finetune_from', type=str, default=None,
                        help="Path to a checkpoint to load ONLY the weights from (no optimizer/scheduler).")
    args = parser.parse_args()
    main(args)