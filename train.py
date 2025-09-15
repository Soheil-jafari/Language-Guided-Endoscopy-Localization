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
from torchvision.transforms.functional import normalize
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, average_precision_score

def _decode_segments(binary_vec):
    segs = []
    s = None
    for i, v in enumerate(binary_vec):
        if v and s is None:
            s = i
        if (not v or i == len(binary_vec)-1) and s is not None:
            e = i if v else i-1
            segs.append((s, e))  # inclusive frame indices
            s = None
    return segs

def _iou_1d(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = (a[1]-a[0]+1) + (b[1]-b[0]+1) - inter
    return inter / union if union > 0 else 0.0

def _tiou_prf1(pred_segs, gt_segs, thr=0.5):
    used_gt = set()
    tp = 0
    for ps in pred_segs:
        best, best_i = -1, -1
        for i, gs in enumerate(gt_segs):
            if i in used_gt:
                continue
            v = _iou_1d(ps, gs)
            if v > best:
                best, best_i = v, i
        if best >= thr:
            tp += 1
            used_gt.add(best_i)
    fp = max(0, len(pred_segs) - tp)
    fn = max(0, len(gt_segs) - tp)
    prec = tp / (tp + fp) if (tp+fp) else 0.0
    rec  = tp / (tp + fn) if (tp+fn) else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    return prec, rec, f1

def best_f1_threshold_from_logits(logits_1d, labels_1d):
    """
    logits_1d: numpy array of raw logits (not sigmoid), shape (N,)
    labels_1d: numpy array of {0,1}, shape (N,)
    Returns: (thr*, bestF1)
    """
    p = 1.0 / (1.0 + np.exp(-np.asarray(logits_1d)))
    y = np.asarray(labels_1d).astype(int)
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 19):
        pred = (p >= t)
        tp = (pred & (y == 1)).sum()
        fp = (pred & (y == 0)).sum()
        fn = ((~pred) & (y == 1)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t, best_f1

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

    # 1) Here we Try safe (tensors-only) load on PyTorch 2.6+
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
        evidence = evidential_output
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

        pos_weight_value = 1.0
        pos_weight_tensor = torch.tensor([pos_weight_value], device=config.TRAIN.DEVICE)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        self.l1_loss = nn.L1Loss()
        if self.use_uncertainty:
            self.evidential_loss = EvidentialLoss()
        if self.use_bilevel:
            try:
                weights = Raft_Small_Weights.C_T_SKHT_KITTI_V2
            except Exception:
                try:
                    weights = Raft_Small_Weights.DEFAULT
                except Exception:
                    weights = None  # last resort, uninitialized

            self.optical_flow_model = raft_small(weights=weights, progress=False).to(config.TRAIN.DEVICE)

            # IMPORTANT: do NOT wrap RAFT in DataParallel; it adds overhead and isn’t needed
            self.optical_flow_model.eval()
            for p in self.optical_flow_model.parameters():
                p.requires_grad_(False)

    def _get_bilevel_consistency_loss(self, semantic_features, spatial_features, video_clip):
        """
        Bi-level consistency = temporal smoothness on semantic features
                               + optical-flow-based consistency on spatial features.
        video_clip: (B, C, T, H, W)  -- ImageNet-normalized already
        semantic_features: (B, T, D_sem) or (B, T, C, H, W) depending on your head
        spatial_features:  (B, T, C', H', W')  -- features you warp to t+1
        """
        device = video_clip.device

        # -------------------- 1) Semantic temporal smoothness --------------------
        loss_semantic = torch.tensor(0.0, device=device)
        if (self.config.TRAIN.SEMANTIC_LOSS_WEIGHT > 0 and
                semantic_features is not None and
                semantic_features.shape[1] > 1):
            # L1 between consecutive timesteps (t vs t+1)
            # Works for (B,T,...) with any trailing dims
            loss_semantic = self.l1_loss(semantic_features[:, 1:], semantic_features[:, :-1])

        # -------------------- 2) Optical-flow consistency (RAFT) --------------------
        loss_flow = torch.tensor(0.0, device=device)
        if (self.config.TRAIN.OPTICAL_FLOW_LOSS_WEIGHT > 0 and
                spatial_features is not None and
                video_clip.shape[2] > 1):
            B, C, T, H, W = video_clip.shape

            # (a) Denormalize from ImageNet back to [0,1]
            imgnet_mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None, None]
            imgnet_std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None, None]
            video_01 = (video_clip * imgnet_std + imgnet_mean).clamp(0.0, 1.0).float()  # (B,C,T,H,W)

            # (b) Build adjacent pairs as (B*(T-1), C, H, W)
            video_btchw = video_01.permute(0, 2, 1, 3, 4)  # (B,T,C,H,W)
            vid_t = video_btchw[:, :-1].reshape(-1, C, H, W).contiguous()
            vid_t1 = video_btchw[:, 1:].reshape(-1, C, H, W).contiguous()

            # (c) Downscale once to a RAFT-safe size (divisible by 8)
            target_raft_size = (128, 128)
            vid_t_small = F.interpolate(vid_t, size=target_raft_size, mode='bilinear', align_corners=False)
            vid_t1_small = F.interpolate(vid_t1, size=target_raft_size, mode='bilinear', align_corners=False)

            # (d) RAFT normalization to [-1,1]
            vid_t_small = normalize(vid_t_small, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            vid_t1_small = normalize(vid_t1_small, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

            # (e) Call RAFT with no grads + AMP
            with torch.no_grad():
                with autocast(True):
                    # torchvision RAFT returns a list of flows; last is the finest
                    flow_small = self.optical_flow_model(vid_t_small, vid_t1_small)[-1]

            flow_small = flow_small.float()  # (B*(T-1), 2, h, w)
            src_flow_size = flow_small.shape[-2:]  # (h, w) used for warping

            # (f) Warp features from t -> t+1 using the computed flow
            warped_spatial = self.warp_features(spatial_features, flow_small, src_flow_size=src_flow_size)
            actual_next = spatial_features[:, 1:]  # ground-truth features at t+1

            # (g) L1 consistency
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

    def warp_features(self, features, flow, src_flow_size=(128, 128)):
        """
        features: (B, T, H_feat, W_feat, C_feat)
        flow: (B*(T-1), 2, H_src, W_src) from RAFT (e.g., 128x128)
        src_flow_size: (H_src, W_src) used when flow was computed
        """
        B, T, H_feat, W_feat, C_feat = features.shape
        assert T > 1, "Need at least two frames for warping."

        # prepare feature tensor: (B*(T-1), C, H_feat, W_feat)
        features_to_warp = features[:, :-1].reshape(B * (T - 1), H_feat, W_feat, C_feat).permute(0, 3, 1, 2)

        # resize flow to feature resolution
        flow_resized = F.interpolate(flow.to(features.dtype), size=(H_feat, W_feat), mode='bilinear',
                                     align_corners=False)

        # scale u (x) and v (y) components from src_flow_size -> (H_feat, W_feat)
        H_src, W_src = src_flow_size
        scale_x = W_feat / float(W_src)
        scale_y = H_feat / float(H_src)
        flow_resized[:, 0, ...] = flow_resized[:, 0, ...] * scale_x  # u
        flow_resized[:, 1, ...] = flow_resized[:, 1, ...] * scale_y  # v

        # grid sample expects flow in (B*(T-1), H_feat, W_feat, 2)
        flow_feat = flow_resized.permute(0, 2, 3, 1)

        # build base grid and add flow (in pixel units), then normalize to [-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H_feat, device=flow.device),
            torch.arange(W_feat, device=flow.device),
            indexing="ij"
        )
        base_grid = torch.stack((grid_x, grid_y), 2).float().unsqueeze(0).expand(B * (T - 1), -1, -1, -1)

        new_grid = base_grid + flow_feat
        new_grid[..., 0] = 2.0 * new_grid[..., 0] / max(W_feat - 1, 1) - 1.0  # x to [-1,1]
        new_grid[..., 1] = 2.0 * new_grid[..., 1] / max(H_feat - 1, 1) - 1.0  # y to [-1,1]

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

    leftover = (i + 1) % config.TRAIN.GRADIENT_ACCUMULATION_STEPS
    if leftover != 0:
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

    return running_loss / len(dataloader)


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device,
                       ignore_index=-100, use_uncertainty=None, center_only=False):
    """
    - Handles USE_UNCERTAINTY correctly (no double sigmoid).
    - Ignores padded/invalid frames via ignore_index.
    - Sweeps probability thresholds (not logits) to get best F1.
    - Optionally evaluates center frame only (center_only=True).
    """
    model.eval()
    running_loss = 0.0

    # Infer USE_UNCERTAINTY from model.config if not given
    if use_uncertainty is None:
        try:
            use_uncertainty = bool(getattr(getattr(model, "module", model).config.MODEL, "USE_UNCERTAINTY", False))
        except Exception:
            use_uncertainty = False

    all_probs = []
    all_labels = []

    for batch in dataloader:
        video_clip = batch["video_clip"].to(device, non_blocking=True)      # (B,C,T,H,W)
        input_ids  = batch["input_ids"].to(device, non_blocking=True)
        attn_mask  = batch["attention_mask"].to(device, non_blocking=True)
        labels     = batch["labels"].to(device, non_blocking=True)          # (B,T) or (B,T,1)

        outputs = model(video_clip, input_ids, attn_mask)
        total_val_loss, _, _ = criterion(outputs, video_clip, labels)
        running_loss += float(total_val_loss.item())

        refined = outputs[0]                                                # (B,T) or (B,T,1)
        if refined.dim() == 3 and refined.size(-1) == 1:
            refined = refined.squeeze(-1)                                   # (B,T)

        # labels (B,T)
        if labels.dim() == 3 and labels.size(-1) == 1:
            labels_eval = labels.squeeze(-1)
        else:
            labels_eval = labels

        # Convert model output to probabilities correctly
        if use_uncertainty:
            # already in [0,1]
            probs_bt = refined.clamp(0, 1)
        else:
            probs_bt = torch.sigmoid(refined)

        # (optional) center frame only evaluation (common if loss trains on center frame)
        if center_only:
            T = probs_bt.size(1)
            c = T // 2
            probs_bt = probs_bt[:, c:c+1]          # (B,1)
            labels_eval = labels_eval[:, c:c+1]

        # mask out invalid frames (padding etc.)
        valid_mask = (labels_eval != ignore_index)
        if valid_mask.sum().item() == 0:
            continue

        all_probs.append(probs_bt[valid_mask].detach().cpu().float())
        all_labels.append(labels_eval[valid_mask].detach().cpu().int())

    if len(all_probs) == 0:
        print("[VAL] No valid frames found (check ignore_index / labels).")
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    y_prob = torch.cat(all_probs, dim=0).numpy()     # (N,)
    y_true = torch.cat(all_labels, dim=0).numpy()    # (N,)
    # Safety: clamp labels to {0,1}
    y_true = (y_true > 0).astype(np.int32)

    # --- AUROC / AUPRC ---
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        # roc_auc needs both classes present
        auroc = roc_auc_score(y_true, y_prob) if (y_true.min() != y_true.max()) else float("nan")
        auprc = average_precision_score(y_true, y_prob)
    except Exception:
        auroc, auprc = float("nan"), float("nan")

    # --- Accuracy at 0.5 ---
    preds_05   = (y_prob >= 0.5).astype(np.int32)
    val_acc_05 = float((preds_05 == y_true).mean())

    # --- Best F1 over probability thresholds (not logits) ---
    thr_grid = np.linspace(0.05, 0.95, 19, dtype=np.float32)
    best_f1, best_thr = -1.0, None
    best_prec, best_rec, acc_at_thr = 0.0, 0.0, 0.0

    for thr in thr_grid:
        pred = (y_prob >= thr).astype(np.int32)
        tp = int(((y_true == 1) & (pred == 1)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())
        fn = int(((y_true == 1) & (pred == 0)).sum())
        tn = int(((y_true == 0) & (pred == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        acc  = (tp + tn) / max(len(y_true), 1)
        if (f1 > best_f1) or (f1 == best_f1 and rec > best_rec):
            best_f1, best_thr = f1, float(thr)
            best_prec, best_rec, acc_at_thr = prec, rec, acc

    avg_loss = running_loss / max(len(dataloader), 1)
    pos = int((y_true == 1).sum()); neg = int((y_true == 0).sum())
    print(f"[VAL] loss={avg_loss:.4f}  AUROC={auroc:.3f}  AUPRC={auprc:.3f}  "
          f"acc@0.50={val_acc_05:.3f}  bestF1={best_f1:.3f}  thr*={best_thr:.2f}  "
          f"acc@thr*={acc_at_thr:.3f}  (pos={pos}, neg={neg}, N={len(y_true)})")

    # Keep your original return signature:
    return avg_loss, auroc, auprc, val_acc_05, best_f1, best_thr, acc_at_thr

@torch.no_grad()
def evaluate_single_query(model, tokenizer, config, video_id_int, query_text, gt_segments_csv):
    """
    Validates ONLY the specified video + text query:
      - per-frame metrics (AUROC, AUPRC, F1/Acc at thr*)
      - segment metrics (tIoU@{0.3,0.5,0.7}) using gt_segments.csv
    """

    # 1) Build the full val dataset
    from dataset import EndoscopyLocalizationDataset
    val_csv = config.DATA.VAL_TRIPLETS
    val_ds  = EndoscopyLocalizationDataset(val_csv, tokenizer, clip_length=config.DATA.CLIP_LENGTH, is_training=False)

    # 2) Filter rows to (video 5, given query)
    import os, pandas as pd
    import numpy as np
    val_df = pd.read_csv(val_csv)
    target_vid = f"CHOLEC80__video{video_id_int:02d}"
    mask = (val_df["text_query"].str.strip().str.lower() == query_text.strip().lower()) & \
           (val_df["frame_path"].str.contains(f"/{target_vid}/"))
    idxs = np.nonzero(mask.values)[0].tolist()
    if len(idxs) == 0:
        print(f"[EVAL] No rows matched video={target_vid}, query='{query_text}'. Check spelling/case & triplets.")
        return

    from torch.utils.data import Subset, DataLoader
    sub_ds = Subset(val_ds, idxs)
    val_loader = DataLoader(
        sub_ds,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if config.DATA.NUM_WORKERS > 0 else False
    )

    # 3) Run forward passes, collect per-frame probs/labels and indices
    model.eval()
    all_probs, all_labels = [], []
    timeline = {}  # frame_idx -> (prob,label) aggregated (mean if duplicates)
    for batch in val_loader:
        video_clip = batch["video_clip"].to(config.TRAIN.DEVICE, non_blocking=True)
        input_ids = batch["input_ids"].to(config.TRAIN.DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(config.TRAIN.DEVICE, non_blocking=True)
        labels = batch["labels"].to(config.TRAIN.DEVICE, non_blocking=True)  # (B,T)
        out = model(video_clip, input_ids, attention_mask)  # adapt if your forward signature differs

        # 1) pick the refined output
        refined = out[0] if not (isinstance(out, dict) and "logits" in out) else out["logits"]
        if refined.dim() == 3 and refined.size(-1) == 1:
            refined = refined.squeeze(-1)  # (B,T)

        # 2) infer USE_UNCERTAINTY from model.config
        try:
            use_uncertainty_eval = bool(getattr(getattr(model, "module", model).config.MODEL, "USE_UNCERTAINTY", False))
        except Exception:
            use_uncertainty_eval = False

        # 3) convert to probabilities (avoid double-sigmoid when uncertainty is on)
        probs = refined.clamp(0, 1) if use_uncertainty_eval else torch.sigmoid(refined)
        probs = probs.detach()

        # Per-frame pool for AUROC/AUPRC/F1 (flatten valid)
        valid = (labels >= 0)  # ignore_index-safe if you use -1
        if valid.sum().item() > 0:
            all_probs.append(probs[valid].float().cpu())
            all_labels.append(labels[valid].int().cpu())

        # Build timeline (this subset is all from the same video & query)
        vids = batch["video_id"]
        fidx = batch["frame_indices"].cpu().numpy()  # (B,T)
        p_np = probs.cpu().numpy()
        y_np = (labels > 0).int().cpu().numpy()
        for b in range(p_np.shape[0]):
            for t in range(p_np.shape[1]):
                k = int(fidx[b, t])
                if k not in timeline:
                    timeline[k] = {"p": [], "y": []}
                timeline[k]["p"].append(float(p_np[b, t]))
                timeline[k]["y"].append(int(y_np[b, t]))

    if len(all_probs) == 0:
        print("[EVAL] No valid frames found in subset.")
        return

    # 4) Per-frame metrics
    import numpy as np
    y_prob = torch.cat(all_probs).numpy()
    y_true = (torch.cat(all_labels).numpy() > 0).astype(np.int32)

    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    auprc = average_precision_score(y_true, y_prob)

    # best F1 sweep
    thrs = np.linspace(0.0, 1.0, 101)
    f1s  = []
    for th in thrs:
        y_hat = (y_prob >= th).astype(np.int32)
        if y_hat.max() == 0 and y_true.max() == 0:
            f1s.append(0.0)
        else:
            f1s.append(f1_score(y_true, y_hat))
    best_idx = int(np.argmax(f1s))
    best_thr = float(thrs[best_idx])
    best_f1  = float(f1s[best_idx])

    # accuracy at best_thr
    y_hat_best = (y_prob >= best_thr).astype(np.int32)
    acc_at_thr = float(accuracy_score(y_true, y_hat_best))

    print(f"[EVAL][Per-frame] AUROC={auroc:.3f} AUPRC={auprc:.3f}  Best F1={best_f1:.3f}  Acc@thr*={acc_at_thr:.3f}  thr*={best_thr:.2f}")

    # 5) Segment-level tIoU using gt_segments.csv
    # Build predicted segments on the timeline (frame units)
    if len(timeline) == 0:
        print("[EVAL] No timeline collected. Skipping tIoU.")
        return

    ks = sorted(timeline.keys())
    prob_line = [float(np.mean(timeline[k]["p"])) for k in ks]
    pred_bin  = [1 if p >= best_thr else 0 for p in prob_line]
    pred_segs_rel = _decode_segments(pred_bin)  # relative to ks positions
    # Map back to absolute frame indices
    pred_segs = [(ks[s], ks[e]) for (s, e) in pred_segs_rel]

    # Load GT segments (expect columns: video_id, text_query, start_frame, end_frame)
    gt_df = pd.read_csv(gt_segments_csv)
    mask_gt = (gt_df["video_id"].astype(str).str.strip().str.lower() == target_vid.lower()) & \
              (gt_df["text_query"].str.strip().str.lower() == query_text.strip().lower())
    gt_rows = gt_df.loc[mask_gt]
    gt_segs = []
    for _, r in gt_rows.iterrows():
        s = int(r["start_frame"]); e = int(r["end_frame"])
        if s > e: s, e = e, s
        gt_segs.append((s, e))

    if len(gt_segs) == 0:
        print(f"[EVAL][tIoU] No GT segments found in {gt_segments_csv} for {target_vid} / '{query_text}'.")
        return

    print(f"[EVAL] Pred segs: {len(pred_segs)}   GT segs: {len(gt_segs)}   (units: frames)")
    for thr in (0.3, 0.5, 0.7):
        prec, rec, f1 = _tiou_prf1(pred_segs, gt_segs, thr=thr)
        print(f"[EVAL][tIoU@{thr:.1f}]  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")

    # Optional: boundary error diagnostics for best match (if you want)
    # (Match each pred to its best-IoU GT and print start/end errors)
    matches = []
    used = set()
    for ps in pred_segs:
        best, best_i = -1, -1
        for i, gs in enumerate(gt_segs):
            if i in used: continue
            v = _iou_1d(ps, gs)
            if v > best:
                best, best_i = v, i
        if best_i >= 0:
            used.add(best_i)
            gs = gt_segs[best_i]
            se = abs(ps[0]-gs[0]); ee = abs(ps[1]-gs[1])
            matches.append((best, se, ee))
    if matches:
        iou_mean = float(np.mean([m[0] for m in matches]))
        se_mean  = float(np.mean([m[1] for m in matches]))
        ee_mean  = float(np.mean([m[2] for m in matches]))
        print(f"[EVAL][Boundary] mean IoU={iou_mean:.3f}  mean|start_err|={se_mean:.1f}  mean|end_err|={ee_mean:.1f}")

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
    # checkpoint directory handling
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
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
        print(f"[LOAD] Loaded checkpoint. missing={len(missing)} unexpected={len(unexpected)}")

    if args.eval_single_query:
        evaluate_single_query(model, tokenizer_for_dataloaders, config,
                              video_id_int=args.video_id,
                              query_text=args.query,
                              gt_segments_csv=args.gt_segments or "/users/2/240331715/data/project_folder/Language-Guided-Endoscopy-Localization/gt_segments.csv")
        return

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
        _ = load_weights_for_finetune(
            model.module if isinstance(model, nn.DataParallel) else model,
            args.finetune_from,
            device
        )
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
        val_loss, auroc, auprc, val_acc_05, best_f1, thr, acc_at_thr = validate_one_epoch(
            model, val_loader, criterion, device
        )

        print(f"\tValidation Loss: {val_loss:.4f}")
        print(f"\tValidation AUROC: {auroc:.3f}")
        print(f"\tValidation AUPRC: {auprc:.3f}")
        print(f"\tValidation Accuracy@0.50: {val_acc_05:.4f} ({val_acc_05:.2%})")
        print(f"\tBest F1: {best_f1:.3f} (thr={thr:.2f})")
        print(f"\tValidation Accuracy@thr*: {acc_at_thr:.4f} ({acc_at_thr:.2%})")

        try:
            current_lr = scheduler.get_last_lr()[0]
        except Exception:
            current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"\tTraining Loss: {train_loss:.4f}")
        print(f"\tValidation Loss: {val_loss:.4f}")
        print(f"\tValidation Accuracy@0.50: {val_acc_05:.4f} ({val_acc_05:.2%})")
        print(f"\tCurrent Learning Rate: {current_lr:.6f}")

        # --- Save checkpoints (latest and best) ---
        latest_model_path = os.path.join(checkpoint_dir, f"latest_model_epoch_{epoch + 1}.pth")
        save_checkpoint(model, optimizer, scheduler, None, epoch + 1, best_val_loss, latest_model_path)

        if math.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            print(f"	*** New best model found! Validation loss: {val_loss:.4f} ***")
            save_checkpoint(model, optimizer, scheduler, None, epoch + 1, best_val_loss, best_model_path)

        if args.eval_single_each_epoch:
            evaluate_single_query(
                model,
                tokenizer_for_dataloaders,  # already defined earlier in main
                config,
                video_id_int=args.video_id,
                query_text=args.query,
                gt_segments_csv=(
                        args.gt_segments
                        or "/users/2/240331715/data/project_folder/Language-Guided-Endoscopy-Localization/gt_segments.csv"
                ),
            )

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
    parser.add_argument("--eval_single_query", action="store_true",
                        help="Run evaluation on a single (video, query) pair and exit.")
    parser.add_argument("--video_id", type=int, default=5,
                        help="Video number (e.g., 5 for CHOLEC80__video05).")
    parser.add_argument("--query", type=str, default="Calot triangle dissection phase",
                        help="Exact text query to evaluate.")
    parser.add_argument("--gt_segments", type=str, required=False,
                        help="Path to gt_segments.csv (columns: video_id,text_query,start_frame,end_frame).")
    parser.add_argument("--checkpoint", type=str, required=False,
                        help="Path to model weights to load before evaluation (e.g., best_model.pth).")
    parser.add_argument("--eval_single_each_epoch", action="store_true",
                        help="After each epoch, run single (video,query) evaluation (prints tIoU).")

    args = parser.parse_args()
    main(args)