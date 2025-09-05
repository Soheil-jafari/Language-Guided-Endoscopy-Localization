import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import generalized_temporal_iou, span_cxw_to_xx

class SetCriterion(nn.Module):
    def __init__(self, matcher, weight_dict, eos_coef):
        super().__init__(); self.matcher, self.weight_dict, self.eos_coef = matcher, weight_dict, eos_coef
        self.register_buffer('empty_weight', torch.tensor([1.0, eos_coef]))
    def loss_spans(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        src, target = outputs['pred_spans'][idx], torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        return {'loss_span': F.l1_loss(src, target, reduction='mean'), 'loss_giou': 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src), span_cxw_to_xx(target))).mean()}
    def loss_labels(self, outputs, targets, indices):
        src_logits = outputs['pred_logits']; idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 1, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        return {'loss_ce': F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)}
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        losses = {}
        for loss_type in ['spans', 'labels']:
            losses.update(getattr(self, f'loss_{loss_type}')(outputs, targets, indices))

        # --- ADD THIS FINAL BLOCK ---
        # Apply the weights from the config to each loss component
        for loss_key, weight in self.weight_dict.items():
            if loss_key in losses:
                losses[loss_key] *= weight
        # --- END OF BLOCK ---

        return losses

