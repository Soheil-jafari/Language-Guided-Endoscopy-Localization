import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
from .utils import generalized_temporal_iou, span_cxw_to_xx

class HungarianMatcher(nn.Module):
    def __init__(self, cost_span: float = 1, cost_giou: float = 1):
        super().__init__(); self.cost_span, self.cost_giou = cost_span, cost_giou
    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_spans"].shape[:2]
        out_span, tgt_span = outputs["pred_spans"].flatten(0, 1), torch.cat([v["spans"] for v in targets])
        cost_span = torch.cdist(out_span, tgt_span, p=1)
        cost_giou = -generalized_temporal_iou(span_cxw_to_xx(out_span), span_cxw_to_xx(tgt_span))
        C = (self.cost_span * cost_span + self.cost_giou * cost_giou).view(bs, num_queries, -1).cpu()
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split([len(v["spans"]) for v in targets], -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

