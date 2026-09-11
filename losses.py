"""Masked binary objectives and backward-flow feature consistency.

Evidence order is [positive, negative]. EDL uses expected categorical NLL and
target-adjusted Dirichlet KL to the uniform prior (Sensoy et al., 2018).
This objective does not by itself establish calibrated unknown-event rejection.
"""
import torch
from torch import nn
from torch.nn import functional as F
from data_contract import IGNORE_INDEX


def valid_targets(target):
    mask = target != IGNORE_INDEX
    if not torch.all((target[mask] == 0) | (target[mask] == 1)):
        raise ValueError('Targets must be 0, 1, or -100')
    return mask


class EvidentialLoss(nn.Module):
    def __init__(self, regularizer_weight=0.2):
        super().__init__()
        self.regularizer_weight = regularizer_weight
        self.annealing = 1.0

    def forward(self, evidence, target):
        mask = valid_targets(target)
        if not mask.any():
            return evidence.sum() * 0
        ev = evidence.float()[mask]
        if not torch.isfinite(ev).all() or (ev < 0).any():
            raise ValueError('Evidence must be finite and nonnegative')
        y = target.float()[mask]
        onehot = torch.stack((y, 1-y), dim=-1)
        alpha = ev + 1
        nll = (onehot * (torch.digamma(alpha.sum(-1, keepdim=True)) - torch.digamma(alpha))).sum(-1)
        adjusted = onehot + (1-onehot)*alpha
        total = adjusted.sum(-1, keepdim=True)
        kl = (torch.lgamma(total).squeeze(-1) - torch.lgamma(adjusted).sum(-1)
              + ((adjusted-1)*(torch.digamma(adjusted)-torch.digamma(total))).sum(-1))
        return (nll + self.regularizer_weight*self.annealing*kl).mean()


def backward_warp(source, backward_flow):
    """Sample source at target pixel + flow(target->source); return in-bounds mask.

    source: N,C,H,W. Flow: N,2,h,w, in pixels at its own resolution.
    Out-of-bounds samples are excluded; this is not an occlusion detector.
    """
    n, _, h, w = source.shape
    fh, fw = backward_flow.shape[-2:]
    flow = F.interpolate(backward_flow.float(), (h,w), mode='bilinear', align_corners=False)
    flow = flow * flow.new_tensor([w/fw, h/fh])[None,:,None,None]
    yy, xx = torch.meshgrid(torch.arange(h,device=source.device), torch.arange(w,device=source.device), indexing='ij')
    xy = torch.stack((xx,yy), dim=-1).float()[None] + flow.permute(0,2,3,1)
    valid = (xy[...,0]>=0)&(xy[...,0]<=w-1)&(xy[...,1]>=0)&(xy[...,1]<=h-1)
    grid = 2*(xy+0.5)/xy.new_tensor([w,h])-1
    warped = F.grid_sample(source.float(), grid, align_corners=False, padding_mode='zeros')
    return warped, valid[:,None]


class MasterLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_uncertainty = config.MODEL.USE_UNCERTAINTY
        self.use_bilevel = config.TRAIN.USE_BILEVEL_CONSISTENCY
        self.register_buffer('pos_weight', torch.tensor(float(config.TRAIN.BCE_POS_WEIGHT)))
        self.evidential_loss = EvidentialLoss(config.TRAIN.EVIDENTIAL_LAMBDA)
        if self.use_bilevel and config.TRAIN.OPTICAL_FLOW_LOSS_WEIGHT > 0:
            from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
            self.optical_flow_model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False)
            self.optical_flow_model.requires_grad_(False).eval()

    def set_epoch(self, epoch):
        self.evidential_loss.annealing = min(1., (epoch+1)/max(1,self.config.TRAIN.EVIDENTIAL_ANNEAL_EPOCHS))

    def forward(self, outputs, video, target):
        refined, raw, _, semantic, spatial, evidence = outputs
        mask = valid_targets(target)
        if not mask.any():
            zero = raw.sum()*0 + refined.sum()*0
            return zero, zero, zero
        def bce(logits):
            return F.binary_cross_entropy_with_logits(logits.squeeze(-1).float()[mask], target.float()[mask], pos_weight=self.pos_weight)
        primary = bce(raw) + (self.evidential_loss(evidence,target) if self.use_uncertainty else bce(refined))
        regularizer = primary*0
        pairs = mask[:,1:] & mask[:,:-1]
        if pairs.any():
            if self.use_bilevel:
                if self.config.TRAIN.SEMANTIC_LOSS_WEIGHT > 0:
                    regularizer = regularizer + self.config.TRAIN.SEMANTIC_LOSS_WEIGHT * (semantic[:,1:]-semantic[:,:-1]).abs()[pairs].mean()
                if self.config.TRAIN.OPTICAL_FLOW_LOSS_WEIGHT > 0:
                    b,c,t,h,w = video.shape
                    mean = video.new_tensor([.485,.456,.406])[None,:,None,None,None]
                    std = video.new_tensor([.229,.224,.225])[None,:,None,None,None]
                    images = (video*std+mean).float().clamp(0,1).permute(0,2,1,3,4)
                    prev = F.interpolate(images[:,:-1].reshape(-1,c,h,w),(128,128),mode='bilinear',align_corners=False)
                    nxt = F.interpolate(images[:,1:].reshape(-1,c,h,w),(128,128),mode='bilinear',align_corners=False)
                    self.optical_flow_model.eval()
                    with torch.no_grad(), torch.autocast(device_type=video.device.type, enabled=False):
                        flow = self.optical_flow_model(nxt*2-1,prev*2-1)[-1]
                    _,_,sh,sw,dim = spatial.shape
                    source = spatial[:,:-1].reshape(-1,sh,sw,dim).permute(0,3,1,2)
                    destination = spatial[:,1:].reshape(-1,sh,sw,dim).permute(0,3,1,2)
                    warped, valid = backward_warp(source,flow)
                    valid = valid & pairs.reshape(-1,1,1,1)
                    if valid.any():
                        diff = (warped-destination.float()).abs()
                        regularizer = regularizer + self.config.TRAIN.OPTICAL_FLOW_LOSS_WEIGHT * diff.masked_select(valid.expand_as(diff)).mean()
            elif self.config.TRAIN.TEMPORAL_LOSS_WEIGHT > 0:
                probs = refined if self.use_uncertainty else refined.sigmoid()
                regularizer = self.config.TRAIN.TEMPORAL_LOSS_WEIGHT*(probs[:,1:]-probs[:,:-1]).abs()[pairs].mean()
        return primary+regularizer, primary, regularizer
