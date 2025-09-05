import torch
import torch.nn as nn
import math

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=2*math.pi):
        super().__init__()
        self.num_pos_feats, self.temperature, self.normalize, self.scale = num_pos_feats, temperature, normalize, scale
    def forward(self, x, mask):
        x_embed = mask.cumsum(1, dtype=torch.float32)
        if self.normalize: x_embed = x_embed / (x_embed[:, -1:] + 1e-6) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos = x_embed[:, :, None] / dim_t
        return torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3).flatten(2)

def build_position_encoding(cfg):
    return PositionEmbeddingSine(cfg.hidden_dim, normalize=True)

