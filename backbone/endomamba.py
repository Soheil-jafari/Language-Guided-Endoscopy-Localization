import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional, List
import os
import math

from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False
        pos = torch.arange(0, max_len, device=device).float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        # x is expected to have shape (batch_size, seq_len, d_model) or similar
        # We need to get the sequence length from the correct dimension
        seq_len = x.size(1)
        return self.encoding[:seq_len, :]


class Block(nn.Module):
    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=True, residual_in_fp32=True, drop_path=0.):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.bimamba = mixer_cls.keywords.get('bimamba', True)

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None):
        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight, self.norm.bias, residual=residual, prenorm=True,
                residual_in_fp32=self.residual_in_fp32, eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual


def create_block(d_model, ssm_cfg=None, norm_epsilon=1e-5, drop_path=0., rms_norm=True, residual_in_fp32=True,
                 fused_add_norm=True, layer_idx=None, bimamba=True, device=None, dtype=None, return_last_state=False):
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None: ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba=bimamba, **ssm_cfg, return_last_state=return_last_state,
                        **factory_kwargs)
    norm_cls = partial(RMSNorm, eps=norm_epsilon)
    block = Block(d_model, mixer_cls, norm_cls=norm_cls, drop_path=drop_path, fused_add_norm=fused_add_norm,
                  residual_in_fp32=residual_in_fp32)
    block.layer_idx = layer_idx
    return block


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, kernel_size=1, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=(kernel_size, patch_size[0], patch_size[1]),
                              stride=(kernel_size, patch_size[0], patch_size[1]))

    def forward(self, x):
        return self.proj(x)


class OriginalEndoMamba(nn.Module):
    def __init__(self, img_size=224, patch_size=16, depth=24, embed_dim=192, channels=3, num_classes=0,
                 drop_path_rate=0.1, ssm_cfg=None, norm_epsilon=1e-5, fused_add_norm=True, rms_norm=True,
                 residual_in_fp32=True, kernel_size=1, device=None, dtype=None, num_spatial_layers=12,
                 with_cls_token=True, **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, kernel_size=kernel_size,
                                      in_chans=channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.with_cls_token = with_cls_token
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.temporal_pos_embedding = PositionalEncoding(embed_dim, 8192, device=device)
        self.pos_drop = nn.Dropout(p=0.0)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.layers = nn.ModuleList()
        for i in range(depth):
            current_bimamba = True if i < num_spatial_layers else False
            block = create_block(embed_dim, ssm_cfg=ssm_cfg, norm_epsilon=norm_epsilon, rms_norm=rms_norm,
                                 residual_in_fp32=residual_in_fp32, fused_add_norm=fused_add_norm, layer_idx=i,
                                 drop_path=dpr[i], bimamba=current_bimamba, device=device, dtype=dtype)
            self.layers.append(block)

        self.norm_f = RMSNorm(embed_dim, eps=norm_epsilon, **factory_kwargs)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        if with_cls_token:
            trunc_normal_(self.cls_token, std=.02)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        cls_token = self.cls_token.expand(B * T, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed

        cls_tokens = x[:, :1, :]
        patch_tokens = x[:, 1:, :]

        patch_tokens = rearrange(patch_tokens, '(b t) n c -> (b n) t c', b=B, t=T)
        patch_tokens = patch_tokens + self.temporal_pos_embedding(patch_tokens).unsqueeze(0)
        patch_tokens = rearrange(patch_tokens, '(b n) t c -> b t n c', b=B, t=T)

        cls_tokens = rearrange(cls_tokens, '(b t) n c -> b t n c', b=B, t=T)
        hidden_states = torch.cat((cls_tokens, patch_tokens), dim=2)
        hidden_states = self.pos_drop(hidden_states)

        residual = None

        for layer in self.layers:
            if layer.bimamba:
                hidden_states = rearrange(hidden_states, 'b t n m -> (b t) n m')
                if residual is not None: residual = rearrange(residual, 'b t n m -> (b t) n m')
            else:
                hidden_states = rearrange(hidden_states, 'b t n m -> b (t n) m')
                if residual is not None: residual = rearrange(residual, 'b t n m -> b (t n) m')

            hidden_states, residual = layer(hidden_states, residual)

            if layer.bimamba:
                hidden_states = rearrange(hidden_states, '(b t) n m -> b t n m', b=B, t=T)
                if residual is not None: residual = rearrange(residual, '(b t) n m -> b t n m', b=B, t=T)
            else:
                hidden_states = rearrange(hidden_states, 'b (t n) m -> b t n m', n=x.shape[1], t=T)
                if residual is not None: residual = rearrange(residual, 'b (t n) m -> b t n m', n=x.shape[1], t=T)

        fused_add_norm_fn = rms_norm_fn
        hidden_states = fused_add_norm_fn(hidden_states, self.norm_f.weight, self.norm_f.bias, eps=self.norm_f.eps,
                                          residual=residual, prenorm=False, residual_in_fp32=True)
        return hidden_states


class EndoMambaBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = OriginalEndoMamba(
            embed_dim=384,
            depth=24,
            num_spatial_layers=12,
            num_classes=0,
        )
        self.embed_dim = self.model.embed_dim

        weights_path = config.MODEL.ENDOMAMBA_WEIGHTS_PATH
        if weights_path and os.path.exists(weights_path):
            print(f"Loading EndoMamba pretrained weights from: {weights_path}")
            state_dict = torch.load(weights_path, map_location='cpu')

            if 'model' in state_dict: state_dict = state_dict['model']

            state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items()}
            msg = self.model.load_state_dict(state_dict, strict=False)
            print(f"Weight loading message: {msg}")
        else:
            print(f"Warning: EndoMamba weights not found at '{weights_path}'. Using random initialization.")

    def forward_features(self, x, get_all=True):
        """
        Provides an output compatible with your `LocalizationFramework`.
        Output shape: (B, 1 + T*num_patches, C)
        """
        # The output of the OriginalEndoMamba is (B, T, 1+num_patches, C)
        final_tokens = self.model(x)

        # 1. Average the CLS token from each frame to get one global CLS token
        cls_output_tokens = final_tokens[:, :, 0, :].mean(dim=1, keepdim=True)

        # 2. Get all patch tokens from all frames
        patch_output_tokens = final_tokens[:, :, 1:, :]

        # 3. Flatten the patch tokens
        patch_output_tokens = rearrange(patch_output_tokens, 'b t n c -> b (t n) c')

        # 4. Concatenate the global CLS token with all patch tokens
        all_tokens = torch.cat((cls_output_tokens, patch_output_tokens), dim=1)

        return all_tokens

    @property
    def patch_embed(self):
        return self.model.patch_embed