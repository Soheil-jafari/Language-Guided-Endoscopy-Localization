import math
import os
from collections import OrderedDict
import warnings
from itertools import repeat
from functools import partial
import collections.abc as container_abcs
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub
import torch.utils.checkpoint as checkpoint
from einops import rearrange

# --- Constants and Utilities ---
_logger = logging.getLogger(__name__)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Computes standard normal cumulative distribution function
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution.
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# --- Helper functions from helpers.py and timesformer.py ---

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vit_base_patch16_224': _cfg(
        url="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
    ),
    # Add other model configs if needed
}


def _conv_filter(state_dict, model):
    """Adapt 2D ViT patch-embed weights to a 3D Conv (1, p, p)."""
    out = {}
    for k, v in state_dict.items():
        if k.endswith('patch_embed.proj.weight'):
            # v: (out, 3, p, p)  --> need (out, 3, 1, p, p)
            if v.ndim == 4:
                v = v.unsqueeze(2)
        out[k] = v
    return out

# --- Helper function for loading pre-trained weights into VisionTransformer ---
# This function explicitly handles local file paths first, then URLs, then random init.
def load_pretrained(model, cfg=None, num_classes=1000, in_chans=3, filter_fn=None,
                    img_size=224, num_frames=16, num_patches=196,
                    attention_type='divided_space_time', pretrained_model='',
                    strict=False):
    """
    Load ViT weights (local file wins; else URL). Defaults to strict=False so
    temporal params missing in 2D ViTs won't break loading.
    """
    if cfg is None:
        cfg = getattr(model, 'default_cfg', None) or default_cfgs.get('vit_base_patch16_224')
        if cfg is None:
            _logger.warning("No cfg and no default_cfg; using random init.")
            warnings.warn("No valid pretrained model found. Using random initialization.")
            return

    if filter_fn is None:
        filter_fn = _conv_filter

    def _clean_sd(sd):
        # unwrap common wrappers
        if any(k.startswith('module.') for k in sd.keys()):
            sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
        if any(k.startswith('backbone.') for k in sd.keys()):
            sd = {k.replace('backbone.', '', 1): v for k, v in sd.items()}
        return sd

    def _maybe_drop_head(sd):
        if num_classes == 0:
            for k in list(sd.keys()):
                if k.startswith('head.'):
                    del sd[k]
        return sd

    # 1) Local file
    if pretrained_model and os.path.exists(pretrained_model):
        _logger.info(f'Loading pretrained from local: {pretrained_model}')
        try:
            ckpt = torch.load(pretrained_model, map_location='cpu')
            sd = ckpt.get('model', ckpt)
            sd = _clean_sd(sd)
            sd = _maybe_drop_head(sd)
            sd = filter_fn(sd, model)  # <— pass model for Conv3d inflation
            missing, unexpected = model.load_state_dict(sd, strict=strict)
            _logger.info(f'loaded. missing:{len(missing)} unexpected:{len(unexpected)}')
            return
        except Exception as e:
            _logger.error(f"Local load failed: {e}")
            warnings.warn(f"Failed local load: {e}. Falling back to URL/random.")

    # 2) URL
    if cfg.get('url'):
        _logger.info(f'Loading pretrained from URL: {cfg["url"]}')
        try:
            sd = torch.hub.load_state_dict_from_url(cfg['url'], map_location='cpu', progress=True)
            sd = sd.get('model', sd)
            sd = _clean_sd(sd)
            sd = _maybe_drop_head(sd)
            sd = filter_fn(sd, model)
            missing, unexpected = model.load_state_dict(sd, strict=strict)
            _logger.info(f'loaded. missing:{len(missing)} unexpected:{len(unexpected)}')
            return
        except Exception as e:
            warnings.warn(f"Failed URL load: {e}. Using random init.")

    warnings.warn("No valid pretrained model; using random initialization.")

# --- Core Backbone Classes ---

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x, return_attn=False):
        B, N, C = x.shape
        if self.with_qkv:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q, k, v = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
            x = self.proj(x)
            x = self.proj_drop(x)
        if return_attn:
            return x, attn
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
        super().__init__()
        self.attention_type = attention_type
        self.class_tokens = 1  # Assuming only one CLS token is used per frame or video
        assert (attention_type in ['divided_space_time', 'space_only', 'joint_space_time'])

        self.norm1 = norm_layer(dim)
        # This is the spatial attention (used if attention_type is not joint_space_time)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # Temporal Attention Parameters (for divided_space_time)
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, B, T, W, return_attn=False):
        # x: (B, 1 + T*H*W, D) if class_tokens==1 else (B, T*H*W, D)
        num_spatial_tokens = (x.size(1) - self.class_tokens) // T
        H = num_spatial_tokens // W  # patch-grid height

        if self.attention_type in ['space_only', 'joint_space_time']:
            # standard ViT-style residuals: DropPath at the add, not inside attn/mlp
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

        elif self.attention_type == 'divided_space_time':
            # ====== Split CLS vs patch tokens ======
            if self.class_tokens == 1:
                cls_tok = x[:, :1, :]  # (B, 1, D)
                patch_tok = x[:, 1:, :]  # (B, T*H*W, D)
            else:
                cls_tok = None
                patch_tok = x  # (B, T*H*W, D)

            # 1) TEMPORAL ATTENTION on patches (per location)
            # reshape: (B, T*H*W, D) -> ((B*H*W), T, D)
            xt = rearrange(patch_tok, 'b (h w t) d -> (b h w) t d', b=B, h=H, w=W, t=T)

            # ViT-style: norm -> attn ; apply DropPath at the add
            temp_out = self.temporal_attn(self.temporal_norm1(xt))  # ((B*H*W), T, D)
            temp_out = rearrange(temp_out, '(b h w) t d -> b (h w t) d', b=B, h=H, w=W, t=T)  # (B, T*H*W, D)

            # build a residual tensor aligned with x: zeros for CLS, temporal residual for patches
            temp_residual_full = torch.zeros_like(x)
            if cls_tok is not None:
                temp_residual_full[:, 1:, :] = temp_out
            else:
                temp_residual_full = temp_out

            # add temporal residual (DropPath applied here, once)
            x = x + self.drop_path(temp_residual_full)

            # after temporal add, refresh tokens for spatial step
            if self.class_tokens == 1:
                cls_tok = x[:, :1, :]
                patch_tok = x[:, 1:, :]
            else:
                patch_tok = x

            # 2) SPATIAL ATTENTION on each frame (per time step)
            # reshape patches to per-frame sequences: (B, T*H*W, D) -> (B*T, H*W, D)
            xs = rearrange(patch_tok, 'b (h w t) d -> (b t) (h w) d', b=B, h=H, w=W, t=T)

            # replicate CLS for each frame if present and prepend
            if cls_tok is not None:
                cls_for_frames = cls_tok.repeat(1, T, 1)  # (B, T, D)
                cls_for_frames = rearrange(cls_for_frames, 'b t d -> (b t) 1 d')  # (B*T, 1, D)
                xs = torch.cat([cls_for_frames, xs], dim=1)  # (B*T, 1 + H*W, D)

            if return_attn:
                _, attn = self.attn(self.norm1(xs), return_attn=True)
                return attn

            # spatial attention (again, DropPath only at the add)
            spatial_out = self.attn(self.norm1(xs))  # (B*T, 1+H*W, D) or (B*T, H*W, D)

            # split CLS vs patches in spatial output
            if cls_tok is not None:
                cls_per_t = spatial_out[:, 0, :]  # (B*T, D)
                cls_per_t = rearrange(cls_per_t, '(b t) d -> b t d', b=B, t=T)  # (B, T, D)
                cls_residual = cls_per_t.mean(dim=1, keepdim=True)  # (B, 1, D) — average across time
                spatial_patches = spatial_out[:, 1:, :]  # (B*T, H*W, D)
            else:
                cls_residual = None
                spatial_patches = spatial_out  # (B*T, H*W, D)

            # put patch spatial residuals back to (B, T*H*W, D)
            spatial_patches = rearrange(spatial_patches, '(b t) (h w) d -> b (h w t) d', b=B, h=H, w=W, t=T)

            # build residual tensor aligned with x: CLS in slot 0, patches in 1:
            spatial_residual_full = torch.zeros_like(x)
            if cls_tok is not None:
                spatial_residual_full[:, :1, :] = cls_residual
                spatial_residual_full[:, 1:, :] = spatial_patches
            else:
                spatial_residual_full = spatial_patches

            # add spatial residual (DropPath applied here, once)
            x = x + self.drop_path(spatial_residual_full)

            # 3) MLP sublayer (standard ViT)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class PatchEmbed(nn.Module):
    """ Video to Patch Embedding for Spatio-Temporal Transformer (e.g., M²CRL)
        Performs 3D convolution to create patches.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16,
                 attention_type='divided_space_time'):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        # Calculate number of patches per frame
        self.num_patches_h = img_size[0] // patch_size[0]
        self.num_patches_w = img_size[1] // patch_size[1]
        self.num_patches = self.num_patches_h * self.num_patches_w

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.attention_type = attention_type

        # Use 3D convolution for patch embedding (kernel and stride for time is 1)
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=(1, patch_size[0], patch_size[1]),
                              stride=(1, patch_size[0], patch_size[1]))

    def forward(self, x):
        B, C, T, H, W = x.shape  # x: (B, C, T, H, W)

        # Ensure that H and W are divisible by patch_size
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # Project patches, result is (B, embed_dim, T, num_patches_h, num_patches_w)
        x = self.proj(x)

        # Reshape to (B, T, num_patches_h * num_patches_w, embed_dim) for attention
        x = x.flatten(3).permute(0, 2, 3, 1)  # (B, T, num_patches, embed_dim)

        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for 3D inputs (video)
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm,
                 num_frames=16, attention_type='divided_space_time',
                 use_checkpoint=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.num_frames = num_frames
        self.patch_size = to_2tuple(patch_size)  # ensure tuple (pH, pW)
        self.default_cfg = default_cfgs.get('vit_base_patch16_224')
        self.attention_type = attention_type
        self.use_checkpoint = use_checkpoint

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=num_frames,
            attention_type=attention_type)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))  # Position embedding for CLS + all patches in a single frame
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Time-specific embeddings (learnable)
        self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))  # Time embedding for each frame
        self.time_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attention_type=attention_type)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head (not used in this framework, but kept for compatibility with pretrained models)
        self.head = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.temporal_embed, std=.02)  # Initialize time embedding
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'temporal_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _resize_pos_embed(self, posemb, grid_size_height, grid_size_width, num_frames, start_index=1):
        """
        Resize a 2D ViT positional embedding (1 + H*W, D) to our current spatial grid (H', W'),
        then tile across time (T = num_frames). Works even if the source grid wasn't square.
        """
        # posemb: (1, 1 + old_tokens, D)
        posemb_tok = posemb[:, :start_index]  # (1, 1, D) CLS
        posemb_grid = posemb[:, start_index:]  # (1, old_H*old_W, D)
        old_num = posemb_grid.shape[1]

        # Try to infer old (H, W)
        old_h = int(round((old_num) ** 0.5))
        old_w = old_h
        if old_h * old_w != old_num:
            # fallback: keep aspect similar to target if not square
            # choose (old_h, old_w) such that old_h*old_w == old_num and |old_h/old_w - H'/W'| is small
            target_ratio = grid_size_height / float(grid_size_width)
            best = None
            for h in range(1, old_num + 1):
                if old_num % h == 0:
                    w = old_num // h
                    ratio = h / float(w)
                    score = abs(ratio - target_ratio)
                    if best is None or score < best[0]:
                        best = (score, h, w)
            _, old_h, old_w = best

        # (1, old_H*old_W, D) -> (1, D, old_H, old_W)
        posemb_grid = posemb_grid.reshape(1, old_h, old_w, -1).permute(0, 3, 1, 2)
        # resize to (H', W')
        posemb_grid = F.interpolate(posemb_grid, size=(grid_size_height, grid_size_width),
                                    mode="bicubic", align_corners=False)
        # (1, D, H', W') -> (1, H'*W', D)
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, grid_size_height * grid_size_width, -1)
        # tile across time
        posemb_grid = posemb_grid.repeat(1, num_frames, 1)  # (1, T*H'*W', D)
        # concat CLS back
        return torch.cat([posemb_tok, posemb_grid], dim=1)

    def forward_features(self, x, get_all=False, get_attn=False):
        # x: (B, C, T, H, W)
        B = x.shape[0]
        T = x.shape[2]  # Number of frames

        # Patch Embedding
        # x will be (B, T, num_patches, embed_dim)
        x = self.patch_embed(x)
        num_patches = x.shape[2]  # Number of spatial patches per frame

        # Add CLS token
        # cls_tokens: (B, 1, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # Reshape x to (B, T*num_patches, embed_dim) for adding position embed
        x = x.reshape(B, T * num_patches, self.embed_dim)

        # Concatenate CLS token with flattened patch tokens for positional embedding
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1 + T*num_patches, embed_dim)

        # Add position embeddings (applies to CLS + all spatial patches)
        # Handle resizing pos_embed if inference input size doesn't match pretrain
        # original_pos_embed_length is 1 (CLS) + num_spatial_patches_per_original_frame (e.g., 197)
        original_pos_embed_length = self.pos_embed.size(1)

        # current_full_sequence_length is 1 (CLS) + (num_spatial_patches_per_current_frame * T_frames) (e.g., 3137)
        current_full_sequence_length = x.size(1)

        if current_full_sequence_length != original_pos_embed_length:
            warnings.warn("Positional embedding size mismatch. Resizing dynamically. Performance may be impacted.")
            # Call the modified _resize_pos_embed function with the new spatial dimensions AND num_frames (T)
            new_pos_embed = self._resize_pos_embed(
                self.pos_embed,
                self.patch_embed.num_patches_h,  # Target spatial height of patches
                self.patch_embed.num_patches_w,  # Target spatial width of patches
                T  # Pass the current number of frames
            )
            x = x + new_pos_embed
        else:
            # If lengths match (e.g., if T=1 and spatial dimensions match original), directly add
            x = x + self.pos_embed

        x = self.pos_drop(x)

        # --- Add Time Embeddings (with T-adaptive handling) ---
        cls_tokens_after_pos = x[:, :1, :]
        patch_tokens_after_pos = x[:, 1:, :]  # (B, T*num_patches, D)

        patch_tokens_reshaped = patch_tokens_after_pos.reshape(B, T, num_patches, self.embed_dim)

        # Adapt temporal embedding to runtime T if needed
        te = self.temporal_embed  # (1, F, D)
        if te.shape[1] != T:
            te = F.interpolate(te.transpose(1, 2), size=T, mode='linear', align_corners=False).transpose(1, 2)

        temporal_embed_expanded = te.unsqueeze(2).expand(-1, -1, num_patches, -1)
        patch_tokens_with_time = (patch_tokens_reshaped + temporal_embed_expanded).reshape(B, T * num_patches,
                                                                                           self.embed_dim)
        patch_tokens_with_time = self.time_drop(patch_tokens_with_time)

        x = torch.cat((cls_tokens_after_pos, patch_tokens_with_time), dim=1)  # (B, 1 + T*num_patches, embed_dim)

        # Pass through Transformer Blocks
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(lambda inp: blk(inp, B, T, self.patch_embed.num_patches_w), x)
            else:
                x = blk(x, B, T, self.patch_embed.num_patches_w)

        x = self.norm(x)
        if get_all:  # Return all tokens (CLS + patch tokens)
            return x
        else:  # Return only CLS token (for classification, though I don't use it directly)
            return x[:, 0]

    def forward(self, x, use_head=False):
        x = self.forward_features(x)
        if use_head:  # This is for default classification head, not the custom one
            x = self.head(x)
        return x

    def get_intermediate_layers(self, x, n=1, return_patch_tokens=False):
        """
        Return the last n hidden states (after norm) as a list.
        If return_patch_tokens=False => return CLS tokens only (B, D) per layer.
        If True => return full tokens (B, 1 + T*H*W, D) per layer.
        """
        self.eval()  # typical behavior; you can remove if you prefer training-mode extraction
        B, C, T, H, W = x.shape

        # --- same pre-processing as forward_features ---
        x = self.patch_embed(x)  # (B, T, N, D)
        num_patches = x.shape[2]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = x.reshape(B, T * num_patches, self.embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1 + T*N, D)

        original_pos_embed_length = self.pos_embed.size(1)
        current_full_sequence_length = x.size(1)
        if current_full_sequence_length != original_pos_embed_length:
            new_pos_embed = self._resize_pos_embed(
                self.pos_embed,
                self.patch_embed.num_patches_h,
                self.patch_embed.num_patches_w,
                T
            )
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed

        x = self.pos_drop(x)

        # time embedding (T-adaptive)
        cls_tokens_after_pos = x[:, :1, :]
        patch_tokens_after_pos = x[:, 1:, :]  # (B, T*N, D)
        patch_tokens_reshaped = patch_tokens_after_pos.reshape(B, T, num_patches, self.embed_dim)
        te = self.temporal_embed
        if te.shape[1] != T:
            te = F.interpolate(te.transpose(1, 2), size=T, mode='linear', align_corners=False).transpose(1, 2)
        temporal_embed_expanded = te.unsqueeze(2).expand(-1, -1, num_patches, -1)
        patch_tokens_with_time = (patch_tokens_reshaped + temporal_embed_expanded).reshape(B, T * num_patches,
                                                                                           self.embed_dim)
        patch_tokens_with_time = self.time_drop(patch_tokens_with_time)
        x = torch.cat((cls_tokens_after_pos, patch_tokens_with_time), dim=1)

        # --- transformer blocks, collect last n ---
        collected = []
        last_k = max(1, int(n))
        for i, blk in enumerate(self.blocks):
            x = blk(x, B, T, self.patch_embed.num_patches_w)
            # collect if in last n
            if i >= len(self.blocks) - last_k:
                xs = self.norm(x)
                if not return_patch_tokens:
                    xs = xs[:, 0]  # CLS only (B, D)
                collected.append(xs)
        return collected

    def get_last_selfattention(self, x):
        """
        Returns the spatial self-attention maps from the LAST block.
        Shape: (B, T, num_heads, L, L), where L = 1 + H*W if CLS is present, else H*W.
        """
        self.eval()
        B, C, T, H, W = x.shape

        # --- same pre-processing as forward_features up to the blocks ---
        x = self.patch_embed(x)  # (B, T, N, D)
        num_patches = x.shape[2]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = x.reshape(B, T * num_patches, self.embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1 + T*N, D)

        original_pos_embed_length = self.pos_embed.size(1)
        current_full_sequence_length = x.size(1)
        if current_full_sequence_length != original_pos_embed_length:
            new_pos_embed = self._resize_pos_embed(
                self.pos_embed,
                self.patch_embed.num_patches_h,
                self.patch_embed.num_patches_w,
                T
            )
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed

        x = self.pos_drop(x)

        # time embedding (T-adaptive)
        cls_tokens_after_pos = x[:, :1, :]
        patch_tokens_after_pos = x[:, 1:, :]
        patch_tokens_reshaped = patch_tokens_after_pos.reshape(B, T, num_patches, self.embed_dim)
        te = self.temporal_embed
        if te.shape[1] != T:
            te = F.interpolate(te.transpose(1, 2), size=T, mode='linear', align_corners=False).transpose(1, 2)
        temporal_embed_expanded = te.unsqueeze(2).expand(-1, -1, num_patches, -1)
        patch_tokens_with_time = (patch_tokens_reshaped + temporal_embed_expanded).reshape(B, T * num_patches,
                                                                                           self.embed_dim)
        patch_tokens_with_time = self.time_drop(patch_tokens_with_time)
        x = torch.cat((cls_tokens_after_pos, patch_tokens_with_time), dim=1)

        # --- run all blocks except the last normally ---
        for blk in self.blocks[:-1]:
            x = blk(x, B, T, self.patch_embed.num_patches_w)

        # --- last block: return attention instead of tokens ---
        attn = self.blocks[-1](x, B, T, self.patch_embed.num_patches_w, return_attn=True)
        # attn shape from Attention: ((B*T), heads, L, L) where L = 1 + H*W
        num_heads = attn.shape[1]
        L = attn.shape[-1]
        attn = attn.reshape(B, T, num_heads, L, L)
        return attn
