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
import torch.hub  # Changed from torch.utils.model_zoo for load_state_dict_from_url
import torch.utils.checkpoint as checkpoint
from einops import rearrange  # Assuming einops is installed and available in your environment

# --- Constants and Utilities ---
# Using a basic logger instead of timm's _logger for simplicity
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


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            if v.shape[-1] != patch_size:
                patch_size = v.shape[-1]
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


# --- Helper function for loading pre-trained weights into VisionTransformer ---
# This version explicitly handles local file paths first, then URLs, then random init.
def load_pretrained(model, cfg=None, num_classes=1000, in_chans=3, filter_fn=None, img_size=224, num_frames=16,
                    num_patches=196, attention_type='divided_space_time', pretrained_model='', strict=True):
    if cfg is None:
        # Resolve config from model's default_cfg or a default if arch not found
        cfg = getattr(model, 'default_cfg', None) or default_cfgs.get('vit_base_patch16_224')
        if cfg is None:
            _logger.warning("Could not resolve default config for model and no explicit cfg was provided.")
            warnings.warn("No valid pretrained model found. Using random initialization.")
            return  # Fallback to random initialization early

    if filter_fn is None:
        filter_fn = _conv_filter

    # --- Priority 1: Local pretrained_model path ---
    if pretrained_model and os.path.exists(pretrained_model):
        _logger.info(f'Attempting to load pretrained weights from local file: {pretrained_model}')
        try:
            checkpoint = torch.load(pretrained_model, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']  # Assuming the common case of 'model' key
            else:
                state_dict = checkpoint  # Otherwise, assume it's the state_dict directly

            # Filter out classifier head if present and not needed (num_classes=0 means no head for fine-tuning)
            if num_classes == 0:
                for k in list(state_dict.keys()):
                    if 'head' in k:
                        del state_dict[k]

            # Apply filter function for patch embedding conversion if needed
            state_dict = filter_fn(state_dict, model.patch_size[0])

            model.load_state_dict(state_dict, strict=strict)
            _logger.info(f'Successfully loaded pretrained weights from {pretrained_model}')
            return  # Successfully loaded, exit function



        except Exception as e:
            _logger.error(f"Error loading local checkpoint '{pretrained_model}': {e}")  # ADD THIS LINE
            warnings.warn(
                f"Failed to load pretrained model from {pretrained_model}: {e}. Falling back to URL/random initialization.")

    # --- Priority 2: URL loading (if local path not provided or failed) ---
    if 'url' in cfg and cfg['url']:
        _logger.info(f'Attempting to load pretrained from URL: {cfg["url"]}')
        try:
            state_dict = torch.hub.load_state_dict_from_url(cfg['url'], map_location='cpu', progress=True)
            if 'model' in state_dict:  # Some checkpoints might be wrapped in a 'model' key
                state_dict = state_dict['model']

            if num_classes == 0:
                for k in list(state_dict.keys()):
                    if 'head' in k:
                        del state_dict[k]

            state_dict = filter_fn(state_dict, model.patch_size[0])

            model.load_state_dict(state_dict, strict=strict)
            _logger.info(f'Successfully loaded pretrained weights from URL: {cfg["url"]}')
            return  # Successfully loaded, exit function
        except Exception as e:
            warnings.warn(f"Failed to load pretrained model from URL {cfg['url']}: {e}. Using random initialization.")
            # If URL loading fails, then final fallback is random init.

    # --- Final Fallback: Random Initialization (if no local path or URL worked) ---
    warnings.warn(
        "No valid pretrained model found (local path not provided/failed, or URL invalid/failed). Using random initialization.")


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
            self.temporal_fc = nn.Linear(dim, dim)  # This layer will fuse temporal attention output

        # drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, B, T, W, return_attn=False):
        num_spatial_tokens = (x.size(1) - self.class_tokens) // T
        H = num_spatial_tokens // W  # H of the patch grid

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            # Divided Space Time Attention (as per M²CRL and TimeSformer)

            # Extract CLS token and patch tokens (save original for residual connection)
            if self.class_tokens == 1:
                cls_tokens_for_block_start = x[:, :1, :]  # (B, 1, D)
                patch_tokens_for_block_start = x[:, 1:, :]  # (B, T*num_patches, D)
            else:
                cls_tokens_for_block_start = None
                patch_tokens_for_block_start = x

            # --- Temporal Attention ---
            # Reshape patch_tokens_for_block_start for temporal attention: (B*num_patches, T, dim)
            xt_temp_reshaped = rearrange(patch_tokens_for_block_start, 'b (h w t) m -> (b h w) t m', b=B, h=H, w=W, t=T)

            # Apply temporal attention
            res_temporal_output = self.drop_path(self.temporal_attn(self.temporal_norm1(xt_temp_reshaped)))

            # Reshape temporal attention output back to (B, T*num_patches, dim)
            res_temporal_output = rearrange(res_temporal_output, '(b h w) t m -> b (h w t) m', b=B, h=H, w=W, t=T)

            # Apply linear fusion
            res_temporal_output = self.temporal_fc(res_temporal_output)

            # Add temporal residual to the ORIGINAL patch tokens (patch_tokens_for_block_start)
            patch_tokens_after_temporal = patch_tokens_for_block_start + res_temporal_output
            # This 'patch_tokens_after_temporal' is (B, T*num_patches, D)

            # --- Spatial Attention ---
            # Prepare CLS token for spatial attention across frames
            if cls_tokens_for_block_start is not None:
                # Repeat CLS token for each frame
                cls_token_spatial = cls_tokens_for_block_start.repeat(1, T, 1)  # (B, T, D)
                cls_token_spatial = rearrange(cls_token_spatial, 'b t m -> (b t) m').unsqueeze(1)  # (B*T, 1, D)
            else:
                cls_token_spatial = None

            # Reshape spatial patches for spatial attention: (B*T, num_patches, dim)
            # Use patch_tokens_after_temporal here
            xs = rearrange(patch_tokens_after_temporal, 'b (h w t) m -> (b t) (h w) m', b=B, h=H, w=W, t=T)

            # Concatenate CLS token if present
            if cls_token_spatial is not None:
                xs = torch.cat((cls_token_spatial, xs), 1)  # (B*T, 1 + num_patches, D)

            if return_attn:
                _, attn = self.attn(self.norm1(xs), return_attn=return_attn)
                return attn
            else:
                res_spatial_output = self.drop_path(self.attn(self.norm1(xs)))  # Spatial attention output

            # Extract and process CLS token after spatial attention
            if cls_tokens_for_block_start is not None:
                cls_token_out = res_spatial_output[:, 0, :]  # (B*T, D)
                cls_token_out = rearrange(cls_token_out, '(b t) m -> b t m', b=B, t=T)  # (B, T, D)
                cls_token_out = torch.mean(cls_token_out, 1, True)  # Average across time for final CLS token (B, 1, D)
                res_spatial_patches = res_spatial_output[:, 1:, :]  # Get only patches (B*T, num_patches, D)
            else:
                cls_token_out = None
                res_spatial_patches = res_spatial_output  # All were patches (B*T, num_patches, D)

            # Reshape spatial attention output back to original (B, T*num_patches, dim)
            res_spatial_patches = rearrange(res_spatial_patches, '(b t) (h w) m -> b (h w t) m', b=B, h=H, w=W, t=T)

            # Combine cls_tokens_out with res_spatial_patches for the block's MLP and final residual
            # This is the output of the two attention heads (temporal + spatial)
            if cls_tokens_for_block_start is not None:
                # The output of attention is the original input (x) + attention residual
                # For blocks, the overall residual connection (x + self.mlp(self.norm2(x))) is key.
                # Here, we combine the original CLS and patch tokens with their respective attention-processed residuals.
                attended_output = torch.cat((cls_token_out, res_spatial_patches), 1)  # (B, 1 + T*num_patches, D)
                x = x + self.drop_path(attended_output)  # Block's first residual
            else:
                x = x + self.drop_path(res_spatial_patches)  # Block's first residual (no CLS)

            # Apply MLP and its residual
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
        norm_layer = partial(nn.LayerNorm, eps=1e-6)  # Use functools.partial for default
        self.num_frames = num_frames
        self.patch_size = patch_size
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
        self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))  # Time embedding for each frame
        self.time_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attention_type=attention_type)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head (not used in our framework, but kept for compatibility with pretrained models)
        self.head = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.time_embed, std=.02)  # Initialize time embedding
        self.apply(self._init_weights)

        # Initialization of temporal attention weights' linear layers
        if self.attention_type == 'divided_space_time':
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Block' in m_str:
                    if hasattr(m, 'temporal_fc'):  # Only if temporal_fc exists
                        if i > 0:  # Apply to all but the first block if needed
                            nn.init.constant_(m.temporal_fc.weight, 0)
                            nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1

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
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _resize_pos_embed(self, posemb, grid_size_height, grid_size_width, num_frames,
                          start_index=1):
        posemb_tok = posemb[:, :start_index]
        posemb_grid = posemb[0, start_index:]
        old_grid_size = int(posemb_grid.size(0) ** 0.5)  # Use int() here as discussed

        posemb_grid = posemb_grid.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)
        posemb_grid = nn.functional.interpolate(
            posemb_grid,
            size=(grid_size_height, grid_size_width),
            mode="bicubic",  # Or "bilinear" if that's what you prefer
            align_corners=False
        )
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, grid_size_height * grid_size_width, -1)
        posemb_grid_expanded_for_time = posemb_grid.repeat(1, num_frames, 1)  # This is the temporal part
        posemb = torch.cat([posemb_tok, posemb_grid_expanded_for_time], dim=1)
        return posemb

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

        # Add Time Embeddings
        # Extract CLS token again to apply time embedding only to patch tokens
        cls_tokens_after_pos = x[:, :1, :]
        patch_tokens_after_pos = x[:, 1:, :]  # (B, T*num_patches, embed_dim)

        # Reshape patch_tokens to (B, T, num_patches, embed_dim) to apply time embed
        patch_tokens_reshaped = patch_tokens_after_pos.reshape(B, T, num_patches, self.embed_dim)

        # Expand time_embed to apply to all spatial patches within each frame
        # time_embed: (1, num_frames, embed_dim) -> (B, num_frames, 1, embed_dim)
        time_embed_expanded = self.time_embed.unsqueeze(2).expand(-1, -1, num_patches, -1)

        # Add time embedding and reshape back to (B, T*num_patches, embed_dim)
        patch_tokens_with_time = (patch_tokens_reshaped + time_embed_expanded).reshape(B, T * num_patches,
                                                                                       self.embed_dim)
        patch_tokens_with_time = self.time_drop(patch_tokens_with_time)

        # Recombine CLS token with time-embedded spatial patches
        x = torch.cat((cls_tokens_after_pos, patch_tokens_with_time), dim=1)  # (B, 1 + T*num_patches, embed_dim)

        # Pass through Transformer Blocks
        for blk in self.blocks:
            if self.use_checkpoint:
                # Assuming blk.forward takes (x, B, T, W) as per Block class, need to pass W from patch_embed
                x = checkpoint.checkpoint(blk, x, B, T, self.patch_embed.num_patches_w)
            else:
                x = blk(x, B, T, self.patch_embed.num_patches_w)  # Pass B, T, W to block

        x = self.norm(x)
        if get_all:  # Return all tokens (CLS + patch tokens)
            return x
        else:  # Return only CLS token (for classification, though we don't use it directly)
            return x[:, 0]

    def forward(self, x, use_head=False):
        x = self.forward_features(x)
        if use_head:  # This is for default classification head, not our custom one
            x = self.head(x)
        return x

    def get_intermediate_layers(self, x, n=1):
        raise NotImplementedError(
            "Intermediate layer extraction not fully implemented for this customized VisionTransformer.")

    def get_last_selfattention(self, x):
        raise NotImplementedError(
            "Attention map extraction not fully implemented for this customized VisionTransformer.")
