import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """
    Converts a 2D image into a sequence of 1D patch embeddings.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        # A convolution layer that performs the patching and embedding in one step.
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Input x: [Batch, Channels, Height, Width]
        x = self.proj(x)  # [B, embed_dim, grid_size, grid_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class TransformerEncoderBlock(nn.Module):
    """
    A standard Transformer Encoder block (Self-Attention -> MLP).
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        # Standard multi-head self-attention.
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        # Standard MLP (Feed-Forward) network.
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Standard transformer block forward pass.
        residual = x
        x_norm = self.norm1(x)
        # NOTE: The second and third arguments are the key and value. For self-attention, they are the same as the query.
        attended_x, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = residual + attended_x

        residual = x
        x_norm = self.norm2(x)
        x = residual + self.mlp(x_norm)
        return x


class VisionTransformer(nn.Module):
    """
    A Vision Transformer that acts as a feature extractor.
    It takes an image or a video clip and returns a sequence of feature vectors.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        # A learnable token that will represent the entire image/clip (classification token).
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Learnable positional embeddings for each patch and the CLS token.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)

        # The stack of Transformer Encoder blocks.
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        The forward pass for the backbone.
        Args:
            x (torch.Tensor): Input tensor. Can be a batch of images [B, C, H, W]
                              or a batch of video clips [B, T, C, H, W].
        Returns:
            torch.Tensor: The feature map of shape [B, num_patches, embed_dim].
                          This is the grid of visual features our Language-Guided Head will use.
        """
        is_video_clip = x.ndim == 5
        if is_video_clip:
            B, T, C, H, W = x.shape
            # Reshape the video clip into a batch of frames.
            x = x.reshape(B * T, C, H, W)

        # 1. Patch & Embed
        x = self.patch_embed(x)  # [B*T, num_patches, embed_dim]

        # 2. Prepend CLS token and add Positional Embeddings
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B*T, 1 + num_patches, embed_dim]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 3. Pass through Transformer Encoder blocks
        for blk in self.blocks:
            x = blk(x)

        # 4. Apply final layer normalization
        x = self.norm(x)

        # 5. Get the patch features (ignoring the CLS token)
        patch_features = x[:, 1:, :]  # [B*T, num_patches, embed_dim]

        if is_video_clip:
            # If input was a video, average the features across the time dimension.
            # This provides a single, stable feature map for the entire clip.
            patch_features = patch_features.view(B, T, self.num_patches, -1)
            patch_features = patch_features.mean(dim=1)  # [B, num_patches, embed_dim]

        return patch_features
