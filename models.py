import torch
import torch.nn as nn
from functools import partial
from transformers import AutoTokenizer, CLIPTextModel
import os
import torch.nn.functional as F

# Import the VisionTransformer class directly from the new, self-contained backbone file
from backbone.vision_transformer import VisionTransformer  # This now contains all necessary helpers


# --- Conceptual LoRA Implementation (You would typically use a library like Hugging Face's PEFT) ---
class LoRALinear(nn.Linear):
    def __init__(self, linear_layer, r=8, alpha=16, dropout=0.0):
        super().__init__(linear_layer.in_features, linear_layer.out_features, linear_layer.bias is not None)
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        self.original_weight = linear_layer.weight
        self.original_bias = linear_layer.bias
        self.lora_A = nn.Parameter(torch.zeros(linear_layer.in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, linear_layer.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        original_output = F.linear(x, self.original_weight, self.original_bias)
        lora_output = (self.dropout(x @ self.lora_A) @ self.lora_B) * (self.alpha / self.r)
        return original_output + lora_output


def apply_lora_to_linear_layers(model, r=8, alpha=16, dropout=0.0):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Only apply LoRA to layers that are typically targeted (e.g., Q, K, V projections, MLP layers)
            # You might want to refine this condition based on specific layer names
            setattr(model, name, LoRALinear(module, r, alpha, dropout))
        else:
            apply_lora_to_linear_layers(module, r, alpha, dropout)  # Recurse for nested modules


# --- End Conceptual LoRA Implementation ---


class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL.TEXT_ENCODER_MODEL)
        self.text_encoder = CLIPTextModel.from_pretrained(config.MODEL.TEXT_ENCODER_MODEL)
        self.embed_dim = self.text_encoder.config.hidden_size

        if config.TRAIN.USE_PEFT:
            apply_lora_to_linear_layers(self.text_encoder, r=config.TRAIN.PEFT_LORA_R,
                                        alpha=config.TRAIN.PEFT_LORA_ALPHA,
                                        dropout=config.TRAIN.PEFT_LORA_DROPOUT)
            print("Conceptual LoRA applied to Text Encoder!")

    def forward(self, text_queries):
        inputs = self.tokenizer(text_queries, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.text_encoder.device) for k, v in inputs.items()}
        outputs = self.text_encoder(**inputs)
        return outputs.last_hidden_state, inputs.attention_mask


class LanguageGuidedHead(nn.Module):
    def __init__(self, visual_embed_dim, text_embed_dim, output_dim, num_attention_heads=8, num_layers=2):
        super().__init__()
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=visual_embed_dim,
            nhead=num_attention_heads,
            dim_feedforward=visual_embed_dim * 4,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_layers)
        self.fc_relevance = nn.Linear(visual_embed_dim, output_dim)
        self.text_proj = nn.Linear(text_embed_dim, visual_embed_dim)

    def forward(self, visual_features, text_features, text_attention_mask):
        # visual_features: (B*T, N_patches, C_visual) - spatio-temporal patches flattened across batch and time
        # text_features: (B, L, C_text)
        # text_attention_mask: (B, L)

        B_T, N_patches, C_visual = visual_features.shape  # B_T is (Original_B * T_frames)

        # Expand text features and mask to match the (B*T) dimension of visual_features
        # Get original batch size (B) from text_features
        B_original = text_features.shape[0]
        # Infer T (number of frames per original batch item)
        T_frames = B_T // B_original

        # Project text features and expand to match (B*T) for cross-attention
        text_features_expanded = self.text_proj(text_features).unsqueeze(1).expand(-1, T_frames, -1, -1).reshape(B_T,
                                                                                                                 text_features.shape[
                                                                                                                     1],
                                                                                                                 C_visual)
        text_mask_expanded = text_attention_mask.unsqueeze(1).expand(-1, T_frames, -1).reshape(B_T,
                                                                                               text_attention_mask.shape[
                                                                                                   1])

        fused_features = self.transformer_decoder(
            tgt=visual_features,
            memory=text_features_expanded,
            memory_key_padding_mask=~text_mask_expanded.bool()
        )
        # Average across patches to get per-frame relevance score
        relevance_score = self.fc_relevance(fused_features.mean(dim=1))
        attention_weights_for_xai = None

        return relevance_score, attention_weights_for_xai


class TemporalHead(nn.Module):
    def __init__(self, input_dim, output_dim, num_attention_heads=8, num_layers=2):
        super().__init__()
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_attention_heads,
            dim_feedforward=input_dim * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.fc_output = nn.Linear(input_dim, output_dim)

    def forward(self, frame_relevance_scores_seq):
        # frame_relevance_scores_seq: (B, T, 1) or (B, T, some_feature_dim)
        temporal_fused_scores = self.transformer_encoder(frame_relevance_scores_seq)
        refined_scores = self.fc_output(temporal_fused_scores)
        return refined_scores


class LocalizationFramework(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Vision Backbone (Spatio-Temporal M²CRL)
        self.vision_backbone = VisionTransformer(
            img_size=config.DATA.TRAIN_CROP_SIZE,
            patch_size=16,  # Assuming patch size 16 based on vit_base_patch16_224
            in_chans=3,
            num_classes=0,  # No head needed as we extract features
            embed_dim=768,  # Base ViT embed dim
            depth=12,  # Base ViT depth
            num_heads=12,  # Base ViT num_heads
            mlp_ratio=4.,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            num_frames=config.DATA.NUM_FRAMES,  # Number of frames for the backbone
            attention_type=config.TIMESFORMER.ATTENTION_TYPE  # 'divided_space_time'
        )
        self.vision_embed_dim = self.vision_backbone.embed_dim

        # Load pretrained weights (from the helper function now included in vision_transformer.py)
        # Assuming load_pretrained is a static method or imported correctly
        # This calls the load_pretrained function which is now part of the backbone/vision_transformer.py scope
        # You might need to adjust `pretrained_model` to point to your M2CRL weights
        from backbone.vision_transformer import load_pretrained, _cfg, _conv_filter
        if config.MODEL.BACKBONE_WEIGHTS_PATH:
            load_pretrained(
                self.vision_backbone,
                cfg=_cfg(),  # Use a default config structure if not available from pretrained source
                num_classes=0,  # No classifier head needed
                img_size=config.DATA.TRAIN_CROP_SIZE,
                num_frames=config.DATA.NUM_FRAMES,
                num_patches=self.vision_backbone.patch_embed.num_patches,
                attention_type=config.TIMESFORMER.ATTENTION_TYPE,
                pretrained_model=config.MODEL.BACKBONE_WEIGHTS_PATH
            )
            print(f"Loaded pretrained weights for Vision Backbone from {config.MODEL.BACKBONE_WEIGHTS_PATH}")

        # Apply LoRA to Vision Backbone if configured
        if config.TRAIN.USE_LORA_BACKBONE:
            apply_lora_to_linear_layers(self.vision_backbone, r=config.TRAIN.LORA_R_BACKBONE,
                                        alpha=config.TRAIN.LORA_ALPHA_BACKBONE,
                                        dropout=config.TRAIN.LORA_DROPOUT_BACKBONE)
            print("Conceptual LoRA applied to Vision Backbone!")

        # 2. Text Encoder
        self.text_encoder = TextEncoder(config)
        self.text_embed_dim = self.text_encoder.embed_dim

        # 3. Language-Guided Head (Cross-Modal Fusion)
        self.language_guided_head = LanguageGuidedHead(
            visual_embed_dim=self.vision_embed_dim,  # Embed dim from the backbone
            text_embed_dim=self.text_embed_dim,
            output_dim=1,  # Outputs a single relevance score per patch/frame
            num_attention_heads=config.MODEL.HEAD_NUM_ATTENTION_HEADS,
            num_layers=config.MODEL.HEAD_NUM_LAYERS
        )

        # 4. Temporal Head
        # The input to this head is the sequence of (B, T, 1) relevance scores
        self.temporal_head = TemporalHead(
            input_dim=1,
            output_dim=1,
            num_attention_heads=config.MODEL.HEAD_NUM_ATTENTION_HEADS,
            num_layers=config.MODEL.HEAD_NUM_LAYERS
        )

    def forward(self, video_clip, text_query):
        # video_clip: (B, C, T, H, W)
        # text_query: list of strings

        # 1. Feature Extraction (Spatio-Temporal Backbone)
        # Use forward_features with get_all=True to get all tokens (CLS + patches)
        # all_tokens shape: (B, 1 + num_patches * T, embed_dim)
        all_tokens = self.vision_backbone.forward_features(video_clip, get_all=True)

        # Extract only the patch tokens (remove the CLS token)
        # patch_tokens shape: (B, num_patches * T, embed_dim)
        patch_tokens = all_tokens[:, 1:, :]

        B_original = video_clip.shape[0]  # Original batch size
        T_frames = video_clip.shape[2]  # Number of frames in the clip (from input)
        # Calculate num_patches_per_frame based on backbone's patch embedding
        num_patches_per_frame = self.vision_backbone.patch_embed.num_patches

        # Reshape patch_tokens to (B * T, num_patches_per_frame, embed_dim)
        # This prepares it for the LanguageGuidedHead's batch_first expectation
        visual_features_for_head = patch_tokens.reshape(B_original * T_frames, num_patches_per_frame,
                                                        self.vision_embed_dim)

        # 2. Text Encoding
        text_features, text_attention_mask = self.text_encoder(text_query)

        # 3. Spatial-Semantic Fusion (Language-Guided Head)
        raw_relevance_scores_flat, attention_weights_for_xai = self.language_guided_head(
            visual_features_for_head,
            text_features,
            text_attention_mask
        )
        # Reshape relevance scores back to (B, T, 1) for the temporal head
        raw_relevance_scores = raw_relevance_scores_flat.reshape(B_original, T_frames, 1)

        # 4. Temporal Context Modeling (Temporal Head)
        refined_scores = self.temporal_head(raw_relevance_scores)

        return refined_scores, raw_relevance_scores, attention_weights_for_xai