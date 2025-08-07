import os
import torch
import math
import torch.nn as nn
from functools import partial
from transformers import AutoTokenizer, CLIPTextModel
import torch.nn.functional as F

from project_config import config
from backbone.vision_transformer import VisionTransformer
from backbone.vision_transformer import load_pretrained, _cfg, _conv_filter



# --- Conceptual LoRA Implementation ---
class LoRALinear(nn.Linear):
    def __init__(self, linear_layer, r=8, alpha=16, dropout=0.0):
        super().__init__(linear_layer.in_features, linear_layer.out_features, linear_layer.bias is not None)
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        self.original_weight = linear_layer.weight
        self.original_weight.requires_grad = False
        if linear_layer.bias is not None:
            self.original_bias = linear_layer.bias
            self.original_bias.requires_grad = False
        else:
            self.original_bias = None
        self.lora_A = nn.Parameter(torch.zeros(linear_layer.in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, linear_layer.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        original_output = F.linear(x, self.original_weight, self.original_bias)
        lora_output = (self.dropout(x @ self.lora_A) @ self.lora_B) * (self.alpha / self.r)
        return original_output + lora_output


def apply_lora_to_linear_layers(model, r=8, alpha=16, dropout=0.0):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LoRALinear(module, r, alpha, dropout))
        else:
            apply_lora_to_linear_layers(module, r, alpha, dropout)

class ConfidenceAwareFusion(nn.Module):
    """
    Calculates a confidence score for each frame based on its semantic
    relevance to the text query, as inspired by VTD-CLIP.
    """
    def __init__(self, embed_dim):
        super().__init__()
        # A small network to predict the confidence score from combined features
        self.confidence_network = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid() # Squeezes the output to a score between 0 and 1
        )

    def forward(self, visual_features, text_features_expanded):
        # visual_features: (B*T, N_patches, C)
        # text_features_expanded: (B*T, L, C)

        # Create a single summary vector for all visual patches and all text tokens
        avg_visual = visual_features.mean(dim=1) # Shape: (B*T, C)
        avg_text = text_features_expanded.mean(dim=1) # Shape: (B*T, C)

        # Concatenate the summaries and compute the confidence score
        combined_features = torch.cat([avg_visual, avg_text], dim=1)
        confidence_scores = self.confidence_network(combined_features) # Shape: (B*T, 1)

        return confidence_scores

# --- Start Self-Contained Mamba Block Implementation ---
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            bias=True, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1, padding_mode='zeros'
        ).to(memory_format=torch.channels_last)
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2, bias=False)  # Simplified this line slightly
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.act = nn.SiLU()

        self.A_log = nn.Parameter(torch.log(torch.ones(self.d_inner, self.d_state)))
        self.B = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.C = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

    def ssm_scan(self, x):
        delta = F.softplus(self.dt_proj(x))
        A = -torch.exp(self.A_log.float())

        delta_A = torch.exp(delta.unsqueeze(-1) * A)
        delta_B_x = (delta.unsqueeze(-1) * self.B.unsqueeze(0)) * x.unsqueeze(-1)

        h = torch.zeros(x.size(0), self.d_inner, self.d_state, device=x.device)
        ys = []
        for i in range(x.size(1)):
            h = delta_A[:, i] * h + delta_B_x[:, i]

            y = (h * self.C).sum(dim=-1)
            ys.append(y)

        y = torch.stack(ys, dim=1)

        return y + x * self.D

    def forward(self, x):

        (x_proj, res) = self.in_proj(x).split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x_proj_transposed = x_proj.transpose(1, 2)

        x_conv_transposed = self.conv1d(x_proj_transposed)

        x_conv_transposed = x_conv_transposed[:, :, :x.size(1)]

        x_conv = self.act(x_conv_transposed).transpose(1, 2)

        x_ssm = self.ssm_scan(x_conv)
        x_out = self.out_proj(x_ssm * self.act(res))
        return x_out


# --- End Self-Contained Mamba Block ---

class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL.TEXT_ENCODER_MODEL)
        self.text_encoder = CLIPTextModel.from_pretrained(config.MODEL.TEXT_ENCODER_MODEL)
        self.embed_dim = self.text_encoder.config.hidden_size
        if config.TRAIN.USE_PEFT:
            apply_lora_to_linear_layers(self.text_encoder, r=config.TRAIN.LORA_R, alpha=config.TRAIN.LORA_ALPHA,
                                        dropout=config.TRAIN.LORA_DROPOUT)
            print("Conceptual LoRA applied to Text Encoder.")

    def forward(self, input_ids, attention_mask):
        outputs = self.text_encoder(input_ids=input_ids.to(self.text_encoder.device),
                                    attention_mask=attention_mask.to(self.text_encoder.device))
        return outputs.last_hidden_state, attention_mask


class LanguageGuidedHead(nn.Module):
    def __init__(self, visual_embed_dim, text_embed_dim, num_attention_heads=8, num_layers=2):
        super().__init__()
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=visual_embed_dim, nhead=num_attention_heads,
                                                                    dim_feedforward=visual_embed_dim * 4,
                                                                    batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_layers)
        self.fc_relevance = nn.Linear(visual_embed_dim, 1)
        self.text_proj = nn.Linear(text_embed_dim, visual_embed_dim)

        if config.MODEL.USE_CONFIDENCE_FUSION:
            self.confidence_module = ConfidenceAwareFusion(visual_embed_dim)
        else:
            self.confidence_module = None

    def forward(self, visual_features, text_features, text_attention_mask):
        B_T, _, C_visual = visual_features.shape
        B_original, L_text, _ = text_features.shape
        T_frames = B_T // B_original

        text_features_proj = self.text_proj(text_features)
        text_features_expanded = text_features_proj.unsqueeze(1).expand(-1, T_frames, -1, -1).reshape(B_T, L_text,
                                                                                                      C_visual)
        text_mask_expanded = text_attention_mask.unsqueeze(1).expand(-1, T_frames, -1).reshape(B_T, L_text)

        if self.confidence_module is not None:
            confidence_scores = self.confidence_module(visual_features, text_features_expanded)
            visual_features = visual_features * confidence_scores.unsqueeze(1)

        # This is the full spatial feature map (needed for optical flow)
        fused_spatial_features = self.transformer_decoder(
            tgt=visual_features,
            memory=text_features_expanded,
            memory_key_padding_mask=~text_mask_expanded.bool()
        )

        # This is the averaged semantic feature vector (for the temporal head)
        semantic_features_for_temporal_head = fused_spatial_features.mean(dim=1)

        # === THIS IS THE FIX: Return the correct features ===
        # Return both the semantic vector for the temporal head AND the full spatial map for the loss function
        return semantic_features_for_temporal_head, fused_spatial_features, None

# --- Kept Original Temporal Head ---
class TemporalHead(nn.Module):
    def __init__(self, input_dim, output_dim, num_attention_heads=8, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_attention_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # The output layer's dimension is now controlled by the 'output_dim' parameter
        self.fc_output = nn.Linear(input_dim, output_dim)

    def forward(self, frame_features_seq):
        temporal_features = self.transformer_encoder(frame_features_seq)
        # The output can now be a single score or 4 evidential parameters
        logits = self.fc_output(temporal_features)

        # If output_dim is 4, we are in uncertainty mode. Apply activation.
        if self.fc_output.out_features == 4:
            # Softplus ensures non-negative evidence. Adding 1 for numerical stability.
            return F.softplus(logits)
        else:
            return logits

# --- SSM Temporal Head ---
class TemporalHeadSSM(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([MambaBlock(d_model=input_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(input_dim)
        # The output layer's dimension is now controlled by the 'output_dim' parameter
        self.fc_output = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x  # Residual connection
        x = self.norm(x)
        logits = self.fc_output(x)

        # If output_dim is 4, we are in uncertainty mode. Apply activation.
        if self.fc_output.out_features == 4:
            return F.softplus(logits)
        else:
            return logits


class LocalizationFramework(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.MODEL.VISION_BACKBONE_NAME == 'M2CRL':
            print("Initializing Vision Backbone: M2CRL (TimeSformer)")
            self.vision_backbone = VisionTransformer(
                img_size=config.DATA.TRAIN_CROP_SIZE, patch_size=16, in_chans=3,
                num_classes=0, embed_dim=768, depth=12, num_heads=12,
                mlp_ratio=4., qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                num_frames=config.DATA.NUM_FRAMES,
                attention_type=config.TIMESFORMER.ATTENTION_TYPE
            )
            if config.MODEL.M2CRL_WEIGHTS_PATH and os.path.exists(config.MODEL.M2CRL_WEIGHTS_PATH):
                load_pretrained(
                    self.vision_backbone, cfg=_cfg(), num_classes=0,
                    img_size=config.DATA.TRAIN_CROP_SIZE,
                    num_frames=config.DATA.NUM_FRAMES,
                    num_patches=self.vision_backbone.patch_embed.num_patches,
                    attention_type=config.TIMESFORMER.ATTENTION_TYPE,
                    pretrained_model=config.MODEL.M2CRL_WEIGHTS_PATH
                )
                print(f"Loaded pretrained M2CRL weights from {config.MODEL.M2CRL_WEIGHTS_PATH}")
        elif config.MODEL.VISION_BACKBONE_NAME == 'EndoMamba':
            # This part is a placeholder
            print("Initializing Vision Backbone: EndoMamba (Conceptual)")
            self.vision_backbone = VisionTransformer(img_size=config.DATA.TRAIN_CROP_SIZE,
                                                     num_frames=config.DATA.NUM_FRAMES)
        else:
            raise ValueError(f"Unknown VISION_BACKBONE_NAME: {config.MODEL.VISION_BACKBONE_NAME}")

        self.vision_embed_dim = self.vision_backbone.embed_dim
        self.text_encoder = TextEncoder(config)
        self.language_guided_head = LanguageGuidedHead(visual_embed_dim=self.vision_embed_dim,
                                                       text_embed_dim=self.text_encoder.embed_dim,
                                                       num_attention_heads=config.MODEL.HEAD_NUM_ATTENTION_HEADS,
                                                       num_layers=config.MODEL.HEAD_NUM_LAYERS)

        output_dim = 4 if config.MODEL.USE_UNCERTAINTY else 1
        if config.MODEL.TEMPORAL_HEAD_TYPE == 'SSM':
            self.temporal_head = TemporalHeadSSM(input_dim=self.vision_embed_dim, output_dim=output_dim,
                                                 num_layers=config.MODEL.HEAD_NUM_LAYERS)
        else:  # 'TRANSFORMER'
            self.temporal_head = TemporalHead(input_dim=self.vision_embed_dim, output_dim=output_dim,
                                              num_attention_heads=config.MODEL.HEAD_NUM_ATTENTION_HEADS,
                                              num_layers=config.MODEL.HEAD_NUM_LAYERS)

    def forward(self, video_clip, input_ids, attention_mask):
        B, _, T, H, W = video_clip.shape
        num_patches_h = H // self.vision_backbone.patch_embed.patch_size[0]
        num_patches_w = W // self.vision_backbone.patch_embed.patch_size[1]

        all_visual_tokens = self.vision_backbone.forward_features(video_clip, get_all=True)
        visual_features_for_head = all_visual_tokens[:, 1:, :].reshape(B * T, -1, self.vision_embed_dim)

        text_features, _ = self.text_encoder(input_ids, attention_mask)

        semantic_features_for_temporal, spatial_features_for_loss, xai_weights = self.language_guided_head(
            visual_features_for_head,
            text_features,
            attention_mask
        )

        semantic_features_reshaped = semantic_features_for_temporal.reshape(B, T, self.vision_embed_dim)

        if spatial_features_for_loss is not None:
            spatial_features_reshaped = spatial_features_for_loss.reshape(
                B, T, num_patches_h, num_patches_w, self.vision_embed_dim
            )
        else:
            spatial_features_reshaped = None

        raw_relevance_scores = self.language_guided_head.fc_relevance(semantic_features_for_temporal).reshape(B, T, 1)
        final_output = self.temporal_head(semantic_features_reshaped)

        if self.config.MODEL.USE_UNCERTAINTY:
            # The four evidential parameters are the final output for the loss function.
            evidential_params = final_output

            # The single 'refined_score' is for validation/inference, not the primary loss here.
            alpha = final_output[..., 0:1] + 1
            beta = final_output[..., 1:2] + 1
            refined_scores_for_val_and_unpacker = alpha / (alpha + beta)

            # Return the tuple in the order the MasterLoss expects.
            # (refined_score_placeholder, raw_score, unused, semantic, spatial, EVIDENTIAL_PARAMS)
            return (refined_scores_for_val_and_unpacker, raw_relevance_scores, None,
                    semantic_features_reshaped, spatial_features_reshaped, evidential_params)
        else:
            # When uncertainty is off, the final output IS the refined score.
            refined_scores = final_output

            # The evidential_output is None.
            return (refined_scores, raw_relevance_scores, None,
                    semantic_features_reshaped, spatial_features_reshaped, None)
