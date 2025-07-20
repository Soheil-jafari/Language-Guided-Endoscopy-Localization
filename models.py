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
from backbone.endomamba import EndoMambaBackbone



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
            bias=True, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, self.d_model + 2 * self.d_state, bias=False)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.act = nn.SiLU()
        self.A_log = nn.Parameter(torch.log(torch.ones(self.d_inner, self.d_state)))
        self.B = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.C = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.dt_proj = nn.Linear(self.d_model, self.d_inner, bias=True)

    def ssm_scan(self, x):
        delta = F.softplus(self.dt_proj(x))
        A = -torch.exp(self.A_log.float())
        delta_A = torch.exp(delta.unsqueeze(-1) * A)
        delta_B_x = (delta.unsqueeze(-1) * self.B.unsqueeze(0)) * x.unsqueeze(-1)
        h = torch.zeros(x.size(0), self.d_inner, self.d_state, device=x.device)
        ys = []
        for i in range(x.size(1)):
            h = delta_A[:, i] * h + delta_B_x[:, i]
            y = (h @ self.C.unsqueeze(-1)).squeeze(-1)
            ys.append(y)
        return torch.stack(ys, dim=1) + x * self.D

    def forward(self, x):
        (x_proj, res) = self.in_proj(x).split(split_size=[self.d_inner, self.d_inner], dim=-1)
        x_conv = self.act(self.conv1d(x_proj.transpose(1, 2))).transpose(1, 2)
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

    def forward(self, visual_features, text_features, text_attention_mask):
        B_T, _, C_visual = visual_features.shape
        B_original, L_text, _ = text_features.shape
        T_frames = B_T // B_original
        text_features_expanded = self.text_proj(text_features).unsqueeze(1).expand(-1, T_frames, -1, -1).reshape(B_T,
                                                                                                                 L_text,
                                                                                                                 C_visual)
        text_mask_expanded = text_attention_mask.unsqueeze(1).expand(-1, T_frames, -1).reshape(B_T, L_text)
        fused_features = self.transformer_decoder(tgt=visual_features, memory=text_features_expanded,
                                                  memory_key_padding_mask=~text_mask_expanded.bool())
        frame_features_for_temporal_head = fused_features.mean(dim=1)
        return frame_features_for_temporal_head, None  # None is placeholder for XAI weights


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

# --- New SSM Temporal Head ---
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
                    num_patches=self.vision_backbone.patch_embed.num_patches,  # This was missing
                    attention_type=config.TIMESFORMER.ATTENTION_TYPE,
                    pretrained_model=config.MODEL.M2CRL_WEIGHTS_PATH
                )
                print(f"Loaded pretrained M2CRL weights from {config.MODEL.M2CRL_WEIGHTS_PATH}")

        elif config.MODEL.VISION_BACKBONE_NAME == 'EndoMamba':
            print("Initializing Vision Backbone: EndoMamba")
            self.vision_backbone = EndoMambaBackbone(config)

            if config.TRAIN.USE_LORA_BACKBONE:
                apply_lora_to_linear_layers(
                    self.vision_backbone,
                    r=config.TRAIN.LORA_R_BACKBONE,
                    alpha=config.TRAIN.LORA_ALPHA_BACKBONE,
                    dropout=config.TRAIN.LORA_DROPOUT_BACKBONE
                )
                print("Conceptual LoRA applied to Vision Backbone (EndoMamba)!")

        else:
            raise ValueError(f"Unknown VISION_BACKBONE_NAME: {config.MODEL.VISION_BACKBONE_NAME}")

        self.vision_embed_dim = self.vision_backbone.embed_dim
        # 2. Text Encoder
        self.text_encoder = TextEncoder(config)
        self.text_embed_dim = self.text_encoder.embed_dim
        # 3. Language-Guided Head
        self.language_guided_head = LanguageGuidedHead(visual_embed_dim=self.vision_embed_dim,
                                                       text_embed_dim=self.text_embed_dim,
                                                       num_attention_heads=config.MODEL.HEAD_NUM_ATTENTION_HEADS,
                                                       num_layers=config.MODEL.HEAD_NUM_LAYERS)

        # 4. SELECTABLE Temporal Head with UNCERTAINTY
        # Determine the output dimension based on the uncertainty setting
        output_dim = 4 if config.MODEL.USE_UNCERTAINTY else 1

        if config.MODEL.TEMPORAL_HEAD_TYPE == 'SSM':
            print(
                f"Initializing SSM Temporal Head with output_dim={output_dim} (Uncertainty: {config.MODEL.USE_UNCERTAINTY})")
            self.temporal_head = TemporalHeadSSM(input_dim=self.vision_embed_dim, output_dim=output_dim,
                                                 num_layers=config.MODEL.HEAD_NUM_LAYERS)
        elif config.MODEL.TEMPORAL_HEAD_TYPE == 'TRANSFORMER':
            print(
                f"Initializing Transformer Temporal Head with output_dim={output_dim} (Uncertainty: {config.MODEL.USE_UNCERTAINTY})")
            self.temporal_head = TemporalHead(input_dim=self.vision_embed_dim, output_dim=output_dim,
                                              num_attention_heads=config.MODEL.HEAD_NUM_ATTENTION_HEADS,
                                              num_layers=config.MODEL.HEAD_NUM_LAYERS)
        else:
            raise ValueError(f"Unknown TEMPORAL_HEAD_TYPE: {config.MODEL.TEMPORAL_HEAD_TYPE}")

    def forward(self, video_clip, input_ids, attention_mask):
        B_original, _, T_frames, _, _ = video_clip.shape
        # ... (forward pass for backbone, text encoder, language head remains the same) ...
        all_tokens = self.vision_backbone.forward_features(video_clip, get_all=True)
        visual_features_for_head = all_tokens[:, 1:, :].reshape(B_original * T_frames, -1, self.vision_embed_dim)

        text_features, _ = self.text_encoder(input_ids, attention_mask)

        frame_features, xai_weights = self.language_guided_head(visual_features_for_head, text_features, attention_mask)
        frame_features_reshaped = frame_features.reshape(B_original, T_frames, self.vision_embed_dim)

        raw_relevance_scores = self.language_guided_head.fc_relevance(frame_features).reshape(B_original, T_frames, 1)

        # 4. Temporal Context Modeling
        # This will return either (B, T, 1) or (B, T, 4) depending on the config
        final_output = self.temporal_head(frame_features_reshaped)

        # The 'refined_scores' are now the first part of the model output.
        # When not using uncertainty, final_output is refined_scores.
        # When using uncertainty, we still need a primary score for evaluation.
        # We can calculate it from the evidential parameters (alpha / (alpha + beta)).
        if self.config.MODEL.USE_UNCERTAINTY:
            alpha = final_output[..., 0:1] + 1
            beta = final_output[..., 1:2] + 1
            refined_scores = alpha / (alpha + beta)
            return final_output, refined_scores, raw_relevance_scores, xai_weights
        else:
            # If not using uncertainty, the output is just the scores.
            refined_scores = final_output
            return refined_scores, raw_relevance_scores, xai_weights