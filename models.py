import os
import torch
import math
import torch.nn as nn
from functools import partial
from transformers import AutoTokenizer, CLIPTextModel
import torch.nn.functional as F

from project_config import config
from backbone.vision_transformer import VisionTransformer, load_pretrained, _cfg, _conv_filter



# --- Conceptual LoRA Implementation ---
class LoRALinear(nn.Module):
    def __init__(self, linear_layer, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("original_weight", linear_layer.weight.data.clone())
        if linear_layer.bias is not None:
            self.register_buffer("original_bias", linear_layer.bias.data.clone())
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

def apply_lora_to_linear_layers_selective(module, r=8, alpha=16, dropout=0.0, name_filter=None, _prefix=""):
    """
    Recursively wraps ONLY selected nn.Linear layers with LoRA.
    - module: root nn.Module to traverse (e.g., vision backbone)
    - name_filter: callable(full_name:str) -> bool
    """
    for child_name, child in module.named_children():
        full_name = f"{_prefix}.{child_name}" if _prefix else child_name

        if isinstance(child, nn.Linear) and (name_filter is None or name_filter(full_name)):
            setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
        else:
            apply_lora_to_linear_layers_selective(child, r=r, alpha=alpha, dropout=dropout,
                                                  name_filter=name_filter, _prefix=full_name)


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

# --- Self-Contained Mamba Block Implementation ---
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
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2, bias=False)
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


# --- Self-Contained Mamba Block ---

class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL.TEXT_ENCODER_MODEL)
        self.text_encoder = CLIPTextModel.from_pretrained(config.MODEL.TEXT_ENCODER_MODEL)
        self.embed_dim = self.text_encoder.config.hidden_size
        if config.TRAIN.USE_PEFT:
            # Restrict LoRA to the modules named in config (default: q_proj/v_proj),
            # instead of wrapping every Linear in the text encoder.
            target_modules = config.TRAIN.LORA_TARGET_MODULES
            apply_lora_to_linear_layers_selective(
                self.text_encoder, r=config.TRAIN.LORA_R, alpha=config.TRAIN.LORA_ALPHA,
                dropout=config.TRAIN.LORA_DROPOUT,
                name_filter=lambda full_name: any(t in full_name for t in target_modules)
            )
            print(f"LoRA applied to Text Encoder modules matching {target_modules}.")

    def forward(self, input_ids, attention_mask):
        dev = next(self.text_encoder.parameters()).device
        outputs = self.text_encoder(input_ids=input_ids.to(dev),
                                    attention_mask=attention_mask.to(dev))
        return outputs.last_hidden_state, attention_mask


class LanguageGuidedHead(nn.Module):
    def __init__(self, visual_embed_dim, text_embed_dim, num_attention_heads=8, num_layers=2):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=visual_embed_dim, nhead=num_attention_heads,
                                                   dim_feedforward=visual_embed_dim * 4,
                                                   batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # Per-patch relevance projection. It scores each frame by acting on the
        # mean-pooled fused feature (see LocalizationFramework.forward) and -- when
        # `return_xai_map` is enabled -- it is applied per patch to build the spatial
        # explainability map. Because it is linear,
        #     fc_relevance(mean_patches(F)) == mean_patches(fc_relevance(F)),
        # so the per-patch map is an exact additive decomposition of the frame's
        # relevance logit across space (a faithful attribution, not a post-hoc proxy).
        # Kept OFF during training (zero overhead); flip on for inference/visualization
        # via `model.language_guided_head.return_xai_map = True`.
        self.return_xai_map = False
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

        # Cross-attention fusion: visual patches (query) attend over the text tokens
        # (key/value). Output is the text-conditioned per-patch feature map, also used
        # downstream for the spatial (optical-flow) loss.
        fused_spatial_features = self.transformer_decoder(
            tgt=visual_features,
            memory=text_features_expanded,
            memory_key_padding_mask=~text_mask_expanded.bool()
        )

        # Averaged semantic feature vector (fed to the temporal head)
        semantic_features_for_temporal_head = fused_spatial_features.mean(dim=1)

        # Explainability map, computed on demand for inference/visualization.
        # We resolve the model's OWN relevance function spatially: apply fc_relevance to
        # every fused patch feature to get a per-patch relevance logit (B_T, N_patches).
        # Since the frame relevance is fc_relevance of the mean-pooled feature and
        # fc_relevance is linear, this map is an exact additive decomposition of that
        # relevance logit across space -- a faithful attribution rather than a proxy
        # derived from a single attention layer. It is NOT a softmax distribution, so
        # (unlike raw patch->text cross-attention averaged over the text axis) it retains
        # genuine spatial contrast instead of collapsing to a constant.
        xai_weights = None
        if self.return_xai_map:
            xai_weights = self.fc_relevance(fused_spatial_features).squeeze(-1)  # (B_T, N_patches)

        # Return the semantic vector, the full spatial map (for the loss), and the XAI map
        return semantic_features_for_temporal_head, fused_spatial_features, xai_weights

# --- Temporal Head ---
class TemporalHead(nn.Module):
    def __init__(self, input_dim, output_dim, num_attention_heads=8, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_attention_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # The output layer's dimension is controlled by the 'output_dim' parameter
        self.fc_output = nn.Linear(input_dim, output_dim)

    def forward(self, frame_features_seq):
        temporal_features = self.transformer_encoder(frame_features_seq)
        # The output can now be a single score or 4 evidential parameters
        logits = self.fc_output(temporal_features)

        # If output_dim is 2, we are in uncertainty mode.
        if self.fc_output.out_features == 2:
            # Softplus ensures non-negative evidence. Adding 1 for numerical stability.
            return F.softplus(logits)
        else:
            return logits

# --- SSM Temporal Head ---
class TemporalHeadSSM(nn.Module):
    """
    Mamba / state-space temporal head. It stacks `num_layers` Mamba mixers, each
    mapping a (B, T, D) sequence to (B, T, D), followed by a norm and an output
    projection.

    The mixer is selected at build time:
      * If `use_official_mamba` is True AND the `mamba_ssm` CUDA library imports
        successfully, the fast official `mamba_ssm.Mamba` is used.
      * Otherwise it falls back to the self-contained `MambaBlock` in this file,
        so the head works even when `mamba_ssm` is not installed.
    Both implementations expose the same (B, L, D) -> (B, L, D) interface.
    """

    def __init__(self, input_dim, output_dim, num_layers=4, use_official_mamba=True,
                 d_state=16, d_conv=4, expand=2):
        super().__init__()

        mixer_factory = None
        if use_official_mamba:
            try:
                from mamba_ssm import Mamba

                def mixer_factory():
                    return Mamba(d_model=input_dim, d_state=d_state, d_conv=d_conv, expand=expand)

                print("[SSM] TemporalHeadSSM using official mamba_ssm.Mamba mixer.")
            except Exception as e:
                mixer_factory = None
                print(f"[SSM] mamba_ssm unavailable ({type(e).__name__}: {e}); "
                      f"falling back to built-in MambaBlock.")

        if mixer_factory is None:
            def mixer_factory():
                return MambaBlock(d_model=input_dim, d_state=d_state, d_conv=d_conv, expand=expand)

            if use_official_mamba is False:
                print("[SSM] TemporalHeadSSM using built-in MambaBlock (official Mamba disabled by config).")

        self.layers = nn.ModuleList([mixer_factory() for _ in range(num_layers)])
        self.norm = nn.LayerNorm(input_dim)
        # The output layer's dimension is controlled by the 'output_dim' parameter
        self.fc_output = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x  # Residual connection
        x = self.norm(x)
        logits = self.fc_output(x)

        # If output_dim is 2, we are in uncertainty mode (Beta evidential params).
        if self.fc_output.out_features == 2:
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
                attention_type=config.TIMESFORMER.ATTENTION_TYPE,
                use_checkpoint=getattr(config.TRAIN, "USE_GRADIENT_CHECKPOINTING", False)
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

        if getattr(config.TRAIN, "USE_LORA_BACKBONE", False):
            # Wrap only attention (qkv/proj) + MLP (fc1/fc2) linear layers
            target_names = {"qkv", "proj", "fc1", "fc2"}
            apply_lora_to_linear_layers_selective(
                self.vision_backbone,
                r=config.TRAIN.LORA_R_BACKBONE,
                alpha=config.TRAIN.LORA_ALPHA_BACKBONE,
                dropout=config.TRAIN.LORA_DROPOUT_BACKBONE,
                name_filter=lambda full_name: any(t in full_name.lower() for t in target_names)
            )
            print("LoRA applied to Vision Backbone (qkv/proj/fc1/fc2).")

            if getattr(config.TRAIN, "FREEZE_BACKBONE_WHEN_LORA", True):
                # Freeze all non-LoRA params in the backbone; train only LoRA A/B
                for n, p in self.vision_backbone.named_parameters():
                    p.requires_grad = ("lora_A" in n) or ("lora_B" in n)
                print("Frozen non-LoRA backbone params (FREEZE_BACKBONE_WHEN_LORA=True).")

        self.vision_embed_dim = self.vision_backbone.embed_dim
        self.text_encoder = TextEncoder(config)
        self.language_guided_head = LanguageGuidedHead(visual_embed_dim=self.vision_embed_dim,
                                                       text_embed_dim=self.text_encoder.embed_dim,
                                                       num_attention_heads=config.MODEL.HEAD_NUM_ATTENTION_HEADS,
                                                       num_layers=config.MODEL.HEAD_NUM_LAYERS)

        output_dim = 2 if config.MODEL.USE_UNCERTAINTY else 1
        if config.MODEL.TEMPORAL_HEAD_TYPE == 'SSM':
            print("Initializing Temporal Head: SSM (Mamba)")
            self.temporal_head = TemporalHeadSSM(
                input_dim=self.vision_embed_dim, output_dim=output_dim,
                num_layers=getattr(config.MODEL, 'SSM_NUM_LAYERS', config.MODEL.HEAD_NUM_LAYERS),
                use_official_mamba=getattr(config.MODEL, 'SSM_USE_OFFICIAL_MAMBA', True),
                d_state=getattr(config.MODEL, 'SSM_D_STATE', 16),
                d_conv=getattr(config.MODEL, 'SSM_D_CONV', 4),
                expand=getattr(config.MODEL, 'SSM_EXPAND', 2),
            )
        else:  # 'TRANSFORMER'
            self.temporal_head = TemporalHead(input_dim=self.vision_embed_dim, output_dim=output_dim,
                                              num_attention_heads=config.MODEL.HEAD_NUM_ATTENTION_HEADS,
                                              num_layers=config.MODEL.HEAD_NUM_LAYERS)

    def forward(self, video_clip, input_ids, attention_mask, text_features=None):
        B, _, T, H, W = video_clip.shape
        num_patches_h = H // self.vision_backbone.patch_embed.patch_size[0]
        num_patches_w = W // self.vision_backbone.patch_embed.patch_size[1]

        all_visual_tokens = self.vision_backbone.forward_features(video_clip, get_all=True)
        visual_features_for_head = all_visual_tokens[:, 1:, :].reshape(B * T, -1, self.vision_embed_dim)

        # Allow precomputed text features (e.g. cached at inference, since the query is
        # fixed). Must be shaped (B, L_text, D_text). Falls back to encoding on the fly.
        if text_features is None:
            text_features, _ = self.text_encoder(input_ids, attention_mask)

        semantic_features_for_temporal, spatial_features_for_loss, xai_weights = self.language_guided_head(
            visual_features_for_head,
            text_features,
            attention_mask
        )

        if xai_weights is not None:
            # (B*T, N_patches) -> (B, T, num_patches_h, num_patches_w) spatial attention map
            xai_weights = xai_weights.reshape(B, T, num_patches_h, num_patches_w)

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
            return (refined_scores_for_val_and_unpacker, raw_relevance_scores, xai_weights,
                    semantic_features_reshaped, spatial_features_reshaped, evidential_params)
        else:
            # When uncertainty is off, the final output IS the refined score.
            refined_scores = final_output

            # The evidential_output is None.
            return (refined_scores, raw_relevance_scores, xai_weights,
                    semantic_features_reshaped, spatial_features_reshaped, None)

