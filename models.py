# models.py
# This script defines the complete end-to-end model architecture.
# It integrates the Vision Transformer backbone, Text Encoder,
# a true Cross-Attention Language-Guided Head, and a Temporal Transformer Head.

import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model
import os
import sys

# Import our project's configuration and backbone
import config
from backbone.vision_transformer import VisionTransformer


# --- 1. Text Encoder (with PEFT) ---
class TextEncoder(nn.Module):
    """
    A pre-trained transformer-based text encoder (from CLIP).
    Adapted using PEFT (LoRA) for efficient fine-tuning.
    """

    def __init__(self):
        super().__init__()
        self.text_model = CLIPTextModel.from_pretrained(config.TEXT_MODEL_NAME)

        if config.USE_PEFT:
            lora_config = LoraConfig(
                r=config.LORA_R,
                lora_alpha=config.LORA_ALPHA,
                lora_dropout=config.LORA_DROPOUT,
                bias="none",
                target_modules=config.LORA_TARGET_MODULES,
            )
            self.text_model = get_peft_model(self.text_model, lora_config)
            print("PEFT (LoRA) enabled for Text Encoder.")
            self.text_model.print_trainable_parameters()
        else:
            for param in self.text_model.parameters():
                param.requires_grad = False
            print("Text Encoder is frozen (PEFT not enabled).")

    def forward(self, input_ids, attention_mask):
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


# --- 2. Language-Guided Head (True Cross-Attention) ---
class LanguageGuidedHead(nn.Module):
    """
    A true Cross-Modal Transformer that fuses visual patch features with text token features.
    """

    def __init__(self, visual_embed_dim=768, text_embed_dim=512, depth=config.FUSION_HEAD_DEPTH,
                 num_heads=config.FUSION_HEAD_NUM_HEADS):
        super().__init__()
        self.text_proj = nn.Linear(text_embed_dim, visual_embed_dim)
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=visual_embed_dim,
                nhead=num_heads,
                batch_first=True
            ) for _ in range(depth)
        ])
        self.relevance_head = nn.Linear(visual_embed_dim, 1)

    def forward(self, visual_features, text_features, text_attention_mask):
        text_memory = self.text_proj(text_features)
        fused_output = visual_features
        for layer in self.decoder_layers:
            fused_output = layer(tgt=fused_output, memory=text_memory,
                                 memory_key_padding_mask=(1 - text_attention_mask).bool())
        frame_embedding = fused_output.mean(dim=1)
        relevance_score = self.relevance_head(frame_embedding)
        attention_weights_for_xai = None
        return relevance_score, attention_weights_for_xai


# --- 3. Temporal Head (Temporal Transformer) ---
class TemporalHead(nn.Module):
    """
    A small Temporal Transformer that models the sequence of frame relevance scores.
    """

    def __init__(self, input_dim=1, model_dim=64, depth=config.TEMPORAL_HEAD_DEPTH,
                 num_heads=config.TEMPORAL_HEAD_NUM_HEADS):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.output_proj = nn.Linear(model_dim, 1)

    def forward(self, score_sequence):
        x = self.input_proj(score_sequence)
        x = self.temporal_transformer(x)
        refined_scores = self.output_proj(x)
        return refined_scores


# --- 4. Full End-to-End Model ---
class LocalizationFramework(nn.Module):
    """
    Integrates all components into a single end-to-end model.
    """

    def __init__(self):
        super().__init__()
        # 1. Vision Backbone (M²CRL)
        print("Initializing Vision Backbone...")
        self.vision_backbone = VisionTransformer(
            img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12
        )
        if os.path.exists(config.BACKBONE_WEIGHTS_PATH):
            print(f"Loading backbone weights from {config.BACKBONE_WEIGHTS_PATH}")
            # --- CORRECTED LINE ---
            # Added weights_only=False to handle checkpoints saved with older PyTorch versions
            # or those containing non-tensor data.
            state_dict = torch.load(config.BACKBONE_WEIGHTS_PATH, map_location='cpu', weights_only=False)
            if 'model' in state_dict:
                state_dict = state_dict['model']
            self.vision_backbone.load_state_dict(state_dict, strict=False)
        else:
            print(f"Warning: Backbone weights not found at {config.BACKBONE_WEIGHTS_PATH}. Using random init.",
                  file=sys.stderr)

        for param in self.vision_backbone.parameters():
            param.requires_grad = False
        print("Vision Backbone is frozen.")

        # 2. Text Encoder
        print("Initializing Text Encoder...")
        self.text_encoder = TextEncoder()

        # 3. Language-Guided Head
        print("Initializing Language-Guided Head...")
        self.language_guided_head = LanguageGuidedHead(
            visual_embed_dim=768,
            text_embed_dim=config.TEXT_EMBED_DIM
        )

        # 4. Temporal Head
        print("Initializing Temporal Head...")
        self.temporal_head = TemporalHead()

    def forward(self, video_clip, input_ids, attention_mask):
        B, T, C, H, W = video_clip.shape
        frames = video_clip.reshape(B * T, C, H, W)
        visual_features = self.vision_backbone(frames)
        text_features = self.text_encoder(input_ids, attention_mask)
        text_features_expanded = text_features.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, text_features.shape[1],
                                                                                          -1)
        text_mask_expanded = attention_mask.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
        raw_scores, _ = self.language_guided_head(visual_features, text_features_expanded, text_mask_expanded)
        raw_scores_sequence = raw_scores.view(B, T, 1)
        refined_scores_sequence = self.temporal_head(raw_scores_sequence)
        return raw_scores_sequence, refined_scores_sequence


if __name__ == '__main__':
    print("Testing the full LocalizationFramework...")

    if not os.path.exists(config.BACKBONE_WEIGHTS_PATH):
        print(f"Creating dummy backbone weights at {config.BACKBONE_WEIGHTS_PATH} for testing.")
        os.makedirs(os.path.dirname(config.BACKBONE_WEIGHTS_PATH), exist_ok=True)
        dummy_backbone = VisionTransformer()
        torch.save(dummy_backbone.state_dict(), config.BACKBONE_WEIGHTS_PATH)
        del dummy_backbone

    model = LocalizationFramework().to(config.DEVICE)

    B, T = 2, 16
    dummy_video = torch.randn(B, T, 3, 224, 224).to(config.DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    dummy_text = ["a polyp being removed", "healthy cecum tissue"]
    inputs = tokenizer(dummy_text, padding=True, return_tensors="pt").to(config.DEVICE)

    with torch.no_grad():
        raw_scores, refined_scores = model(dummy_video, inputs.input_ids, inputs.attention_mask)

    print("\n--- Model Test Successful ---")
    print(f"Input video shape: {dummy_video.shape}")
    print(f"Output raw scores shape: {raw_scores.shape}")
    print(f"Output refined scores shape: {refined_scores.shape}")
    print("---------------------------\n")
