import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
import os
import sys
from transformers import AutoTokenizer, AutoModel
# Import the project's configuration and backbone
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
        # Load the pre-trained text model from Hugging Face.
        # We only need the text projection part of the CLIP model.
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)

        if config.USE_PEFT:
            # Configure LoRA for the text encoder's attention layers.
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
            # If not using PEFT, freeze the entire model.
            for param in self.text_model.parameters():
                param.requires_grad = False
            print("Text Encoder is frozen (PEFT not enabled).")

    def forward(self, input_ids, attention_mask):
        # The text model outputs include the last hidden state and a pooled output.
        # We need the last_hidden_state for cross-attention.
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        # Return the sequence of token embeddings: [Batch, SequenceLength, TextEmbedDim]
        return outputs.last_hidden_state


# --- 2. Language-Guided Head (True Cross-Attention) ---
class LanguageGuidedHead(nn.Module):
    """
    A true Cross-Modal Transformer that fuses visual patch features with text token features.
    It uses cross-attention where visual patches are the queries and text tokens are the keys/values.
    """

    def __init__(self, visual_embed_dim=768, text_embed_dim=512, depth=config.FUSION_HEAD_DEPTH,
                 num_heads=config.FUSION_HEAD_NUM_HEADS):
        super().__init__()
        # Layer to project text features to the same dimension as visual features.
        self.text_proj = nn.Linear(text_embed_dim, visual_embed_dim)
        # A learnable token that will aggregate the fused information.
        self.fusion_token = nn.Parameter(torch.zeros(1, 1, visual_embed_dim))

        # A stack of cross-attention decoder layers.
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=visual_embed_dim,
                nhead=num_heads,
                batch_first=True
            ) for _ in range(depth)
        ])
        # The final head to predict a single relevance score.
        self.relevance_head = nn.Linear(visual_embed_dim, 1)

    def forward(self, visual_features, text_features, text_attention_mask):
        """
        Args:
            visual_features (torch.Tensor): [B, num_patches, visual_embed_dim]
            text_features (torch.Tensor): [B, text_seq_len, text_embed_dim]
            text_attention_mask (torch.Tensor): [B, text_seq_len] for padding mask.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (relevance_score, attention_weights_for_xai)
        """
        # Project text features to match visual dimension.
        text_memory = self.text_proj(text_features)  # [B, text_seq_len, visual_embed_dim]

        # Prepare the fusion token as the initial query for the decoder.
        fusion_query = self.fusion_token.expand(visual_features.shape[0], -1, -1)  # [B, 1, visual_embed_dim]

        # The visual features will be added to the query later or used as another memory.
        # For a standard decoder, we fuse the text memory with the fusion query.
        # We will then combine this with the visual features.

        # Let's use a more direct approach: visual features as query, text as memory.
        fused_output = visual_features

        # This is where the core fusion happens.
        for layer in self.decoder_layers:
            # In each layer, the visual features (queries) attend to the text features (memory).
            fused_output = layer(tgt=fused_output, memory=text_memory,
                                 memory_key_padding_mask=(1 - text_attention_mask).bool())

        # After fusion, we average the patch features to get a single vector per frame.
        frame_embedding = fused_output.mean(dim=1)  # [B, visual_embed_dim]

        # Predict the relevance score from this fused frame embedding.
        relevance_score = self.relevance_head(frame_embedding)  # [B, 1]

        # For XAI, we would need to capture attention weights from the final decoder layer.
        # This requires a custom forward pass on the layer. For now, we return None.
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
        # Input: [B, T, 1]
        x = self.input_proj(score_sequence)  # [B, T, model_dim]
        x = self.temporal_transformer(x)  # [B, T, model_dim]
        refined_scores = self.output_proj(x)  # [B, T, 1]
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
        # Load pre-trained weights and freeze the backbone
        if os.path.exists(config.BACKBONE_WEIGHTS_PATH):
            print(f"Loading backbone weights from {config.BACKBONE_WEIGHTS_PATH}")
            state_dict = torch.load(config.BACKBONE_WEIGHTS_PATH, map_location='cpu')
            # Handle potential 'model' key in checkpoint
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
            visual_embed_dim=768,  # From ViT-B/16
            text_embed_dim=config.TEXT_EMBED_DIM
        )

        # 4. Temporal Head
        print("Initializing Temporal Head...")
        self.temporal_head = TemporalHead()

    def forward(self, video_clip, input_ids, attention_mask):
        """
        Performs a full forward pass on a video clip.
        Args:
            video_clip (torch.Tensor): [B, T, C, H, W]
            input_ids (torch.Tensor): [B, text_seq_len]
            attention_mask (torch.Tensor): [B, text_seq_len]
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (raw_scores, refined_scores)
        """
        B, T, C, H, W = video_clip.shape

        # Reshape for backbone: [B*T, C, H, W]
        frames = video_clip.reshape(B * T, C, H, W)

        # Stage 1: Get visual features for all frames
        visual_features = self.vision_backbone(frames)  # [B*T, num_patches, visual_embed_dim]

        # Stage 2: Get text features (once per clip)
        text_features = self.text_encoder(input_ids, attention_mask)  # [B, text_seq_len, text_embed_dim]

        # Expand text features to match the batch size of frames
        # [B, text_seq_len, D] -> [B, 1, text_seq_len, D] -> [B, T, text_seq_len, D] -> [B*T, text_seq_len, D]
        text_features_expanded = text_features.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, text_features.shape[1],
                                                                                          -1)
        text_mask_expanded = attention_mask.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)

        # Stage 3: Get raw relevance score for each frame
        raw_scores, _ = self.language_guided_head(visual_features, text_features_expanded,
                                                  text_mask_expanded)  # [B*T, 1]

        # Reshape scores back to sequence: [B, T, 1]
        raw_scores_sequence = raw_scores.view(B, T, 1)

        # Stage 4: Refine scores with Temporal Head
        refined_scores_sequence = self.temporal_head(raw_scores_sequence)  # [B, T, 1]

        return raw_scores_sequence, refined_scores_sequence


if __name__ == '__main__':
    # This block allows you to test the full model definition
    print("Testing the full LocalizationFramework...")

    # Create a dummy M2CRL weights file for testing if it doesn't exist
    if not os.path.exists(config.BACKBONE_WEIGHTS_PATH):
        print(f"Creating dummy backbone weights at {config.BACKBONE_WEIGHTS_PATH} for testing.")
        os.makedirs(os.path.dirname(config.BACKBONE_WEIGHTS_PATH), exist_ok=True)
        dummy_backbone = VisionTransformer()
        torch.save(dummy_backbone.state_dict(), config.BACKBONE_WEIGHTS_PATH)
        del dummy_backbone

    model = LocalizationFramework().to(config.DEVICE)

    # Create dummy inputs
    B, T = 2, 16  # Batch size of 2, 16 frames per clip
    dummy_video = torch.randn(B, T, 3, 224, 224).to(config.DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    dummy_text = ["a polyp being removed", "healthy cecum tissue"]
    inputs = tokenizer(dummy_text, padding=True, return_tensors="pt").to(config.DEVICE)

    # Test forward pass
    with torch.no_grad():
        raw_scores, refined_scores = model(dummy_video, inputs.input_ids, inputs.attention_mask)

    print("\n--- Model Test Successful ---")
    print(f"Input video shape: {dummy_video.shape}")
    print(f"Output raw scores shape: {raw_scores.shape}")
    print(f"Output refined scores shape: {refined_scores.shape}")
    print("---------------------------\n")

