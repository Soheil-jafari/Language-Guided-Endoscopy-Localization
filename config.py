import torch
import os

# --- Project Paths & Directories ---
# !! IMPORTANT !! You MUST update these paths to match your system's layout.
# Path to the directory containing your video datasets.
VIDEO_DATA_DIR = "/path/to/your/video_datasets/"
# Path to your consolidated annotation file (e.g., a JSON file).
ANNOTATION_FILE = "/path/to/your/annotations.json"
# Path to the pre-trained M²CRL weights file you copied earlier.
BACKBONE_WEIGHTS_PATH = "./pretrained/m2crl_weights.pth"
# Directory where trained model checkpoints will be saved.
CHECKPOINT_DIR = "./checkpoints/"
# Directory where inference results (videos with heatmaps) will be saved.
OUTPUT_DIR = "./outputs/"


# --- Training Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 4  # Adjust based on your GPU memory
EPOCHS = 50
# Weight for the secondary temporal consistency loss.
TEMPORAL_LOSS_WEIGHT = 0.5


# --- Model & Architecture Config ---
# Configuration for the Text Encoder and Language-Guided Head
TEXT_MODEL_NAME = "openai/clip-vit-base-patch32" # Using a standard CLIP text encoder
TEXT_EMBED_DIM = 512 # Embedding dimension for CLIP's text encoder
IMG_SIZE = 224
# Number of cross-attention layers in our Language-Guided Head.
FUSION_HEAD_DEPTH = 2
FUSION_HEAD_NUM_HEADS = 8

# Configuration for the Temporal Transformer Head
TEMPORAL_HEAD_DEPTH = 2
TEMPORAL_HEAD_NUM_HEADS = 8


# --- PEFT LoRA Config ---
# Configuration for applying LoRA to the Text Encoder.
USE_PEFT = True
LORA_R = 8          # Rank of the LoRA matrices.
LORA_ALPHA = 16     # Alpha scaling factor.
LORA_DROPOUT = 0.05
# Modules in the CLIP text encoder to apply LoRA to.
LORA_TARGET_MODULES = ["q_proj", "v_proj"]


# --- Dataset & Dataloader Config ---
# The frame rate to process videos at. This standardizes all videos.
TARGET_FPS = 10
# The number of frames to sample from a video clip during inference.
# For a 5-second clip at 10 FPS, this would be 50.
NUM_INFERENCE_FRAMES = 50
# Maximum number of text tokens for the tokenizer.
MAX_TEXT_LENGTH = 77 # Standard for CLIP
NUM_WORKERS = 4 # Number of CPU cores for data loading. Adjust based on your system.
TRIPLETS_CSV_PATH = "./data/real_colon_training_triplets.csv"
