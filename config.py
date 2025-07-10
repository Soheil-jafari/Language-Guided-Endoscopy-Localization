# config.py
# Final version for server training.

import torch
import os

# --- Project Paths & Directories ---
# Base directory for all medical video data on the server
UNIFIED_MEDICAL_VIDEOS_DIR = "/data/unified_medical_videos/"

# Path to the directory containing all extracted frames
EXTRACTED_FRAMES_DIR = os.path.join(UNIFIED_MEDICAL_VIDEOS_DIR, "extracted_frames/")

# Path to the directory where you will save the train/val/test split files
SPLIT_FILES_DIR = os.path.join(UNIFIED_MEDICAL_VIDEOS_DIR, "final_triplets/cholec80_splits/")

# --- NEW: Paths to the final triplet CSVs ---
TRAIN_TRIPLETS_CSV_PATH = os.path.join(UNIFIED_MEDICAL_VIDEOS_DIR, "final_triplets/cholec80_train_triplets.csv")
VAL_TRIPLETS_CSV_PATH = os.path.join(UNIFIED_MEDICAL_VIDEOS_DIR, "final_triplets/cholec80_val_triplets.csv")
# The test set will be used later with inference.py
TEST_TRIPLETS_CSV_PATH = os.path.join(UNIFIED_MEDICAL_VIDEOS_DIR, "final_triplets/cholec80_test_triplets.csv")


# --- Other Paths ---
BACKBONE_WEIGHTS_PATH = "./pretrained/checkpoint.pth"
CHECKPOINT_DIR = "./checkpoints/cholec80/"
OUTPUT_DIR = "./outputs/cholec80/"


# --- Training Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
EPOCHS = 50
TEMPORAL_LOSS_WEIGHT = 0.5


# --- Cholec80 Label to Text Query Mapping ---
LABEL_TO_TEXT_QUERY = {
    'Preparation': 'the preparation phase of surgery',
    'CalotTriangleDissection': 'the dissection of the Calot triangle',
    'ClippingCutting': 'clipping and cutting',
    'GallbladderDissection': 'the dissection of the gallbladder',
    'GallbladderPackaging': 'packaging the gallbladder in a bag',
    'CleaningCoagulation': 'cleaning and coagulation',
    'GallbladderRetraction': 'the retraction of the gallbladder',
    'Grasper': 'a grasper tool is present',
    'Bipolar': 'a bipolar tool is present',
    'Hook': 'a hook tool is present',
    'Scissors': 'scissors are present',
    'Clipper': 'a clipper tool is present',
    'Irrigator': 'an irrigator tool is present',
    'SpecimenBag': 'a specimen bag is present',
}


# --- Model & Architecture Config ---
TEXT_MODEL_NAME = "openai/clip-vit-base-patch32"
TEXT_EMBED_DIM = 512
IMG_SIZE = 224
FUSION_HEAD_DEPTH = 2
FUSION_HEAD_NUM_HEADS = 8
TEMPORAL_HEAD_DEPTH = 2
TEMPORAL_HEAD_NUM_HEADS = 8


# --- PEFT LoRA Config ---
USE_PEFT = True
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]


# --- Dataset & Dataloader Config ---
NUM_INFERENCE_FRAMES = 50
MAX_TEXT_LENGTH = 77
NUM_WORKERS = 4
