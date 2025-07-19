import torch
import os


class Config:
    def __init__(self):
        # --- Base Directories on ML Server ---
        # Assuming '/home/240331715/' is your user's home directory on the server
        self.ML_SERVER_HOME = "/home/240331715/"

        # Base directory for all data (unified_medical_videos as per your structure)
        self.UNIFIED_MEDICAL_VIDEOS_DIR = os.path.join(self.ML_SERVER_HOME, "data", "unified_medical_videos")

        # Your project's root directory on the server
        self.PROJECT_ROOT = os.path.join(self.ML_SERVER_HOME, "data", "project_folder",
                                         "Language-Guided-Endoscopy-Localization")

        # --- Data Paths (derived from UNIFIED_MEDICAL_VIDEOS_DIR) ---
        self.EXTRACTED_FRAMES_DIR = os.path.join(self.UNIFIED_MEDICAL_VIDEOS_DIR, "extracted_frames")
        # Direct paths needed by prepare_cholec80.py and dataset.py
        self.SPLIT_FILES_DIR = os.path.join(self.UNIFIED_MEDICAL_VIDEOS_DIR, "final_triplets", "cholec80_splits")
        self.OUTPUT_TRIPLETS_DIR = os.path.join(self.UNIFIED_MEDICAL_VIDEOS_DIR, "final_triplets")  # ADDED THIS LINE
        self.CHOLEC80_PARSED_ANNOTATIONS = os.path.join(self.UNIFIED_MEDICAL_VIDEOS_DIR, "parsed_annotations",
                                                        "CHOLEC80_parsed_annotations.csv")  # ADDED THIS LINE

        self.TRAIN_TRIPLETS_CSV_PATH = os.path.join(self.UNIFIED_MEDICAL_VIDEOS_DIR, "final_triplets",
                                                    "cholec80_train_triplets.csv")
        self.VAL_TRIPLETS_CSV_PATH = os.path.join(self.UNIFIED_MEDICAL_VIDEOS_DIR, "final_triplets",
                                                  "cholec80_val_triplets.csv")
        self.TEST_TRIPLETS_CSV_PATH = os.path.join(self.UNIFIED_MEDICAL_VIDEOS_DIR, "final_triplets",
                                                   "cholec80_test_triplets.csv")
        self.VIDEO_ROOT_PATH = os.path.join(self.UNIFIED_MEDICAL_VIDEOS_DIR,
                                            "extracted_frames")  # Assuming videos are extracted frames

        # --- Model & Checkpoint Paths (derived from PROJECT_ROOT) ---
        # Path to your pre-trained M2CRL backbone weights
        self.BACKBONE_WEIGHTS_PATH = os.path.join(self.PROJECT_ROOT, "pretrained", "checkpoint.pth")
        # Directory to save training checkpoints
        self.CHECKPOINT_DIR = os.path.join(self.PROJECT_ROOT, "checkpoints", "cholec80")
        # Directory for model outputs (e.g., inference results, visualizations)
        self.OUTPUT_DIR = os.path.join(self.PROJECT_ROOT, "outputs", "cholec80")

        # --- Data Parameters (consistent with your previous configuration) ---
        self.DATA = self.DataConfig()

        # --- Model Architecture Parameters (consistent with your previous configuration) ---
        self.MODEL = self.ModelConfig()

        # --- Training Hyperparameters (consistent with your previous configuration) ---
        self.TRAIN = self.TrainConfig()

        # --- TimeSformer (M²CRL Backbone) Specific Parameters ---
        self.TIMESFORMER = self.TimesformerConfig()

        # --- Cholec80 Label to Text Query Mapping ---
        self.LABEL_TO_TEXT_QUERY = {
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

    class DataConfig:
        def __init__(self):
            self.TRAIN_CROP_SIZE = 224
            self.AUGMENT_PROB = 0.5
            self.NUM_FRAMES = 16  # Number of frames in a video clip for spatio-temporal backbone
            self.FRAME_RATE = 30  # Original video frame rate
            self.CLIP_LENGTH = 16  # Keep consistent with NUM_FRAMES
            self.NUM_INFERENCE_FRAMES = 50  # From your original config snippet
            self.MAX_TEXT_LENGTH = 77  # From your original config snippet
            self.NUM_WORKERS = 4  # From your original config snippet

    class ModelConfig:
        def __init__(self):
            # --- Backbone Selection ---
            # Choose your backbone: 'M2CRL' or 'EndoMamba'
            self.VISION_BACKBONE_NAME = "EndoMamba"

            # --- Backbone-Specific Paths ---
            # Path for your original M2CRL weights
            self.M2CRL_WEIGHTS_PATH = os.path.join(config.PROJECT_ROOT, "pretrained", "checkpoint.pth")
            # Path for the downloaded EndoMamba weights
            self.ENDOMAMBA_WEIGHTS_PATH = os.path.join(config.PROJECT_ROOT, "pretrained",
                                                       "endomamba_small_b48_seqlen16_withteacher_MIX12_checkpoint-499.pth")

            # --- General Model Parameters ---
            self.TEXT_ENCODER_MODEL = "openai/clip-vit-base-patch32"
            self.HEAD_NUM_ATTENTION_HEADS = 8
            self.HEAD_NUM_LAYERS = 2
            # NOTE: This embed_dim is for the text encoder and heads.
            # The vision_embed_dim will be set dynamically in the model itself.
            self.EMBED_DIM = 768

    class TrainConfig:
        def __init__(self):
            self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # From your original config snippet
            self.LEARNING_RATE = 1e-3  # From your original config snippet
            self.BATCH_SIZE = 24  # From your original config snippet
            self.NUM_EPOCHS = 10  # From your original config snippet, renamed from EPOCHS
            self.TEMPORAL_LOSS_WEIGHT = 0.5  # From your original config snippet
            self.WARMUP_EPOCHS = 1
            self.WEIGHT_DECAY = 1e-2  # Added for full training config, common default
            self.GRADIENT_ACCUMULATION_STEPS = 1  # Added for full training config, common default
            self.LOG_INTERVAL = 10  # Added for training logging
            self.SAVE_INTERVAL = 2  # Added for checkpointing
            self.USE_CUDA = torch.cuda.is_available()  # Ensures device is set correctly

            # PEFT LoRA Config for Text Encoder (consistent with your original config snippet)
            self.USE_PEFT = True
            self.LORA_R = 8  # Renamed from PEFT_LORA_R for consistency
            self.LORA_ALPHA = 16  # Renamed from PEFT_LORA_ALPHA for consistency
            self.LORA_DROPOUT = 0.05  # Renamed from PEFT_LORA_DROPOUT for consistency
            self.LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # From your original config snippet

            # LoRA Configuration for Vision Backbone (added in previous turn)
            self.USE_LORA_BACKBONE = True  # Set to True to apply conceptual LoRA to the backbone
            self.LORA_R_BACKBONE = 8
            self.LORA_ALPHA_BACKBONE = 16
            self.LORA_DROPOUT_BACKBONE = 0.1

    class TimesformerConfig:
        def __init__(self):
            self.ATTENTION_TYPE = 'divided_space_time'  # Crucial for spatio-temporal M²CRL
            self.PRETRAINED_MODEL = ""  # Optional: If using a TimeSformer-specific pre-trained model URL/path


# Instantiate the config for use in other scripts
config = Config()