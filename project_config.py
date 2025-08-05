import torch
import os


class Config:
    def __init__(self):
        # --- Base Directories on ML Server ---
        self.ML_SERVER_HOME = "/home/240331715/"

        self.UNIFIED_MEDICAL_VIDEOS_DIR = os.path.join(self.ML_SERVER_HOME, "data", "unified_medical_videos")


        self.PROJECT_ROOT = os.path.join(self.ML_SERVER_HOME, "data", "project_folder",
                                         "Language-Guided-Endoscopy-Localization")

        self.EXTRACTED_FRAMES_DIR = os.path.join(self.UNIFIED_MEDICAL_VIDEOS_DIR, "extracted_frames")
        self.SPLIT_FILES_DIR = os.path.join(self.UNIFIED_MEDICAL_VIDEOS_DIR, "final_triplets", "cholec80_splits")
        self.OUTPUT_TRIPLETS_DIR = os.path.join(self.UNIFIED_MEDICAL_VIDEOS_DIR, "final_triplets")
        self.CHOLEC80_PARSED_ANNOTATIONS = os.path.join(self.UNIFIED_MEDICAL_VIDEOS_DIR, "parsed_annotations",
                                                        "CHOLEC80_parsed_annotations.csv")

        self.TRAIN_TRIPLETS_CSV_PATH = os.path.join(self.UNIFIED_MEDICAL_VIDEOS_DIR, "final_triplets",
                                                    "cholec80_train_triplets.csv")
        self.VAL_TRIPLETS_CSV_PATH = os.path.join(self.UNIFIED_MEDICAL_VIDEOS_DIR, "final_triplets",
                                                  "cholec80_val_triplets.csv")
        self.TEST_TRIPLETS_CSV_PATH = os.path.join(self.UNIFIED_MEDICAL_VIDEOS_DIR, "final_triplets",
                                                   "cholec80_test_triplets.csv")
        self.VIDEO_ROOT_PATH = os.path.join(self.UNIFIED_MEDICAL_VIDEOS_DIR,
                                            "extracted_frames")

        # --- Model & Checkpoint Paths (derived from PROJECT_ROOT) ---
        self.BACKBONE_WEIGHTS_PATH = os.path.join(self.PROJECT_ROOT, "pretrained", "checkpoint.pth")
        # Directory to save training checkpoints
        self.CHECKPOINT_DIR = os.path.join(self.PROJECT_ROOT, "checkpoints", "cholec80")
        # Directory for model outputs (e.g., inference results, visualizations)
        self.OUTPUT_DIR = os.path.join(self.PROJECT_ROOT, "outputs", "cholec80")

        self.DATA = self.DataConfig()

        self.TRAIN = self.TrainConfig()

        self.MODEL = self.ModelConfig(project_root=self.PROJECT_ROOT)
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
            self.NUM_FRAMES = 16
            self.FRAME_RATE = 30
            self.CLIP_LENGTH = 16
            self.NUM_INFERENCE_FRAMES = 50
            self.MAX_TEXT_LENGTH = 77
            self.NUM_WORKERS = 4

    class ModelConfig:
        def __init__(self, project_root):
            # --- Backbone Selection ---
            self.VISION_BACKBONE_NAME = "M2CRL"

            # --- Backbone-Specific Paths ---
            self.M2CRL_WEIGHTS_PATH = os.path.join(project_root, "pretrained", "checkpoint.pth")
            self.ENDOMAMBA_WEIGHTS_PATH = os.path.join(project_root, "pretrained", "checkpoint-499.pth")

            # --- General Model Parameters ---
            self.TEXT_ENCODER_MODEL = "openai/clip-vit-base-patch32"
            self.HEAD_NUM_ATTENTION_HEADS = 8
            self.HEAD_NUM_LAYERS = 2
            self.TEMPORAL_HEAD_TYPE = 'TRANSFORMER'
            self.USE_UNCERTAINTY = False
            self.USE_CONFIDENCE_FUSION = False
            # NOTE: This embed_dim is for the text encoder and heads.
            # The vision_embed_dim will be set dynamically in the model itself.
            self.EMBED_DIM = 768

    class TrainConfig:
        def __init__(self):
            self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            self.LEARNING_RATE = 8e-6
            self.BATCH_SIZE = 24
            self.NUM_EPOCHS = 30
            self.TEMPORAL_LOSS_WEIGHT = 0.2
            self.WARMUP_EPOCHS = 1
            self.WEIGHT_DECAY = 0.1
            self.EVIDENTIAL_LAMBDA = 0.2
            self.GRADIENT_ACCUMULATION_STEPS = 1
            self.LOG_INTERVAL = 10
            self.SAVE_INTERVAL = 1
            self.USE_BILEVEL_CONSISTENCY = False
            self.SEMANTIC_LOSS_WEIGHT = 0.3
            self.OPTICAL_FLOW_LOSS_WEIGHT = 0.5
            self.USE_CUDA = torch.cuda.is_available()

            self.USE_PEFT = True
            self.LORA_R = 8
            self.LORA_ALPHA = 16
            self.LORA_DROPOUT = 0.1
            self.LORA_TARGET_MODULES = ["q_proj", "v_proj"]

            # LoRA Configuration for Vision Backbone
            self.USE_LORA_BACKBONE = True  # Set to True to apply conceptual LoRA to the backbone
            self.LORA_R_BACKBONE = 8
            self.LORA_ALPHA_BACKBONE = 16
            self.LORA_DROPOUT_BACKBONE = 0.2

    class TimesformerConfig:
        def __init__(self):
            self.ATTENTION_TYPE = 'divided_space_time'  # Crucial for spatio-temporal M²CRL
            self.PRETRAINED_MODEL = ""  # Optional: If using a TimeSformer-specific pre-trained model URL/path


# Instantiate the config for use in other scripts
config = Config()