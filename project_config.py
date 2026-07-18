import torch
import os


class Config:
    def __init__(self):
        # --- Base Directories on ML Server ---
        # Override with the PROJECT_HOME environment variable if your server layout
        # differs; otherwise defaults to RunPod's standard persistent-volume mount.
        self.ML_SERVER_HOME = os.environ.get("PROJECT_HOME", "/workspace/")

        self.UNIFIED_MEDICAL_VIDEOS_DIR = os.path.join(self.ML_SERVER_HOME, "data", "unified_medical_videos")

        self.PROJECT_ROOT = os.path.join(self.ML_SERVER_HOME, "data", "project_folder",
                                         "Language-Guided-Endoscopy-Localization")

        # Derived from UNIFIED_MEDICAL_VIDEOS_DIR (not hardcoded separately) so changing
        # ML_SERVER_HOME/PROJECT_HOME is the only edit needed on a new machine.
        self.EXTRACTED_FRAMES_DIR = os.path.join(self.UNIFIED_MEDICAL_VIDEOS_DIR, "extracted_frames")
        self.SPLIT_FILES_DIR = os.path.join(self.UNIFIED_MEDICAL_VIDEOS_DIR, "final_triplets", "cholec80_splits")
        self.OUTPUT_TRIPLETS_DIR = os.path.join(self.UNIFIED_MEDICAL_VIDEOS_DIR, "final_triplets")
        self.CHOLEC80_PARSED_ANNOTATIONS = os.path.join(self.UNIFIED_MEDICAL_VIDEOS_DIR, "parsed_annotations",
                                                        "CHOLEC80_parsed_annotations.csv")

        # NOTE: these point at the _remapped triplets (see
        # dataset_preprocessing/remap_triplet_paths.py). The original CSVs still
        # contain frame_path values from whatever server they were first generated
        # on; the _remapped copies have frame_path rewritten to this machine's
        # EXTRACTED_FRAMES_DIR. Do not point this back at the un-remapped files.
        self.TRAIN_TRIPLETS_CSV_PATH = os.path.join(self.UNIFIED_MEDICAL_VIDEOS_DIR, "final_triplets",
                                                    "cholec80_train_triplets_remapped.csv")
        self.VAL_TRIPLETS_CSV_PATH = os.path.join(self.UNIFIED_MEDICAL_VIDEOS_DIR, "final_triplets",
                                                  "cholec80_val_triplets_remapped.csv")
        self.TEST_TRIPLETS_CSV_PATH = os.path.join(self.UNIFIED_MEDICAL_VIDEOS_DIR, "final_triplets",
                                                   "cholec80_test_triplets_remapped.csv")
        self.VIDEO_ROOT_PATH = os.path.join(self.UNIFIED_MEDICAL_VIDEOS_DIR,
                                            "extracted_frames")

        # --- Model & Checkpoint Paths  ---
        self.BACKBONE_WEIGHTS_PATH = os.path.join(self.PROJECT_ROOT, "pretrained", "checkpoint.pth")
        # Directory to save training checkpoints
        self.EXPERIMENT_TAG = "baseline"  # or "improved"
        self.CHECKPOINT_DIR = os.path.join(self.PROJECT_ROOT, "checkpoints", f"cholec80_{self.EXPERIMENT_TAG}")
        # Directory for model outputs (e.g., inference results, visualizations)
        self.OUTPUT_DIR = os.path.join(self.PROJECT_ROOT, "outputs", "cholec80")

        self.DATA = self.DataConfig()
        self.TRAIN = self.TrainConfig()
        self.MODEL = self.ModelConfig(project_root=self.PROJECT_ROOT)
        # --- TimeSformer (M²CRL Backbone) Specific Parameters ---
        self.TIMESFORMER = self.TimesformerConfig()

        # Inference Settings
        self.SEGMENT_THRESHOLD = 0.5  # higher = fewer/cleaner segments, lower = more/longer segments
        self.MIN_SEGMENT_DURATION = 0.4  # drop very short bursts
        self.MERGE_GAP = 0.2  # merge close segments
        self.INFER_IMG_SIZE = 224

        self.LABEL_TO_TEXT_QUERY = {
            # phases — canonical short keys
            "calot": "Calot triangle dissection phase",
            "dissection": "Gallbladder dissection phase",
            "cleaning": "Cleaning and coagulation phase",
            "clipping": "Clipping and cutting phase",
            "preparation": "Preparation phase",
            "packaging": "Gallbladder packaging phase",
            "retraction": "Retraction phase",

            # phases — raw CHOLEC80 names
            "CalotTriangleDissection": "Calot triangle dissection phase",
            "GallbladderDissection": "Gallbladder dissection phase",
            "CleaningCoagulation": "Cleaning and coagulation phase",
            "ClippingCutting": "Clipping and cutting phase",
            "Preparation": "Preparation phase",
            "GallbladderPackaging": "Gallbladder packaging phase",
            "GallbladderRetraction": "Retraction phase",

            # tools
            "grasper": "a grasper is present",
            "bipolar": "a bipolar forceps is present",
            "hook": "a hook cautery is present",
            "scissors": "scissors are present",
            "clip": "a clip applier is present",
            "clipper": "a clip applier is present",
            "irrigator": "an irrigator is present",
            "suction": "a suction instrument is present",
            "specimen": "a specimen retrieval bag is present",
            "bag": "a specimen retrieval bag is present",
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

            # --- Temporal head selection ---
            # 'TRANSFORMER' (default) or 'SSM' to use the Mamba state-space head.
            self.TEMPORAL_HEAD_TYPE = 'TRANSFORMER'
            # SSM (Mamba) temporal-head options; only used when TEMPORAL_HEAD_TYPE == 'SSM'.
            # Prefer the official mamba_ssm CUDA library; auto-fallback to the built-in
            # MambaBlock if it is not installed (so training never breaks on a missing lib).
            self.SSM_USE_OFFICIAL_MAMBA = True
            self.SSM_NUM_LAYERS = 4
            self.SSM_D_STATE = 16
            self.SSM_D_CONV = 4
            self.SSM_EXPAND = 2

            self.USE_UNCERTAINTY = False
            self.USE_CONFIDENCE_FUSION = False
            # This embed_dim is for the text encoder and heads.
            # The vision_embed_dim will be set dynamically in the model itself.
            self.EMBED_DIM = 768

    class TrainConfig:
        def __init__(self):
            self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            self.LEARNING_RATE = 2e-5
            # BATCH_SIZE=32 OOMs a single RTX 4090 on the very first forward pass
            # (measured: 22.7GB used before crashing). 8 is confirmed to run cleanly
            # with headroom to spare on this model. Paired with
            # GRADIENT_ACCUMULATION_STEPS=24 below to preserve the same effective
            # batch size (8*24 = 32*6 = 192) the rest of the config was tuned around.
            self.BATCH_SIZE = 8
            self.NUM_EPOCHS = 20
            self.TEMPORAL_LOSS_WEIGHT = 0.45
            # Positive-class weight for BCEWithLogitsLoss. Per-frame relevance is
            # negative-skewed (most frames are not the queried concept), so a value > 1
            # (e.g. the neg/pos ratio) trades precision for recall. Default 1.0 = no weighting.
            self.BCE_POS_WEIGHT = 1.0
            self.WARMUP_EPOCHS = 3
            self.WEIGHT_DECAY = 0.2
            self.EVIDENTIAL_LAMBDA = 0.2
            self.GRADIENT_ACCUMULATION_STEPS = 24
            # Mixed-precision dtype for autocast: 'fp16' (default) or 'bf16'. Use 'bf16'
            # on Ampere+ GPUs (A100/H100/RTX 30xx+) for more stable training with no loss
            # scaling. The GradScaler is disabled automatically when using bf16.
            self.AMP_DTYPE = 'fp16'
            # Trade compute for memory in the vision backbone (activation checkpointing).
            # Lets you fit a larger batch or longer clips; ~20-30% slower per step.
            self.USE_GRADIENT_CHECKPOINTING = False
            self.LOG_INTERVAL = 10
            self.SAVE_INTERVAL = 1
            self.USE_BILEVEL_CONSISTENCY = False
            self.SEMANTIC_LOSS_WEIGHT = 0.3
            self.OPTICAL_FLOW_LOSS_WEIGHT = 0.5
            self.SUBSET_RATIO = 0.2
            self.USE_CUDA = torch.cuda.is_available()

            self.USE_PEFT = True
            self.LORA_R = 8
            self.LORA_ALPHA = 16
            self.LORA_DROPOUT = 0.4
            self.LORA_TARGET_MODULES = ["q_proj", "v_proj"]

            # LoRA Configuration for Vision Backbone
            self.USE_LORA_BACKBONE = False  # toggle LoRA on the vision backbone
            self.FREEZE_BACKBONE_WHEN_LORA = True  # optional: freeze all other backbone weights when using LoRA
            self.LORA_R_BACKBONE = 8
            self.LORA_ALPHA_BACKBONE = 6
            self.LORA_DROPOUT_BACKBONE = 0.4

    class TimesformerConfig:
        def __init__(self):
            self.ATTENTION_TYPE = 'divided_space_time'  # Crucial for spatio-temporal M²CRL
            self.PRETRAINED_MODEL = ""  # Optional: If using a TimeSformer-specific pre-trained model URL/path


# Instantiate the config for use in other scripts
config = Config()
