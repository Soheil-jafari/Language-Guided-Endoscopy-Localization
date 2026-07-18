import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import os
import cv2
import zipfile
import torchvision.transforms.v2 as T
from transformers import AutoTokenizer
import sys
import random
import re
import numpy as np
from project_config import config

# --- KEYWORD MAPPING ---
PHASE_KEYWORDS = {
    "calot triangle dissection": 3,
    "dissection": 3,
    "preparation": 0, "calot": 1, "clipping": 2, "cutting": 2,
    "packaging": 4, "cleaning": 5, "coagulation": 5,
    "retraction": 6
}

TOOL_KEYWORDS = {
    "grasper": 0, "bipolar": 1, "hook": 2, "scissors": 3,
    "clip": 4, "clipper": 4, "irrigator": 5, "suction": 5,
    "specimen": 6, "bag": 6
}

NUM_TOOL_CLASSES = max(TOOL_KEYWORDS.values()) + 1

# --- CONCEPT PARSING FUNCTION ---
def parse_query_kind(text_query: str):
    q = text_query.lower()
    for k, pid in PHASE_KEYWORDS.items():
        if k in q:
            return ("phase", pid)
    for k, tid in TOOL_KEYWORDS.items():
        if k in q:
            return ("tool", tid)
    return ("unknown", None)


class EndoscopyLocalizationDataset(Dataset):
    """
    Loads 16-frame clips and generates per-frame labels by robustly mapping
    the triplet's text query to a concept (phase/tool) and looking up ground
    truth labels from the parsed annotations file.
    """

    def __init__(self, triplets_csv_path, tokenizer, clip_length=16, is_training=True):
        # ---- Read triplets ----
        self.triplets_df = pd.read_csv(triplets_csv_path)
        assert "frame_path" in self.triplets_df.columns, "Triplets CSV must have 'frame_path'"
        assert "text_query" in self.triplets_df.columns, "Triplets CSV must have 'text_query'"

        self.tokenizer = tokenizer
        self.clip_length = clip_length
        self.is_training = is_training
        self._video_bounds = {}
        # Cache of open ZipFile handles, keyed by video_dir. Populated lazily per
        # Dataset instance -- i.e. per DataLoader worker process when num_workers > 0 --
        # so repeated frame reads from the same video are seek+read on an already-open
        # handle instead of a fresh file open each time. See _get_zip_handle/_load_frame.
        self._zip_handles = {}

        # ==================== CONCEPT-BASED LABEL LOOKUP ====================
        print("Building concept-based label lookup from parsed annotations...")
        ann_path = config.CHOLEC80_PARSED_ANNOTATIONS
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Parsed annotations CSV not found at: {ann_path}")

        ann_df = pd.read_csv(ann_path)

        # Standardize video IDs to match the format derived from frame_path (e.g., 'CHOLEC80__video01')
        if 'standardized_video_id' not in ann_df.columns:
            raise KeyError("Parsed annotations must have 'standardized_video_id' column")

        # Create two separate, efficient lookups: one for phases, one for tools.
        self.phase_label_lookup = {}  # Key: (video_id_str, frame_idx), Value: phase_id
        self.tool_label_lookup = {}  # Key: (video_id_str, frame_idx), Value: list of 7 tool presence flags [0,1,0,0,1,0,0]

        # Use tqdm for progress tracking
        from tqdm import tqdm
        for _, row in tqdm(ann_df.iterrows(), total=len(ann_df), desc="Processing annotations"):
            video_id_str = row['standardized_video_id']
            frame_idx = int(row['frame_idx'])
            key = (video_id_str, frame_idx)

            # For phases, we store the phase ID directly.
            # We assume one phase per frame. 'original_label' seems to hold the phase name.
            phase_name = str(row.get('original_label', '')).lower()
            if phase_name:
                for keyword, phase_id in PHASE_KEYWORDS.items():
                    if keyword in phase_name:
                        self.phase_label_lookup[key] = phase_id
                        break  # Move to next row once phase is found

            # For tools, we build a multi-hot vector for all 7 tool types.
            if key not in self.tool_label_lookup:
                self.tool_label_lookup[key] = [0] * NUM_TOOL_CLASSES  # Initialize with all zeros

            # Populate the tool vector based on tool columns (grasper, bipolar, etc.)
            for tool_keyword, tool_id in TOOL_KEYWORDS.items():
                # Check if a column matching the tool keyword exists and its value is 1
                if tool_keyword in ann_df.columns and int(row.get(tool_keyword, 0)) == 1:
                    self.tool_label_lookup[key][tool_id] = 1

        print(
            f"Lookup created. Found annotations for {len(self.phase_label_lookup)} phase instances and {len(self.tool_label_lookup)} tool instances.")

        # ---- Transforms ----
        self.train_transforms = T.Compose([
            T.RandomResizedCrop((config.DATA.TRAIN_CROP_SIZE, config.DATA.TRAIN_CROP_SIZE), scale=(0.5, 1.0),
                                ratio=(0.7, 1.4)),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
            T.RandomHorizontalFlip(p=config.DATA.AUGMENT_PROB),
            T.RandomApply([T.RandomRotation(degrees=20)], p=0.5),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.val_transforms = T.Compose([
            T.Resize(size=config.DATA.TRAIN_CROP_SIZE + 32),
            T.CenterCrop(size=config.DATA.TRAIN_CROP_SIZE),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.triplets_df)

    def _get_zip_handle(self, video_dir):
        """
        Return a cached, already-open ZipFile handle for this video's archive
        (extract_needed_frames.py's ZIP_STORED output), or None if no archive exists
        for it -- e.g. a video extracted before the archive format was introduced,
        which still has a loose-file directory instead. Cached per Dataset instance
        (per DataLoader worker process), so every read after the first for a given
        video is a seek+read on an already-open handle rather than a fresh file open
        -- this is what makes per-batch frame loading fast on network filesystems.
        """
        if video_dir in self._zip_handles:
            return self._zip_handles[video_dir]
        zip_path = video_dir.rstrip(os.sep) + ".zip"
        zf = None
        if os.path.exists(zip_path):
            try:
                zf = zipfile.ZipFile(zip_path, "r")
            except zipfile.BadZipFile:
                zf = None
        self._zip_handles[video_dir] = zf
        return zf

    def _load_frame(self, video_dir, fname):
        """
        Load a single frame given its video directory and entry filename (e.g.
        'frame_0000001.jpg'). Prefers video_dir's zip archive if one exists; falls
        back to a loose file directly in video_dir for videos extracted before the
        archive format was introduced.
        """
        zf = self._get_zip_handle(video_dir)
        if zf is not None:
            try:
                data = zf.read(fname)
            except KeyError:
                return None
            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return None
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Legacy loose-file fallback.
        fpath = os.path.join(video_dir, fname)
        if not os.path.exists(fpath):
            return None
        img = cv2.imread(fpath)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        row = self.triplets_df.iloc[idx]
        frame_path = str(row["frame_path"])
        text_query = str(row["text_query"])

        # Parse video_id_str and center_frame_idx from the frame_path
        video_id_str = os.path.basename(os.path.dirname(frame_path))
        m = re.search(r"frame_(\d+)\.jpg$", os.path.basename(frame_path), flags=re.IGNORECASE)
        digits = len(m.group(1)) if m else 7
        center_frame_idx = int(m.group(1)) if m else 0

        # Build 16-frame window and load frames
        start_frame_idx = max(0, center_frame_idx - self.clip_length // 2)
        video_dir = os.path.dirname(frame_path)

        # discover min/max once per video dir (from the zip archive's entries if one
        # exists, otherwise from the loose-file directory listing)
        if video_dir not in self._video_bounds:
            min_idx, max_idx = 0, 0
            try:
                idxs = []
                zf = self._get_zip_handle(video_dir)
                names = zf.namelist() if zf is not None else os.listdir(video_dir)
                for f in names:
                    m2 = re.match(r"frame_(\d+)\.jpg$", f)
                    if m2:
                        idxs.append(int(m2.group(1)))
                if idxs:
                    min_idx, max_idx = min(idxs), max(idxs)
            except FileNotFoundError:
                pass
            self._video_bounds[video_dir] = (min_idx, max_idx)

        min_idx, max_idx = self._video_bounds[video_dir]

        # shift/clamp the window so it stays inside [min_idx, max_idx]
        start_frame_idx = max(min_idx, start_frame_idx)
        start_frame_idx = min(start_frame_idx, max(min_idx, max_idx - self.clip_length + 1))

        frames, last_valid = [], None
        for t in range(self.clip_length):
            fidx = start_frame_idx + t
            fname = f"frame_{fidx:0{digits}d}.jpg"
            img = self._load_frame(video_dir, fname)
            if img is None:
                if last_valid is not None:
                    frames.append(last_valid.copy())
                else:
                    frames.append(
                        np.zeros((config.DATA.TRAIN_CROP_SIZE, config.DATA.TRAIN_CROP_SIZE, 3), dtype=np.uint8))
            else:
                frames.append(img)
                last_valid = img

        pil_frames = [T.ToPILImage()(f) for f in frames]
        transform = self.train_transforms if self.is_training else self.val_transforms
        video_clip_tensor = transform(pil_frames)
        if isinstance(video_clip_tensor, list):
            video_clip_tensor = torch.stack(video_clip_tensor, dim=0)
        video_clip = video_clip_tensor.permute(1, 0, 2, 3).contiguous()

        # Tokenize text
        text_inputs = self.tokenizer(text_query, padding='max_length', truncation=True,
                                     max_length=config.DATA.MAX_TEXT_LENGTH, return_tensors="pt")
        input_ids = text_inputs['input_ids'].squeeze(0)
        attention_mask = text_inputs['attention_mask'].squeeze(0)

        # ==================== PER-FRAME LABEL GENERATION ====================
        # Parse the text query to find out what concept (phase/tool) we are looking for.
        concept, concept_id = parse_query_kind(text_query)

        labels = []
        for t in range(self.clip_length):
            fidx = start_frame_idx + t
            key = (video_id_str, fidx)
            label = 0.0  # Default label is 0 (not relevant)

            if concept == "phase":
                # Check if the phase at this frame matches the query's phase concept.
                if self.phase_label_lookup.get(key, -1) == concept_id:
                    label = 1.0
            elif concept == "tool":
                # Check if the specific tool is present in this frame.
                tool_flags = self.tool_label_lookup.get(key)
                if tool_flags and tool_flags[concept_id] == 1:
                    label = 1.0

            labels.append(label)

        relevance_tensor = torch.tensor(labels, dtype=torch.float32)
        frame_indices = list(range(start_frame_idx, start_frame_idx + self.clip_length))

        return {
            "video_clip": video_clip,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": relevance_tensor,
            "video_id": video_id_str,
            "frame_indices": torch.tensor(frame_indices, dtype=torch.int32),
        }


# --- DataLoaders function ---
def create_dataloaders(train_csv_path, val_csv_path, tokenizer, clip_length=16, subset_ratio=1.0):
    print(f"Loading training data from: {train_csv_path}")
    train_dataset = EndoscopyLocalizationDataset(
        train_csv_path, tokenizer, clip_length, is_training=True
    )

    print(f"Loading validation data from: {val_csv_path}")
    val_dataset = EndoscopyLocalizationDataset(
        val_csv_path, tokenizer, clip_length, is_training=False
    )

    if subset_ratio < 1.0:
        random.seed(42)
        torch.manual_seed(42)

        train_size = int(len(train_dataset) * subset_ratio)
        val_size = int(len(val_dataset) * subset_ratio)

        train_size = max(1, train_size)
        val_size = max(1, val_size)

        train_indices = random.sample(range(len(train_dataset)), train_size)
        val_indices = random.sample(range(len(val_dataset)), val_size)

        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

        print(f"Using {train_size} samples ({subset_ratio * 100:.1f}%) for training subset.")
        print(f"Using {val_size} samples ({subset_ratio * 100:.1f}%) for validation subset.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if config.DATA.NUM_WORKERS > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if config.DATA.NUM_WORKERS > 0 else False
    )

    return train_loader, val_loader