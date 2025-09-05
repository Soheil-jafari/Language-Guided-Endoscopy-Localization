# Save this file as: xclip/data.py

import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from PIL import Image
import re
import torch
from torch.utils.data import Dataset

# --- Existing Helper Functions (No changes needed here) ---

@dataclass
class Triplet:
    video: str
    query: str
    start_frame: int
    end_frame: int
    fps: int

def load_triplets(csv_path: str, default_fps: int = 30) -> List[Triplet]:
    """Loads triplets from a CSV, handling different column names."""
    df = pd.read_csv(csv_path)
    triplets = []
    
    # Auto-detect column names
    video_col = 'video' if 'video' in df.columns else 'video_id'
    text_col = 'query' if 'query' in df.columns else 'text'
    start_frame_col = 'start_frame' if 'start_frame' in df.columns else None
    end_frame_col = 'end_frame' if 'end_frame' in df.columns else None
    start_time_col = 'start_time' if 'start_time' in df.columns else None
    end_time_col = 'end_time' if 'end_time' in df.columns else None
    fps_col = 'fps' if 'fps' in df.columns else None

    for _, row in df.iterrows():
        fps = int(row.get(fps_col, default_fps))
        start_frame, end_frame = -1, -1

        if start_frame_col and end_frame_col and pd.notna(row[start_frame_col]):
            start_frame = int(row[start_frame_col])
            end_frame = int(row[end_frame_col])
        elif start_time_col and end_time_col and pd.notna(row[start_time_col]):
            start_frame = int(float(row[start_time_col]) * fps)
            end_frame = int(float(row[end_time_col]) * fps)
        
        if start_frame != -1:
            triplets.append(Triplet(
                video=str(row[video_col]),
                query=str(row[text_col]),
                start_frame=start_frame,
                end_frame=end_frame,
                fps=fps,
            ))
    return triplets

def get_frame_paths(base_dir: str, video_id: str, glob_pattern: str = "*.jpg") -> List[Path]:
    """Finds and sorts frame paths for a given video ID."""
    video_dir = Path(base_dir) / str(video_id)
    if not video_dir.exists():
        return []
    return sorted(list(video_dir.glob(glob_pattern)))

def is_frame_list_format(df_head: pd.DataFrame) -> bool:
    """Detects if CSV is in frame-list format (frame_path, text_query, label)."""
    return "frame_path" in df_head.columns and "relevance_label" in df_head.columns

def load_annotations_from_frame_list(csv_path: str, default_fps: int):
    """
    Input CSV columns (frame-level): frame_path, text_query, relevance_label (0/1)
    Output:
      items: {(video, query): {"gts":[(start,end),...], "fps":float}}
      frames_index_by_video: {}  # not needed by the new pipeline, return empty
    """
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    fp, tq, rl = cols["frame_path"], cols["text_query"], cols["relevance_label"]

    # derive video_id = parent folder name; frame_idx = last number in filename
    df["_video"] = df[fp].map(lambda p: Path(p).parent.name)
    num_re = re.compile(r"(\d+)(?!.*\d)")
    def idx_from_path(p):
        m = num_re.search(Path(p).name)
        return int(m.group(1)) if m else None
    df["_idx"] = df[fp].map(idx_from_path)

    # if some files have no index, order by path
    for (v, q), g in df.groupby(["_video", tq]):
        if g["_idx"].isna().any():
            order = g.sort_values(fp).index
            df.loc[order, "_idx"] = range(len(order))
    df["_idx"] = df["_idx"].astype(int)

    # merge consecutive 1's into [start,end) segments per (video, query)
    items = {}
    for (v, q), g in df.sort_values(["_video", tq, "_idx"]).groupby(["_video", tq]):
        segs, start, prev = [], None, None
        for i, lab in g[["_idx", rl]].values.tolist() + [[None, 0]]:
            if lab == 1 and start is None:
                start = i; prev = i
            elif lab == 1:
                prev = i
            elif lab == 0 and start is not None:
                segs.append((int(start), int(prev)+1))
                start = None; prev = None
        items[(v, str(q))] = {"gts": segs, "fps": float(default_fps)}

    return items, {}

# --- NEW DATASET CLASS FOR PARALLEL LOADING ---

class VideoWindowDataset(Dataset):
    """
    A PyTorch Dataset that represents all possible sliding windows from all videos.
    This allows a DataLoader to fetch windows in parallel, which is much faster.
    """
    def __init__(self, items, config, args, processor):
        self.items = items
        self.config = config
        self.args = args
        self.processor = processor
        self.windows = []

        print("[INFO] Pre-calculating all windows for the dataset...")
        # Flatten all possible windows from all videos into a single list
        # This makes indexing (`__getitem__`) straightforward.
        for (video, query), bundle in self.items.items():
            frame_paths = get_frame_paths(config.EXTRACTED_FRAMES_DIR, video, args.frame_glob)
            if not frame_paths:
                continue

            n_frames = len(frame_paths)
            num_frames_in_window = self.args.num_frames
            
            if n_frames >= num_frames_in_window:
                window_starts = list(range(0, n_frames - num_frames_in_window + 1, self.args.stride))
            else:
                window_starts = [0] # Handle very short videos
            
            if self.args.max_windows_per_video is not None:
                window_starts = window_starts[:self.args.max_windows_per_video]

            for start_frame in window_starts:
                self.windows.append({
                    "video": video,
                    "query": query,
                    "frame_paths": frame_paths,
                    "start_frame": start_frame,
                })

    def __len__(self):
        """Returns the total number of windows across all videos."""
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], str, str, int]:
        """

        Fetches a single window, loads its frames, and prepares it for the model.
        This is where the actual data loading happens, and it will be called in parallel
        by the DataLoader's workers.
        """
        window_info = self.windows[idx]
        num_frames_in_window = self.args.num_frames
        
        start = window_info["start_frame"]
        end = start + num_frames_in_window
        
        # Load ONLY the frames needed for this specific window
        clip_paths = window_info["frame_paths"][start:end]
        try:
            clip_images = [Image.open(p).convert("RGB") for p in clip_paths]
        except Exception as e:
            print(f"[ERROR] Failed to load image for window {idx}: {e}")
            # Return a dummy clip if an image is corrupted
            dummy_image = Image.new('RGB', (224, 224), color = 'red')
            clip_images = [dummy_image] * num_frames_in_window

        # Handle cases where the clip is too short (last window) by padding
        if len(clip_images) < num_frames_in_window:
            last_frame = clip_images[-1] if clip_images else Image.new('RGB', (224, 224))
            padding = [last_frame] * (num_frames_in_window - len(clip_images))
            clip_images.extend(padding)

        # The HuggingFace processor handles tensor conversion and normalization
        inputs = self.processor(
            videos=[clip_images], 
            text=[window_info["query"]], 
            return_tensors="pt",
            padding=True,
        )
        
        # Squeeze to remove the batch dimension of 1 that the processor adds
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # Return identifiers so we can reconstruct the results later
        return inputs, window_info['video'], window_info['query'], window_info['start_frame']
