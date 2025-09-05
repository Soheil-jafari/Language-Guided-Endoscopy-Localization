"""
Moment-DETR Baseline: Step 1 - Preprocessing (Server-Ready, Option A: one record per segment)

Reads frame-level CSVs with columns:
  - frame_path
  - text_query
  - relevance_label (1 for positive)
Merges positive frames into contiguous segments with a small gap tolerance, filters very short segments,
and writes ONE JSONL RECORD PER SEGMENT with exactly one [start, end] span per line.
"""

import re
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# --- Configuration (Paths from your project_config.py) ---
class ServerConfig:
    UNIFIED_MEDICAL_VIDEOS_DIR = "/home/240331715/data/unified_medical_videos"
    TRAIN_CSV = f"{UNIFIED_MEDICAL_VIDEOS_DIR}/final_triplets/cholec80_train_triplets.csv"
    VAL_CSV   = f"{UNIFIED_MEDICAL_VIDEOS_DIR}/final_triplets/cholec80_val_triplets.csv"
    TEST_CSV  = f"{UNIFIED_MEDICAL_VIDEOS_DIR}/final_triplets/cholec80_test_triplets.csv"
    FRAME_RATE = 30

cfg = ServerConfig()

# --- Output location ---
OUTPUT_DIR = Path("./preprocessed_data")
SPLIT_MAPPING = {"train": cfg.TRAIN_CSV, "val": cfg.VAL_CSV, "test": cfg.TEST_CSV}

# --- Segmentation knobs (tune as needed) ---
MERGE_GAP_FRAMES   = 15   # merge gaps <= 15 frames (~0.5s at 30 fps)
MIN_SEG_LEN_FRAMES = 30   # drop segments shorter than 30 frames (~1.0s at 30 fps)

def extract_info_from_path(path_str: str):
    """
    Extracts video_id and frame_idx from a path like:
      '.../CHOLEC80__video01/frame_0000123.jpg'
    """
    video_id_match = re.search(r'(CHOLEC80__video\d+)', str(path_str))
    frame_idx_match = re.search(r'frame_(\d+)\.jpg', str(path_str))
    video_id = video_id_match.group(1) if video_id_match else None
    frame_idx = int(frame_idx_match.group(1)) if frame_idx_match else -1
    return video_id, frame_idx

def merge_frames_with_gap(frames, gap=MERGE_GAP_FRAMES):
    """
    Merge a sorted list of frame indices into [start, end_exclusive] segments,
    allowing gaps up to `gap` frames inside the same (video_id, text_query) group.
    """
    if not frames:
        return []
    frames = sorted(frames)
    segments = []
    s = e = frames[0]
    for f in frames[1:]:
        if f <= e + gap:
            e = f
        else:
            segments.append([s, e + 1])  # end exclusive
            s = e = f
    segments.append([s, e + 1])
    return segments

def main():
    print("[INFO] Starting preprocessing for Moment-DETR (Option A: one record per segment).")
    OUTPUT_DIR.mkdir(exist_ok=True)

    for split, csv_path in SPLIT_MAPPING.items():
        if not Path(csv_path).exists():
            print(f"[ERROR] Cannot find CSV for '{split}': {csv_path}")
            continue

        print(f"\n--> Processing split: {split} from {csv_path}")
        df = pd.read_csv(csv_path)

        # Extract video_id and frame_idx from file paths
        if 'frame_path' not in df.columns or 'text_query' not in df.columns or 'relevance_label' not in df.columns:
            raise ValueError("CSV must contain columns: frame_path, text_query, relevance_label")

        print("    Extracting video_id and frame_idx from frame_path...")
        path_info = df['frame_path'].apply(extract_info_from_path)
        df['video_id']  = [info[0] for info in path_info]
        df['frame_idx'] = [info[1] for info in path_info]

        # Keep only positive labels
        positive_frames = df[df['relevance_label'] == 1].copy()

        # Group by (video, query)
        grouped = positive_frames.groupby(['video_id', 'text_query'])['frame_idx'] \
                                 .apply(list).reset_index()

        output_path = OUTPUT_DIR / f"{split}.jsonl"
        records_written = 0

        print(f"    Merging frames into segments with gap<={MERGE_GAP_FRAMES} and min_len>={MIN_SEG_LEN_FRAMES} ...")
        with open(output_path, "w", encoding="utf-8") as f:
            for _, row in tqdm(grouped.iterrows(), total=len(grouped)):
                video_id, query = row['video_id'], row['text_query']
                frame_indices = sorted(row['frame_idx'])

                # (1) merge with tolerated gaps
                segs = merge_frames_with_gap(frame_indices, gap=MERGE_GAP_FRAMES)

                # (2) filter tiny segments
                segs = [seg for seg in segs if (seg[1] - seg[0]) >= MIN_SEG_LEN_FRAMES]
                if not segs:
                    continue

                # Compute video duration from all frames belonging to this video in the WHOLE CSV
                max_frame = df[df['video_id'] == video_id]['frame_idx'].max()
                duration = float(max_frame + 1) / float(cfg.FRAME_RATE) if max_frame >= 0 else 0.0

                # (3) OPTION A: write ONE RECORD PER SEGMENT (exactly one [start, end] per JSON line)
                for s, e in segs:
                    s_sec, e_sec = s / cfg.FRAME_RATE, e / cfg.FRAME_RATE
                    record = {
                        "video": video_id,
                        "duration": duration,
                        "query": query,
                        "timestamps": [[s_sec, e_sec]]
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records_written += 1

        print(f"    Saved {records_written} records to: {output_path}")

    print("\n[SUCCESS] Preprocessing complete. Output JSONL has exactly one span per record (Option A).")

if __name__ == "__main__":
    main()
