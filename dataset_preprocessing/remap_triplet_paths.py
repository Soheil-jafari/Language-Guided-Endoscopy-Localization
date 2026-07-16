#!/usr/bin/env python3
"""
Remap the frame_path column of existing triplets CSVs to point at THIS machine's
extracted-frames directory, instead of whatever server they were originally
generated on.

The triplets CSVs (frame_path, text_query, relevance_label) and the parsed
annotations CSV are otherwise environment-independent -- only the absolute path
baked into frame_path needs to change when moving to a new machine. This avoids
re-running prepare_cholec80.py (which would re-derive triplets from the
annotations file and re-roll the random hard-negative sampling); it just points
the existing rows at the new location.

Usage (defaults to config.TRAIN/VAL/TEST_TRIPLETS_CSV_PATH, writing
"<name>_remapped.csv" next to each so the originals are untouched until you
verify the output):

    python dataset_preprocessing/remap_triplet_paths.py

Then, once you've checked the output looks right, either point project_config.py
at the "_remapped" files, or re-run with --in_place to overwrite the originals.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from project_config import config

ANCHOR = "extracted_frames/"


def remap_path(old_path: str, new_extracted_frames_dir: str) -> str:
    """
    Keep everything from the video-id folder onward (e.g. 'CHOLEC80__video01/
    frame_0000001.jpg') and reattach it to new_extracted_frames_dir. Robust to
    whatever server-specific prefix came before 'extracted_frames/' originally
    (handles paths generated on different machines/configs without needing to
    know the exact old prefix).
    """
    normalized = old_path.replace("\\", "/")
    idx = normalized.find(ANCHOR)
    if idx == -1:
        raise ValueError(f"Could not find '{ANCHOR}' in path: {old_path}")
    tail = normalized[idx + len(ANCHOR):]  # e.g. "CHOLEC80__video01/frame_0000001.jpg"
    return os.path.join(new_extracted_frames_dir, tail)


def _with_suffix(path, suffix):
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}"


def remap_csv(csv_path: str, new_extracted_frames_dir: str, in_place: bool, suffix: str):
    if not os.path.exists(csv_path):
        print(f"[skip] not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    if "frame_path" not in df.columns:
        print(f"[skip] no 'frame_path' column in: {csv_path}")
        return

    before = df["frame_path"].iloc[0] if len(df) else None
    df["frame_path"] = df["frame_path"].apply(lambda p: remap_path(str(p), new_extracted_frames_dir))
    after = df["frame_path"].iloc[0] if len(df) else None

    out_path = csv_path if in_place else _with_suffix(csv_path, suffix)
    df.to_csv(out_path, index=False)
    print(f"[OK] {csv_path} -> {out_path}  ({len(df)} rows)")
    if before is not None:
        print(f"     example: {before}")
        print(f"          ->  {after}")


def main():
    ap = argparse.ArgumentParser(
        description="Remap triplets CSV frame_path columns to this machine's extracted-frames directory."
    )
    ap.add_argument("--csvs", type=str, nargs="+", default=[
        config.TRAIN_TRIPLETS_CSV_PATH, config.VAL_TRIPLETS_CSV_PATH, config.TEST_TRIPLETS_CSV_PATH
    ], help="Triplets CSVs to remap (default: train/val/test paths from project_config.py).")
    ap.add_argument("--extracted_frames_dir", type=str, default=config.EXTRACTED_FRAMES_DIR,
                    help="New root for extracted frames on this machine (default: config.EXTRACTED_FRAMES_DIR).")
    ap.add_argument("--in_place", action="store_true",
                    help="Overwrite the CSVs directly instead of writing '<name>_remapped.csv'.")
    ap.add_argument("--suffix", type=str, default="_remapped",
                    help="Suffix for output files when not using --in_place (default: '_remapped').")
    args = ap.parse_args()

    for csv_path in args.csvs:
        remap_csv(csv_path, args.extracted_frames_dir, args.in_place, args.suffix)


if __name__ == "__main__":
    main()
