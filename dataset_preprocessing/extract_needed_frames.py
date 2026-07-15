#!/usr/bin/env python3
"""
Extract only the frames actually referenced (directly, or via the +/- clip_length/2
sliding window used by dataset.EndoscopyLocalizationDataset.__getitem__) by the
existing triplets CSVs, instead of every frame of every Cholec80 video.

Why: extracting every frame of all 80 Cholec80 videos can be several hundred GB of
JPEGs, but the triplets CSVs already reference a sparse, stride-subsampled set of
center frames, and __getitem__ only ever reads a `clip_length`-frame window around
each one. This script extracts exactly that set (plus each video's true frame 0 and
last frame, so dataset.py's listdir-based bounds detection sees the video's REAL
start/end and clamps windows identically to how they are computed here) instead of
the full dense extraction -- typically a small fraction of the total video.

Your existing triplets CSVs and parsed annotations CSV do not need to change at all;
this script only recreates the frame image files they already point to.
"""
import argparse
import os
import re
import sys

import cv2
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from project_config import config

FRAME_RE = re.compile(r"frame_(\d+)\.jpg$", re.IGNORECASE)


def needed_indices_for_video(center_indices, clip_length, total_frames):
    """
    Mirror dataset.py's exact start/clamp arithmetic so the extracted set matches
    precisely what __getitem__ will ever request for this video:

        start = max(0, center - clip_length // 2)
        start = max(min_idx, start)
        start = min(start, max(min_idx, max_idx - clip_length + 1))
        frames read = [start, start + clip_length)
    """
    min_idx, max_idx = 0, total_frames - 1
    needed = set()
    for center in center_indices:
        start = max(0, center - clip_length // 2)
        start = max(min_idx, start)
        start = min(start, max(min_idx, max_idx - clip_length + 1))
        for t in range(clip_length):
            needed.add(start + t)
    # Always include the true first/last frame so dataset.py's os.listdir-based bounds
    # detection discovers the REAL video bounds, not just the bounds of this sparse set.
    needed.add(min_idx)
    needed.add(max_idx)
    return needed


def collect_centers_by_video(triplets_csvs):
    """video_id -> set(center_frame_idx), parsed from each triplets CSV's frame_path column."""
    by_video = {}
    for csv_path in triplets_csvs:
        if not os.path.exists(csv_path):
            print(f"[skip] triplets CSV not found: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        for frame_path in df["frame_path"]:
            frame_path = str(frame_path)
            video_id = os.path.basename(os.path.dirname(frame_path))
            m = FRAME_RE.search(os.path.basename(frame_path))
            if not m:
                continue
            idx = int(m.group(1))
            by_video.setdefault(video_id, set()).add(idx)
    return by_video


def main():
    ap = argparse.ArgumentParser(
        description="Extract only the frames needed by existing triplets CSVs (space-efficient alternative "
                    "to extract_cholec80_frames.py's full dense extraction)."
    )
    ap.add_argument("--cholec80_videos_dir", type=str, required=True,
                    help="Directory containing raw videoXX.mp4 files.")
    ap.add_argument("--output_frames_dir", type=str, default=config.EXTRACTED_FRAMES_DIR,
                    help="Root directory to write extracted frames into (must match config.EXTRACTED_FRAMES_DIR "
                        "on the machine you train on).")
    ap.add_argument("--triplets_csvs", type=str, nargs="+", default=[
        config.TRAIN_TRIPLETS_CSV_PATH, config.VAL_TRIPLETS_CSV_PATH, config.TEST_TRIPLETS_CSV_PATH
    ], help="Triplets CSVs to scan for referenced frames (default: train/val/test from project_config.py).")
    ap.add_argument("--clip_length", type=int, default=config.DATA.CLIP_LENGTH,
                    help="Must match the clip_length dataset.py is constructed with (default: from config).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-extract frames even if the target file already exists.")
    args = ap.parse_args()

    print("Scanning triplets CSVs for referenced frames...")
    centers_by_video = collect_centers_by_video(args.triplets_csvs)
    if not centers_by_video:
        print("No referenced frames found in the given triplets CSVs. Nothing to do.", file=sys.stderr)
        sys.exit(1)
    print(f"Found references across {len(centers_by_video)} videos.")

    os.makedirs(args.output_frames_dir, exist_ok=True)
    total_written = 0
    total_needed_all = 0
    total_frames_all = 0

    for video_id, centers in sorted(centers_by_video.items()):
        m = re.search(r"video(\d+)$", video_id)
        if not m:
            print(f"[skip] could not parse a video number from '{video_id}'")
            continue
        video_num = int(m.group(1))
        video_filename = f"video{video_num:02d}.mp4"
        video_path = os.path.join(args.cholec80_videos_dir, video_filename)
        if not os.path.exists(video_path):
            print(f"[skip] source video not found: {video_path}")
            continue

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"[skip] could not read frame count for {video_path}")
            cap.release()
            continue

        needed = needed_indices_for_video(centers, args.clip_length, total_frames)
        out_dir = os.path.join(args.output_frames_dir, video_id)
        os.makedirs(out_dir, exist_ok=True)

        to_write = set()
        for idx in needed:
            fpath = os.path.join(out_dir, f"frame_{idx:07d}.jpg")
            if args.overwrite or not os.path.exists(fpath):
                to_write.add(idx)

        total_needed_all += len(needed)
        total_frames_all += total_frames

        if not to_write:
            print(f"{video_id}: all {len(needed)} needed frames already present, skipping.")
            cap.release()
            continue

        pct = 100.0 * len(needed) / total_frames
        print(f"{video_id}: need {len(needed)}/{total_frames} frames ({pct:.1f}% of the full video), "
              f"{len(to_write)} not yet on disk. Decoding sequentially...")

        # Sequential decode (not cv2.CAP_PROP_POS_FRAMES seeking, which is unreliable
        # across codecs). Cost is one full decode pass per video, same as the original
        # extract_cholec80_frames.py -- the saving here is disk space, not decode time.
        for i in tqdm(range(total_frames), desc=video_id):
            ret, frame = cap.read()
            if not ret:
                continue
            if i not in to_write:
                continue
            fpath = os.path.join(out_dir, f"frame_{i:07d}.jpg")
            cv2.imwrite(fpath, frame)
            total_written += 1

        cap.release()

    print(f"\nDone. Wrote {total_written} new frame files to {args.output_frames_dir}")
    if total_frames_all > 0:
        print(f"Needed frames were {total_needed_all}/{total_frames_all} "
              f"({100.0*total_needed_all/total_frames_all:.1f}%) of the full dense extraction.")
    print("Your existing triplets CSVs and parsed annotations CSV remain valid as-is.")


if __name__ == "__main__":
    main()
