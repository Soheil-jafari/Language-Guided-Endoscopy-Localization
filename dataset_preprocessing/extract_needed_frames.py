#!/usr/bin/env python3
"""
Extract only the frames actually referenced (directly, or via the +/- clip_length/2
sliding window used by dataset.EndoscopyLocalizationDataset.__getitem__) by the
existing triplets CSVs, instead of every frame of every Cholec80 video.

Why one archive per video, not loose files: on network-filesystem-backed storage
(e.g. RunPod's MooseFS-backed /workspace), writing/reading millions of individual
small JPEG files each pays a per-file open/metadata round-trip -- observed at
~44ms/file on such a mount, vs ~0.05ms/file on local disk. Writing all of a video's
needed frames into ONE ZIP_STORED archive (opened once, closed once) collapses
that to a handful of large-file operations instead of millions of tiny ones.
ZIP_STORED (no compression) is used because JPEG bytes are already compressed --
paying CPU for deflate on top buys essentially nothing.

Output layout: <output_frames_dir>/CHOLEC80__videoNN.zip, containing entries named
exactly like the original loose-file convention (frame_0000000.jpg, etc.) so
dataset.py's frame-index parsing is unaffected. See dataset.py's _get_zip_handle /
_load_frame for the matching read side (falls back to legacy loose-directory reads
if a video predates this format and has no .zip).

Your existing triplets CSVs and parsed annotations CSV do not need to change at all;
this script only recreates the frame images they already point to, in a different
on-disk container format.
"""
import argparse
import os
import re
import sys
import zipfile

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
    # Always include the true first/last frame so dataset.py's bounds detection
    # discovers the REAL video bounds, not just the bounds of this sparse set.
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


def entry_name(idx, digits=7):
    return f"frame_{idx:0{digits}d}.jpg"


def existing_entries(zip_path):
    """Names already present in a video's archive, if it exists. Empty set otherwise."""
    if not os.path.exists(zip_path):
        return set()
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            return set(zf.namelist())
    except zipfile.BadZipFile:
        # Truncated/corrupt archive (e.g. from an interrupted run) -- treat as empty
        # so this video gets fully re-extracted rather than silently skipped.
        print(f"[warn] {zip_path} is corrupt/unreadable, re-extracting from scratch.")
        return set()


def main():
    ap = argparse.ArgumentParser(
        description="Extract only the frames needed by existing triplets CSVs into one ZIP_STORED "
                    "archive per video (space- and network-filesystem-friendly alternative to "
                    "extract_cholec80_frames.py's full dense, loose-file extraction)."
    )
    ap.add_argument("--cholec80_videos_dir", type=str, required=True,
                    help="Directory containing raw videoXX.mp4 files.")
    ap.add_argument("--output_frames_dir", type=str, default=config.EXTRACTED_FRAMES_DIR,
                    help="Root directory to write extracted frame archives into (must match "
                        "config.EXTRACTED_FRAMES_DIR on the machine you train on).")
    ap.add_argument("--triplets_csvs", type=str, nargs="+", default=[
        config.TRAIN_TRIPLETS_CSV_PATH, config.VAL_TRIPLETS_CSV_PATH, config.TEST_TRIPLETS_CSV_PATH
    ], help="Triplets CSVs to scan for referenced frames (default: train/val/test from project_config.py).")
    ap.add_argument("--clip_length", type=int, default=config.DATA.CLIP_LENGTH,
                    help="Must match the clip_length dataset.py is constructed with (default: from config).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-extract frames even if already present in a video's archive.")
    ap.add_argument("--delete_source_after_extract", action="store_true",
                    help="Delete each video's raw .mp4 immediately after its archive is fully "
                        "written and verified, to keep peak disk usage down instead of keeping "
                        "all raw videos until the whole run finishes.")
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
        zip_path = os.path.join(args.output_frames_dir, f"{video_id}.zip")

        if not os.path.exists(video_path):
            # Source already deleted (e.g. from a prior --delete_source_after_extract run).
            # That's fine as long as the archive already has everything it needs.
            if os.path.exists(zip_path):
                print(f"[ok] {video_id}: source video already removed, archive present -- skipping.")
            else:
                print(f"[skip] source video not found and no archive exists: {video_path}")
            continue

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"[skip] could not read frame count for {video_path}")
            cap.release()
            continue

        needed = needed_indices_for_video(centers, args.clip_length, total_frames)
        have = set() if args.overwrite else existing_entries(zip_path)
        to_write = {idx for idx in needed if entry_name(idx) not in have}

        total_needed_all += len(needed)
        total_frames_all += total_frames

        if not to_write:
            print(f"{video_id}: all {len(needed)} needed frames already in archive, skipping.")
            cap.release()
            if args.delete_source_after_extract and os.path.exists(video_path):
                os.remove(video_path)
            continue

        pct = 100.0 * len(needed) / total_frames
        print(f"{video_id}: need {len(needed)}/{total_frames} frames ({pct:.1f}% of the full video), "
              f"{len(to_write)} not yet in archive. Decoding sequentially...")

        # Append to the existing archive if resuming (mode 'a'), else create fresh.
        zip_mode = "a" if (have and not args.overwrite) else "w"
        if args.overwrite and os.path.exists(zip_path):
            os.remove(zip_path)  # 'w' would otherwise append duplicate entries to a stale zip
            zip_mode = "w"

        written_this_video = 0
        with zipfile.ZipFile(zip_path, zip_mode, compression=zipfile.ZIP_STORED) as zf:
            # Sequential decode (not cv2.CAP_PROP_POS_FRAMES seeking, which is unreliable
            # across codecs). Cost is one full decode pass per video, same as before --
            # the change here is the OUTPUT container, not the decode strategy.
            for i in tqdm(range(total_frames), desc=video_id):
                ret, frame = cap.read()
                if not ret:
                    continue
                if i not in to_write:
                    continue
                ok, buf = cv2.imencode(".jpg", frame)
                if not ok:
                    print(f"  [warn] failed to encode frame {i} of {video_id}", file=sys.stderr)
                    continue
                zf.writestr(entry_name(i), buf.tobytes())
                written_this_video += 1
                total_written += 1

        cap.release()

        if args.delete_source_after_extract:
            # Verify the archive actually has everything needed before deleting the source.
            final_entries = existing_entries(zip_path)
            missing = {entry_name(idx) for idx in needed} - final_entries
            if missing:
                print(f"  [warn] {video_id}: archive still missing {len(missing)} entries after "
                      f"extraction -- NOT deleting source video.", file=sys.stderr)
            else:
                os.remove(video_path)
                print(f"  {video_id}: source video deleted (archive verified complete).")

    print(f"\nDone. Wrote {total_written} new frame entries to archives in {args.output_frames_dir}")
    if total_frames_all > 0:
        print(f"Needed frames were {total_needed_all}/{total_frames_all} "
              f"({100.0*total_needed_all/total_frames_all:.1f}%) of the full dense extraction.")
    print("Your existing triplets CSVs and parsed annotations CSV remain valid as-is.")


if __name__ == "__main__":
    main()
