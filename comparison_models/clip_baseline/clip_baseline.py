import argparse, os, re, glob, math
from pathlib import Path
from typing import List, Tuple
from project_config import config
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import open_clip

def ensure_video_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if 'video_id' not in df.columns and 'standardized_video_id' in df.columns:
        df = df.copy()
        df['video_id'] = df['standardized_video_id'].str.extract(r'(video\d+)$')
    return df

def load_clip(model_name: str, pretrained: str, device: torch.device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer


@torch.no_grad()
def encode_text(model, tokenizer, device, text: str):
    tokens = tokenizer([text])
    feats = model.encode_text(tokens.to(device))
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats  # [1, D]


@torch.no_grad()
def encode_images(model, preprocess, device, image_paths: List[str], batch_size: int = 64):
    feats_all = []
    valid_paths = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding frames"):
        batch_paths = image_paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                im = Image.open(p).convert("RGB")
                imgs.append(preprocess(im))
                valid_paths.append(p)
            except Exception:
                # skip unreadable frames
                pass
        if not imgs:
            continue
        x = torch.stack(imgs, dim=0).to(device)
        feats = model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats_all.append(feats.cpu())

    if not feats_all:
        return np.zeros((0, 512)), []
    feats_all = torch.cat(feats_all, dim=0).numpy()
    return feats_all, valid_paths


def cosine_scores(img_feats: np.ndarray, txt_feats: torch.Tensor):
    # img_feats: [N, D] on CPU (np); txt_feats: [1, D] on device/CPU
    t = txt_feats.squeeze(0).cpu().numpy()
    sims = (img_feats @ t)   # cosine (because both normalized)
    return sims


def frames_in_window(center_frame_idx: int, clip_length: int, video_dir: str) -> Tuple[List[str], List[int]]:
    start = max(0, center_frame_idx - clip_length // 2)
    idxs = list(range(start, start + clip_length))
    paths = [os.path.join(video_dir, f"frame_{i:07d}.jpg") for i in idxs]
    return paths, idxs


def parse_center_from_path(frame_path: str) -> Tuple[str, int]:
    """Return (video_dir, center_frame_idx) from a path like .../CHOLEC80__video01/frame_0001234.jpg"""
    video_dir = os.path.dirname(frame_path)
    m = re.search(r"frame_(\d+)\.jpg$", os.path.basename(frame_path), flags=re.IGNORECASE)
    center = int(m.group(1)) if m else 0
    return video_dir, center


def run_video_dir(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, preprocess, tokenizer = load_clip(args.model, args.pretrained, device)

    # collect frames
    fpaths = sorted(glob.glob(os.path.join(args.video_dir, "frame_*.jpg")))
    if not fpaths:
        raise FileNotFoundError(f"No frames found under {args.video_dir}")

    text_feats = encode_text(model, tokenizer, device, args.text)
    img_feats, used_paths = encode_images(model, preprocess, device, fpaths, args.batch_size)
    sims = cosine_scores(img_feats, text_feats)

    df = pd.DataFrame({
        "frame_path": used_paths,
        "score": sims
    })
    # also extract frame_idx for convenience
    df["frame_idx"] = df["frame_path"].apply(
        lambda p: int(re.search(r"frame_(\d+)\.jpg$", os.path.basename(p)).group(1))
    )
    df = df.sort_values("frame_idx").reset_index(drop=True)

    out_csv = args.out or "clip_scores_video.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved per-frame CLIP scores → {out_csv}")

    topk = min(10, len(df))
    print("\nTop frames by score:")
    print(df.sort_values("score", ascending=False).head(topk))


def run_triplet_row(args):
    """
    Read a row from your *final_triplets* CSV, build the 16-frame window around the anchor,
    and score those frames with CLIP vs the row's text_query (unless overridden with --text).
    """
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, preprocess, tokenizer = load_clip(args.model, args.pretrained, device)

    trip = pd.read_csv(args.triplets_csv)
    assert "frame_path" in trip.columns and "text_query" in trip.columns, \
        "Triplets CSV must contain 'frame_path' and 'text_query'."

    if args.row_idx < 0 or args.row_idx >= len(trip):
        raise IndexError(f"--row_idx must be in [0, {len(trip)-1}]")

    row = trip.iloc[args.row_idx]
    anchor_fp = str(row["frame_path"])
    query = args.text if args.text is not None else str(row["text_query"])

    video_dir, center_idx = parse_center_from_path(anchor_fp)
    win_paths, win_idxs = frames_in_window(center_idx, args.clip_length, video_dir)

    # Some frames at boundaries may not exist; filter
    win_paths2, win_idxs2 = [], []
    for p, i in zip(win_paths, win_idxs):
        if os.path.exists(p):
            win_paths2.append(p)
            win_idxs2.append(i)

    if not win_paths2:
        raise FileNotFoundError("No frames exist in the requested 16-frame window.")

    print(f"Scoring {len(win_paths2)} frames around center={center_idx} for text: {query!r}")

    text_feats = encode_text(model, tokenizer, device, query)
    img_feats, used_paths = encode_images(model, preprocess, device, win_paths2, args.batch_size)
    sims = cosine_scores(img_feats, text_feats)

    df = pd.DataFrame({
        "frame_path": used_paths,
        "score": sims
    })
    df["frame_idx"] = df["frame_path"].apply(
        lambda p: int(re.search(r"frame_(\d+)\.jpg$", os.path.basename(p)).group(1))
    )
    df = df.sort_values("frame_idx").reset_index(drop=True)

    out_csv = args.out or "clip_scores_window.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved per-frame CLIP scores for the 16-frame window → {out_csv}")
    print(df)


def main():
    ap = argparse.ArgumentParser(description="CLIP baseline scoring for frames vs text.")
    sub = ap.add_subparsers(dest="mode", required=True)

    # Mode A: directory
    a = sub.add_parser("video-dir", help="Score all frames in a video directory.")
    a.add_argument("--video_dir", type=str, required=True,
                   help="Folder containing frame_XXXXXXX.jpg files.")
    a.add_argument("--text", type=str, required=True, help="Text query to score.")
    a.add_argument("--out", type=str, default=None, help="Output CSV path.")
    a.add_argument("--model", type=str, default="ViT-L-14", help="CLIP model name (open_clip).")
    a.add_argument("--pretrained", type=str, default="laion2b_s32b_b82k",
                   help="Pretrained tag for open_clip.")
    a.add_argument("--batch_size", type=int, default=64)
    a.add_argument("--cpu", action="store_true", help="Force CPU.")
    a.set_defaults(func=run_video_dir)

    # Mode B: triplets row → 16-frame window
    b = sub.add_parser("triplet-row", help="Score the 16-frame window around a row's anchor frame.")
    b.add_argument("--triplets_csv", type=str, required=True,
                   help="Path to final_triplets CSV that has 'frame_path' and 'text_query' columns.")
    b.add_argument("--row_idx", type=int, required=True, help="Row index inside the CSV.")
    b.add_argument("--clip_length", type=int, default=16)
    b.add_argument("--text", type=str, default=None,
                   help="Override the CSV text_query (optional).")
    b.add_argument("--out", type=str, default=None, help="Output CSV path.")
    b.add_argument("--model", type=str, default="ViT-L-14")
    b.add_argument("--pretrained", type=str, default="laion2b_s32b_b82k")
    b.add_argument("--batch_size", type=int, default=64)
    b.add_argument("--cpu", action="store_true")
    b.set_defaults(func=run_triplet_row)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate training/validation/test triplet CSVs for Cholec80 from the parsed CSV.")
    parser.add_argument(
        '--split',
        type=str,
        required=True,
        choices=['train', 'val', 'test'],
        help="The data split to process ('train', 'val', or 'test')."
    )
    args = parser.parse_args()

    import os
    import sys
    import pandas as pd
    import random

    random.seed(42)

    # --- Paths  ---
    # Split files: e.g., train_videos.txt / val_videos.txt / test_videos.txt
    split_file_path = os.path.join(config.SPLIT_FILES_DIR, f"{args.split}_videos.txt")

    # Triplets output: will overwrite existing files with the SAME NAME
    os.makedirs(config.OUTPUT_TRIPLETS_DIR, exist_ok=True)
    output_csv_path = os.path.join(config.OUTPUT_TRIPLETS_DIR, f"cholec80_{args.split}_triplets.csv")

    # Parsed annotations CSV you just rebuilt (phases + tools):
    parsed_csv_path = "/users/2/240331715/data/unified_medical_videos/parsed_annotations/CHOLEC80_parsed_annotations.csv"

    print(f"--- Preparing data for '{args.split}' split ---")
    print(f"Reading video IDs from: {split_file_path}")
    print(f"Reading parsed annotations from: {parsed_csv_path}")
    print(f"Triplets will be written to: {output_csv_path}")

    # --- Read split IDs (video01, video02, ...) ---
    try:
        with open(split_file_path, 'r') as f:
            video_ids_for_this_split = [f"video{int(line.strip()):02d}" for line in f.readlines()]
    except FileNotFoundError:
        print(f"Error: Split file not found at {split_file_path}", file=sys.stderr)
        sys.exit(1)

    # --- Read parsed annotations and ensure 'video_id' column exists ---
    if not os.path.exists(parsed_csv_path):
        print(f"Error: Parsed annotations not found at {parsed_csv_path}", file=sys.stderr)
        sys.exit(1)

    all_annotations_df = pd.read_csv(parsed_csv_path, low_memory=False)
    all_annotations_df = ensure_video_id_column(all_annotations_df)

    # --- Filter to this split only ---
    split_df = all_annotations_df[all_annotations_df['video_id'].isin(video_ids_for_this_split)]
    print(f"Parsed annotations rows in this split: {len(split_df)}")

    # --- Build triplets for this split using your existing function ---
    triplets_df = generate_triplets_for_split(video_ids_for_this_split, all_annotations_df)

    # --- Save ---
    triplets_df.to_csv(output_csv_path, index=False)
    print(f"✅ Wrote {len(triplets_df):,} triplets to {output_csv_path}")
