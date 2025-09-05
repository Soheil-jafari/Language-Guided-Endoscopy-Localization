import sys
import os

# Add the project root directory to sys.path so modules like 'config' can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import random
from tqdm import tqdm
import argparse

from project_config import config


def generate_triplets_for_split(video_ids_for_split, all_annotations_df):
    """
    Generates (frame_path, text_query, relevance_label) triplets for a given
    list of video IDs.
    """
    # ensure a 'video_id' column like 'video01'
    if 'video_id' not in all_annotations_df.columns and 'standardized_video_id' in all_annotations_df.columns:
        all_annotations_df = all_annotations_df.copy()
        all_annotations_df['video_id'] = all_annotations_df['standardized_video_id'].str.extract(r'(video\d+)$')

    all_triplets = []

    master_class_list = list(config.LABEL_TO_TEXT_QUERY.keys())

    # Filter the main annotation dataframe to only include videos for the current split
    split_df = all_annotations_df[all_annotations_df['video_id'].isin(video_ids_for_split)]

    print(f"Processing {len(split_df)} annotation entries for this split...")

    # Use tqdm for a progress bar
    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc="Generating Triplets"):
        # Construct the frame_path using video_id, frame_index, and EXTRACTED_FRAMES_DIR
        video_id = row['standardized_video_id']
        frame_index = row['frame_idx']

        STRIDE = 5
        if int(frame_index) % STRIDE != 0:
            continue

        frame_filename = f"frame_{int(frame_index):07d}.jpg"

        # The full path: EXTRACTED_FRAMES_DIR/CHOLEC80__videoXX/frame_YYYYYYY.jpg
        frame_path = os.path.join(config.EXTRACTED_FRAMES_DIR, video_id, frame_filename)

        # Check if the frame file actually exists before creating triplets
        if not os.path.exists(frame_path):
            # print(f"Warning: Frame file not found: {frame_path}. Skipping.")
            continue

        # --- Determine all positive labels for the current frame ---
        positive_labels_for_frame = set()

        # ---- PHASE POSITIVES ----
        phase_raw = str(row.get('original_label') or '')
        phase_raw_l = phase_raw.lower()

        # 1) exact match against keys
        if phase_raw in master_class_list:
            positive_labels_for_frame.add(phase_raw)
        else:
            # 2) clean patterns like "99483\tGallbladderDissection" -> "GallbladderDissection"
            tail = phase_raw.split()[-1].split('\t')[-1]
            if tail in master_class_list:
                positive_labels_for_frame.add(tail)
            else:
                # 3) keyword mapping to canonical short keys
                for kw in ("calot", "dissection", "cleaning", "clipping", "preparation", "packaging", "retraction"):
                    if kw in phase_raw_l and kw in master_class_list:
                        positive_labels_for_frame.add(kw)
                        break

        # ---- TOOL POSITIVES ----
        for tool in master_class_list:
            if tool in row:
                try:
                    if pd.notna(row[tool]) and int(row[tool]) == 1:
                        positive_labels_for_frame.add(tool)
                except Exception:
                    pass
        # --- Generate Positive Samples ---
        for label in positive_labels_for_frame:
            query_text = config.LABEL_TO_TEXT_QUERY.get(label, label)
            all_triplets.append({
                "frame_path": frame_path,
                "text_query": query_text,
                "relevance_label": 1.0
            })

        # --- Generate Hard Negative Samples ---
        negative_candidate_labels = [lbl for lbl in master_class_list if lbl not in positive_labels_for_frame]

        # Add a few negative samples to balance the data
        # Ensure num_neg_samples doesn't exceed available negative candidates or desired positive samples
        num_pos_samples = len(positive_labels_for_frame)
        num_neg_samples = min(len(negative_candidate_labels), 2)

        if num_neg_samples > 0:
            chosen_neg_labels = random.sample(negative_candidate_labels, num_neg_samples)
            for neg_label in chosen_neg_labels:
                neg_query_text = config.LABEL_TO_TEXT_QUERY.get(neg_label, neg_label)
                all_triplets.append({
                    "frame_path": frame_path,
                    "text_query": neg_query_text,
                    "relevance_label": 0.0
                })

    return pd.DataFrame(all_triplets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate training/validation/test triplet CSVs for Cholec80 from a parsed CSV.")
    parser.add_argument(
        '--split',
        type=str,
        required=True,
        choices=['train', 'val', 'test'],
        help="The data split to process ('train', 'val', or 'test')."
    )
    args = parser.parse_args()

    random.seed(42)

    # --- Main Execution Logic ---
    # Access these directly from the config instance
    split_file_path = os.path.join(config.SPLIT_FILES_DIR, f"{args.split}_videos.txt")
    output_csv_path = os.path.join(config.OUTPUT_TRIPLETS_DIR, f"cholec80_{args.split}_triplets.csv")

    print(f"--- Preparing data for '{args.split}' split ---")
    print(f"Reading video IDs from: {split_file_path}")

    try:
        with open(split_file_path, 'r') as f:
            # The Cholec80 video names are like 'video01', 'video02', etc.
            video_ids_for_this_split = [f"video{int(line.strip()):02d}" for line in f.readlines()]
    except FileNotFoundError:
        print(f"Error: Split file not found at {split_file_path}", file=sys.stderr)
        print("Please run create_splits.py on the Cholec80 video directory first.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(video_ids_for_this_split)} videos for the '{args.split}' split.")

    print(f"Loading main parsed annotations from: {config.CHOLEC80_PARSED_ANNOTATIONS}")
    # Access this directly from the config instance
    all_annotations_df = pd.read_csv(config.CHOLEC80_PARSED_ANNOTATIONS)

    # Generate the triplets DataFrame for this split
    df_triplets = generate_triplets_for_split(video_ids_for_this_split, all_annotations_df)

    # Save the final CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_triplets.to_csv(output_csv_path, index=False)
    print(f"\nGenerated {len(df_triplets)} training triplets. Saved to {output_csv_path}")
    print("--- Done! ---")