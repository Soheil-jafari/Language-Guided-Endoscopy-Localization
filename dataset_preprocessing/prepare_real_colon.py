# dataset_preprocessing/prepare_real_colon.py
# This script now reads the pre-generated split files (e.g., train_videos.txt)
# and generates a specific triplets CSV for that split (e.g., train_triplets.csv).

import os
import pandas as pd
from tqdm import tqdm
import random
import xml.etree.ElementTree as ET
import sys
import argparse  # To accept command-line arguments

# --- Configuration: CRITICAL PATHS ---
# These paths point to the root of the dataset on your server.
REAL_COLON_REPO_DIR = os.path.expanduser('~/data/real-colon-dataset')
FRAMES_BASE_DIR = os.path.join(REAL_COLON_REPO_DIR, 'dataset')
TEMPORAL_ANNOTATIONS_DIR = os.path.join(REAL_COLON_REPO_DIR, 'Temporal_Segmentation_Annotation')
POLYP_ANNOTATIONS_BASE_DIR = os.path.join(REAL_COLON_REPO_DIR, 'dataset')

# This is now the base directory for your output CSVs.
# The script will add 'train_', 'val_', or 'test_' prefix.
OUTPUT_DIR = os.path.expanduser('~/my_keyframe_project/data/')  # Points to your project's data folder

# --- Define landmark/phase mapping ---
LABEL_TO_TEXT_QUERY = {
    'outside': "outside the colon",
    'insertion': "the insertion phase of colonoscopy",
    'withdrawal': "the withdrawal phase of colonoscopy",
    'cecum': "the cecum",
    'ileum': "the ileum",
    'ascending colon': "the ascending colon",
    'transverse': "the transverse colon",
    'descending': "the descending colon",
    'sigmoid': "the sigmoid colon",
    'rectum': "the rectum",
    'polyp': "a polyp",
}


def check_polyp_presence_in_xml(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            if obj.find('name').text == 'lesion':
                return True
        return False
    except ET.ParseError as e:
        print(f"Error parsing XML {xml_path}: {e}. Assuming no polyp.", file=sys.stderr)
        return False


def prepare_split_data(video_ids_for_split):
    """
    The core logic to generate triplets, now for a specific list of video IDs.
    """
    all_triplets = []
    all_query_labels = list(LABEL_TO_TEXT_QUERY.keys())

    print("--- Phase 1: Collecting frame metadata for the split ---")
    video_frame_metadata = {}
    for video_id in tqdm(video_ids_for_split, desc="Collecting frame metadata"):
        # ... (The metadata collection logic is the same as before) ...
        current_video_frame_data = {}
        video_frames_dir = os.path.join(FRAMES_BASE_DIR, f"{video_id}_frames")
        temporal_csv_path = os.path.join(TEMPORAL_ANNOTATIONS_DIR, f"{video_id}.csv")
        temporal_df = pd.read_csv(temporal_csv_path) if os.path.exists(temporal_csv_path) else pd.DataFrame()

        if not temporal_df.empty and 'frame_filename' in temporal_df.columns and 'GT' in temporal_df.columns:
            temporal_df['frame_idx'] = temporal_df['frame_filename'].apply(
                lambda x: int(x.split('_')[-1].replace('.jpg', '')))
            temporal_df = temporal_df[['frame_idx', 'GT']].set_index('frame_idx')
            temporal_df.rename(columns={'GT': 'label'}, inplace=True)
            temporal_df['label'] = temporal_df['label'].fillna('background')
        else:
            temporal_df = pd.DataFrame()

        polyp_anno_dir = os.path.join(POLYP_ANNOTATIONS_BASE_DIR, f"{video_id}_annotations")
        if not os.path.exists(video_frames_dir): continue
        frame_files = sorted([f for f in os.listdir(video_frames_dir) if f.endswith('.jpg')])

        for frame_file in frame_files:
            try:
                frame_idx = int(frame_file.split('_')[-1].replace('.jpg', ''))
            except ValueError:
                continue

            frame_path = os.path.join(video_frames_dir, frame_file)
            temporal_label = temporal_df.loc[frame_idx, 'label'] if frame_idx in temporal_df.index else 'background'

            polyp_present = False
            polyp_xml_path = os.path.join(polyp_anno_dir, f"{video_id}_t{frame_idx}.xml")
            if os.path.exists(polyp_xml_path):
                polyp_present = check_polyp_presence_in_xml(polyp_xml_path)

            current_video_frame_data[frame_idx] = {
                'temporal_label': temporal_label,
                'polyp_present': polyp_present,
                'frame_path': frame_path
            }
        video_frame_metadata[video_id] = current_video_frame_data

    print("--- Phase 2: Generating triplets for the split ---")
    for video_id, frames_data in tqdm(video_frame_metadata.items(), desc="Generating triplets"):
        # ... (The triplet generation logic is the same as before) ...
        for frame_idx, data in frames_data.items():
            frame_path = data['frame_path']
            positive_labels_for_frame = set()
            if data['temporal_label'] != 'background':
                positive_labels_for_frame.add(data['temporal_label'])
            if data['polyp_present']:
                positive_labels_for_frame.add('polyp')

            for label in positive_labels_for_frame:
                query_text = LABEL_TO_TEXT_QUERY.get(label, label)
                all_triplets.append({'frame_path': frame_path, 'text_query': query_text, 'relevance_label': 1})

            negative_candidate_labels = [lbl for lbl in all_query_labels if lbl not in positive_labels_for_frame]
            num_neg_samples = min(len(negative_candidate_labels), 2)

            if num_neg_samples > 0:
                chosen_neg_labels = random.sample(negative_candidate_labels, num_neg_samples)
                for neg_label in chosen_neg_labels:
                    neg_query_text = LABEL_TO_TEXT_QUERY.get(neg_label, neg_label)
                    all_triplets.append({'frame_path': frame_path, 'text_query': neg_query_text, 'relevance_label': 0})

    return pd.DataFrame(all_triplets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate training/validation/test triplet CSVs from pre-defined split files.")
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
    split_file_path = os.path.join(REAL_COLON_REPO_DIR, f"{args.split}_videos.txt")
    output_csv_path = os.path.join(OUTPUT_DIR, f"{args.split}_triplets.csv")

    print(f"--- Preparing data for '{args.split}' split ---")
    print(f"Reading video IDs from: {split_file_path}")

    try:
        with open(split_file_path, 'r') as f:
            video_ids_for_this_split = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"Error: Split file not found at {split_file_path}", file=sys.stderr)
        print("Please run create_splits.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(video_ids_for_this_split)} videos for the '{args.split}' split.")

    # Generate the triplets DataFrame for this split
    df_triplets = prepare_split_data(video_ids_for_this_split)

    # Save the final CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_triplets.to_csv(output_csv_path, index=False)
    print(f"\nGenerated {len(df_triplets)} training triplets. Saved to {output_csv_path}")
    print("--- Done! ---")
