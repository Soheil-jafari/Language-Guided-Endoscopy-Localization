# dataset_preprocessing/create_splits.py
# This is a one-time script to create a reproducible train/validation/test split
# for the Cholec80 dataset based on video filenames.

import os
import random
import argparse
import sys
import re  # For extracting numbers from filenames

# We need to import config to know where to save the split files
import config


def create_video_splits(video_dir, train_split=0.7, val_split=0.15):
    """
    Finds all video files, extracts their numerical IDs, shuffles them,
    and splits them into train, val, and test sets.

    Args:
        video_dir (str): The path to the 'videos' directory containing the video files.
        train_split (float): The proportion of data for training.
        val_split (float): The proportion of data for validation.
    """
    print(f"Scanning for video files in: {video_dir}")

    try:
        # List all files in the directory
        all_files = os.listdir(video_dir)
        # Filter for common video file extensions
        video_files = [f for f in all_files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    except FileNotFoundError:
        print(f"Error: Directory not found at {video_dir}. Please check the path.", file=sys.stderr)
        return

    if not video_files:
        print(f"Error: No video files found in {video_dir}. Cannot create splits.", file=sys.stderr)
        return

    # Extract the numerical ID from each video filename (e.g., 'video01.mp4' -> '1')
    video_ids = []
    for f in video_files:
        # Use regex to find the number in the filename
        match = re.search(r'\d+', f)
        if match:
            video_ids.append(match.group(0))

    video_ids = sorted(list(set(video_ids)))  # Get unique, sorted IDs

    print(f"Found a total of {len(video_ids)} unique video IDs.")

    # Shuffle the list of video IDs for a random split
    random.shuffle(video_ids)

    # Calculate split indices
    train_end_idx = int(len(video_ids) * train_split)
    val_end_idx = train_end_idx + int(len(video_ids) * val_split)

    # Create the lists of IDs for each set
    train_ids = video_ids[:train_end_idx]
    val_ids = video_ids[train_end_idx:val_end_idx]
    test_ids = video_ids[val_end_idx:]

    print("\nSplit Summary:")
    print(f"  Training videos: {len(train_ids)}")
    print(f"  Validation videos: {len(val_ids)}")
    print(f"  Test videos: {len(test_ids)}")

    # --- Save the lists to .txt files in the directory specified in config.py ---
    output_dir = config.SPLIT_FILES_DIR
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

    with open(os.path.join(output_dir, 'train_videos.txt'), 'w') as f:
        for video_id in train_ids:
            f.write(f"{video_id}\n")

    with open(os.path.join(output_dir, 'val_videos.txt'), 'w') as f:
        for video_id in val_ids:
            f.write(f"{video_id}\n")

    with open(os.path.join(output_dir, 'test_videos.txt'), 'w') as f:
        for video_id in test_ids:
            f.write(f"{video_id}\n")

    print(f"\nSplit files (train_videos.txt, val_videos.txt, test_videos.txt) saved to: {output_dir}")
    print("--- Split Creation Complete! ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create train/val/test splits for the Cholec80 dataset.")
    parser.add_argument(
        '--video_dir',
        type=str,
        required=True,
        help="Path to the directory containing the video files (e.g., '.../cholec80_dataset/videos')."
    )
    args = parser.parse_args()

    random.seed(42)  # For reproducible splits

    create_video_splits(args.video_dir)
