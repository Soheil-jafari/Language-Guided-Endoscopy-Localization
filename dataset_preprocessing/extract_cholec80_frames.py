import os
import cv2
import argparse
from tqdm import tqdm
import sys
import re
# Ensure project_config can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from project_config import config


def extract_frames_from_video(video_path, output_dir_for_video, frame_prefix="frame_"):
    """
    Extracts frames from a single video and saves them to the specified output directory.
    Frames are named like 'frame_0000000.jpg'.
    """
    os.makedirs(output_dir_for_video, exist_ok=True)  # Ensure video's output directory exists

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}", file=sys.stderr)
        return 0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_extracted = 0

    for i in tqdm(range(frame_count), desc=f"Extracting frames for {os.path.basename(video_path)}"):
        ret, frame = cap.read()
        if not ret:
            # print(f"Warning: Could not read frame {i} from {video_path}", file=sys.stderr)
            continue  # Skip to the next iteration if frame can't be read

        frame_filename = f"{frame_prefix}{i:07d}.jpg"  # e.g., frame_0000000.jpg
        frame_filepath = os.path.join(output_dir_for_video, frame_filename)

        # Check if file already exists to avoid re-saving if not forced
        if not os.path.exists(frame_filepath):  # Simple check to avoid overwriting by default
            cv2.imwrite(frame_filepath, frame)
            frames_extracted += 1
        else:
            # Optionally print a message if skipping, or add --force-overwrite argument
            pass

    cap.release()
    return frames_extracted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extracts frames from Cholec80 videos into a structured directory."
    )
    parser.add_argument(
        '--cholec80_videos_dir',
        type=str,
        default=os.path.join(config.ML_SERVER_HOME, "data", "unified_medical_videos", "raw_downloads",
                             "cholec80_dataset", "videos"),
        help="Path to the directory containing original Cholec80 video files (e.g., video01.mp4)."
    )
    parser.add_argument(
        '--output_frames_dir',
        type=str,
        default=config.EXTRACTED_FRAMES_DIR,
        help="Root directory where extracted frames will be saved (e.g., .../extracted_frames/CHOLEC80__video01/frame_0000000.jpg)."
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help="If set, existing frame files will be overwritten."
    )
    args = parser.parse_args()

    print(f"--- Starting Cholec80 Frame Extraction ---")
    print(f"Source videos directory: {args.cholec80_videos_dir}")
    print(f"Output frames root directory: {args.output_frames_dir}")
    print(f"Overwrite existing frames: {args.overwrite}")

    if not os.path.exists(args.cholec80_videos_dir):
        print(f"Error: Source videos directory not found: {args.cholec80_videos_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_frames_dir, exist_ok=True)

    video_files = sorted([f for f in os.listdir(args.cholec80_videos_dir) if f.endswith('.mp4')])

    if not video_files:
        print(f"No .mp4 video files found in {args.cholec80_videos_dir}", file=sys.stderr)
        sys.exit(1)

    total_extracted_frames = 0

    # Loop through each video file to extract frames
    for video_filename in video_files:
        original_video_path = os.path.join(args.cholec80_videos_dir, video_filename)

        # Extract the numeric part (e.g., '01' from 'video01.mp4')
        video_num_match = re.match(r'video(\d+)\.mp4', video_filename)
        if not video_num_match:
            print(f"Skipping {video_filename}: Does not match 'videoXX.mp4' naming convention.", file=sys.stderr)
            continue

        video_number = int(video_num_match.group(1))

        # Construct the output subdirectory name: CHOLEC80__videoXX (e.g., CHOLEC80__video01)
        output_video_subdir_name = f"CHOLEC80__video{video_number:02d}"
        output_dir_for_current_video = os.path.join(args.output_frames_dir, output_video_subdir_name)

        os.makedirs(output_dir_for_current_video, exist_ok=True)  # Create if not exists

        print(f"\nProcessing video: {video_filename}")

        # Override file existence check if --overwrite is set
        if args.overwrite:
            # If overwrite, clean the directory first to ensure fresh extraction
            for existing_frame in os.listdir(output_dir_for_current_video):
                if existing_frame.endswith('.jpg'):
                    os.remove(os.path.join(output_dir_for_current_video, existing_frame))
            extracted_count = extract_frames_from_video(original_video_path, output_dir_for_current_video)
        else:
            # Without overwrite, the extract_frames_from_video function itself checks existence
            extracted_count = extract_frames_from_video(original_video_path, output_dir_for_current_video)

        total_extracted_frames += extracted_count
        print(f"Extracted {extracted_count} frames for {video_filename}")

    print(f"\n--- Frame Extraction Complete! ---")
    print(f"Total frames extracted/processed: {total_extracted_frames}")
    print(f"Frames saved to: {args.output_frames_dir}")