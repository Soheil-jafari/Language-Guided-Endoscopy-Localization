# inference.py
# This script is for running inference with a trained model checkpoint.
# It can be used for local testing on a sample video or for generating final results.

import torch
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import os
from transformers import AutoTokenizer

# Import our project components
import config
from models import LocalizationFramework


def process_video_for_inference(video_path, target_fps, num_frames):
    """
    Loads a video, resamples it to a target FPS, and extracts a fixed number of frames.

    Args:
        video_path (str): Path to the video file.
        target_fps (int): The desired frames per second.
        num_frames (int): The total number of frames to extract for the clip.

    Returns:
        torch.Tensor: A tensor representing the video clip of shape [1, T, C, H, W].
                      Returns None if the video cannot be processed.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    frame_indices = np.linspace(0, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1, num=num_frames, dtype=int)

    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        # Preprocess frame: BGR to RGB, resize, normalize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0
        # Add normalization if your model expects it

        frames.append(frame)

    cap.release()

    if not frames:
        print("Error: Could not extract any frames from the video.")
        return None

    # Stack frames into a single tensor and add a batch dimension
    video_tensor = torch.stack(frames).unsqueeze(0)  # [1, T, C, H, W]
    return video_tensor


def run_inference(args):
    """
    Main function to run the inference process.
    """
    print("--- Starting Inference ---")
    device = torch.device(config.DEVICE)

    # --- 1. Load Model ---
    print("Loading model architecture...")
    model = LocalizationFramework().to(device)

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {args.checkpoint_path}")

    print(f"Loading trained weights from {args.checkpoint_path}...")
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully.")

    # --- 2. Prepare Inputs ---
    print(f"Processing video: {args.video_path}")
    video_clip = process_video_for_inference(
        args.video_path,
        config.TARGET_FPS,
        config.NUM_INFERENCE_FRAMES
    )
    if video_clip is None:
        return

    video_clip = video_clip.to(device)

    print(f"Tokenizing text query: '{args.text_query}'")
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    text_inputs = tokenizer(
        args.text_query,
        padding="max_length",
        truncation=True,
        max_length=config.MAX_TEXT_LENGTH,
        return_tensors="pt"
    ).to(device)

    # --- 3. Run Model Prediction ---
    print("Running model forward pass...")
    with torch.no_grad():
        raw_scores, refined_scores = model(video_clip, text_inputs.input_ids, text_inputs.attention_mask)

    # The output is for a batch of 1, so squeeze it
    refined_scores = refined_scores.squeeze(0).cpu().numpy()  # [T, 1]

    # Apply sigmoid to convert logits to probabilities
    probabilities = 1 / (1 + np.exp(-refined_scores))

    print("--- Inference Complete ---")

    # --- 4. Display Results ---
    print("\n--- Results ---")
    print(f"Query: '{args.text_query}'")
    print("Frame-by-frame relevance probabilities:")

    for i, prob in enumerate(probabilities):
        print(f"Frame {i:03d}: Probability = {prob[0]:.4f}")

    # Find the frame with the highest relevance
    best_frame_idx = np.argmax(probabilities)
    max_prob = np.max(probabilities)
    print(f"\nHighest relevance found at Frame {best_frame_idx} with probability {max_prob:.4f}")

    # TODO: Add visualization logic here, e.g., save an output video with scores
    # overlaid, or save the frame with the highest score.


if __name__ == '__main__':
    # --- Argument Parser for Command-Line Usage ---
    parser = argparse.ArgumentParser(description="Run inference for language-guided video localization.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--text_query", type=str, required=True, help="The natural language query to search for.")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the trained model checkpoint (.pth file).")

    args = parser.parse_args()

    # Create a dummy checkpoint if it doesn't exist, for testing purposes
    if not os.path.exists(args.checkpoint_path):
        print(f"Warning: Checkpoint not found. Creating a dummy checkpoint at {args.checkpoint_path} for testing.")
        os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
        # We need to initialize the model to save its state_dict
        dummy_model = LocalizationFramework()
        torch.save(dummy_model.state_dict(), args.checkpoint_path)
        del dummy_model

    run_inference(args)
