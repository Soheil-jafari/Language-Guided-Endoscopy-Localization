import argparse
from pathlib import Path
import numpy as np
from transformers import XCLIPProcessor
import torch
from tqdm import tqdm
from xclip.model import XCLIPWrapper
from xclip.utils import load_project_config
from xclip.data import get_frame_paths, load_frames_by_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--frame-glob", type=str, default="*.jpg")
    parser.add_argument("--model-name", type=str, default="microsoft/xclip-base-patch32")
    parser.add_argument("--ckpt", type=str, default="", help="Path to the fine-tuned checkpoint")
    args = parser.parse_args()

    config = load_project_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = XCLIPWrapper(model_name=args.model_name).to(device).eval()

    if args.ckpt and Path(args.ckpt).exists():
        print(f"[INFO] Loading fine-tuned checkpoint from: {args.ckpt}")
        state = torch.load(args.ckpt, map_location=device)
        wrapper.load_state_dict(state["model"], strict=False)
    else:
        print("[INFO] No checkpoint provided. Using the base pre-trained model.")

    processor = wrapper.processor

    frame_paths = get_frame_paths(config.EXTRACTED_FRAMES_DIR, args.video_id, args.frame_glob)
    n = len(frame_paths)
    if n == 0:
        print(f"Error: No frames found for video-id '{args.video_id}'")
        return

    starts = list(range(0, n - args.num_frames + 1, args.stride))
    scores = []

    # <-- MODIFIED: The for loop is now wrapped with tqdm for a progress bar -->
    for s in tqdm(starts, desc=f"Scoring windows for '{args.video_id}'"):
        idxs = list(range(s, s + args.num_frames))
        frames = load_frames_by_indices(frame_paths, idxs)
        inputs = processor(videos=[frames], text=[args.query], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            video_embeds, text_embeds = wrapper(**inputs)
            video_embeds = video_embeds / video_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            sc = float((video_embeds @ text_embeds.T).item())

        scores.append(sc)

    frame_scores = np.zeros(n, dtype=float)
    counts = np.zeros(n, dtype=float)
    for s, sc in zip(starts, scores):
        e = min(n, s + args.num_frames)
        frame_scores[s:e] += sc
        counts[s:e] += 1.0

    counts[counts == 0] = 1.0
    frame_scores /= counts

    print("\n--- Inference Results ---")
    print(f"Video: {args.video_id}")
    print(f"Query: '{args.query}'")
    print(f"Total Frames: {n}")
    print(f"Mean Score: {frame_scores.mean():.4f}")
    print(f"Max Score: {frame_scores.max():.4f}")
    print(f"Top 5 Frame Scores:")
    top_indices = np.argsort(frame_scores)[-5:][::-1]
    for i in top_indices:
        print(f"  - Frame {i}: {frame_scores[i]:.4f}")


if __name__ == "__main__":
    main()