\
import argparse
from pathlib import Path
import numpy as np
from transformers import XCLIPProcessor
import torch
from xclip.model import XCLIPWrapper
from xclip.utils import load_project_config
from xclip.data import get_frame_paths, load_frames_by_indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--frame-glob", type=str, default="*.jpg")
    parser.add_argument("--model-name", type=str, default="microsoft/xclip-base-patch32")
    args = parser.parse_args()

    config = load_project_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = XCLIPWrapper(model_name=args.model_name).to(device).eval()
    processor = wrapper.processor

    frame_paths = get_frame_paths(config.EXTRACTED_FRAMES_DIR, args.video_id, args.frame_glob)
    n = len(frame_paths)
    starts = list(range(0, max(1, n - args.num_frames + 1), args.stride))
    scores = []
    for s in starts:
        idxs = list(range(s, s + args.num_frames))
        frames = load_frames_by_indices(frame_paths, idxs)
        inputs = processor(videos=[frames], text=[args.query], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = wrapper.model(**inputs)
            sc = float(out.logits_per_video[0,0].detach().cpu().item())
        scores.append(sc)
    # splat to frames
    frame_scores = np.zeros(n, dtype=float); counts = np.zeros(n, dtype=float)
    for s, sc in zip(starts, scores):
        e = min(n, s + args.num_frames)
        frame_scores[s:e] += sc; counts[s:e] += 1
    counts[counts == 0] = 1
    frame_scores /= counts
    print(f"Frames: {n}, mean score: {frame_scores.mean():.4f}, max: {frame_scores.max():.4f}")

if __name__ == "__main__":
    main()
