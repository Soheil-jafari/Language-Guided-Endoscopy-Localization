# inference.py (compatible with your class-based project_config)
import torch, cv2, numpy as np, argparse, os, sys, csv, re
from transformers import AutoTokenizer
from pathlib import Path
import project_config
from project_config import config as cfg
from models import LocalizationFramework
from torchvision.transforms.functional import normalize

cfg = project_config.config  # convenience alias

def sanitize(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_.-]+', '_', s.strip())[:80]


def process_video_for_inference(video_path, num_frames, resize):
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found at {video_path}", file=sys.stderr)
        return None, None, None

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1:
        fps = 25.0  # Set a default FPS if reading fails

    if total_frames == 0:
        print(f"❌ Error: Video seems empty or corrupted: {video_path}", file=sys.stderr)
        cap.release()
        return None, None, None

    frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    frames = []
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (resize, resize))
        frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Apply ImageNet normalization (this is also crucial)
        frame = normalize(frame, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        frames.append(frame)
    cap.release()

    if not frames:
        print("❌ Error: Could not extract any frames from the video.", file=sys.stderr)
        return None, None, None

    # --- THIS IS THE CORRECTED TENSOR SHAPING ---
    # Stack along a new dimension to get (T, C, H, W), then permute to (C, T, H, W)
    video_tensor = torch.stack(frames, dim=0).permute(1, 0, 2, 3)
    # Add the batch dimension to get the final (B, C, T, H, W) shape
    video_tensor = video_tensor.unsqueeze(0)

    times_sec = (frame_indices / fps).astype(np.float32)
    return video_tensor, frame_indices, times_sec

def probs_to_segments(probabilities, times_sec, thr=0.5, min_dur=0.4, merge_gap=0.2):
    probs = np.asarray(probabilities).reshape(-1)
    T = len(probs)
    active = probs >= thr

    # find contiguous active runs
    segments = []
    i = 0
    while i < T:
        if active[i]:
            j = i + 1
            while j < T and active[j]:
                j += 1
            s = float(times_sec[i])
            # end: halfway to next sample if available, else current time
            e = float((times_sec[j - 1] + (times_sec[j] if j < T else times_sec[j - 1])) / 2.0)
            seg_score = float(probs[i:j].mean())
            segments.append([s, e, seg_score])
            i = j
        else:
            i += 1

    # merge close segments
    merged = []
    for seg in segments:
        if not merged:
            merged.append(seg)
        else:
            ps, pe, psc = merged[-1]
            cs, ce, csc = seg
            if cs - pe <= merge_gap:
                dur1 = max(1e-6, pe - ps)
                dur2 = max(1e-6, ce - cs)
                avg = (psc * dur1 + csc * dur2) / (dur1 + dur2)
                merged[-1] = [ps, ce, avg]
            else:
                merged.append(seg)

    # filter by min duration
    final_segments = [seg for seg in merged if (seg[1] - seg[0]) >= min_dur]
    return final_segments

def run_inference(args):
    print("--- Starting Inference ---")

    # device
    device_str = cfg.TRAIN.DEVICE if hasattr(cfg, "TRAIN") else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(getattr(cfg.TRAIN, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    # model
    print("Initializing model architecture...")
    model = LocalizationFramework(config=cfg).to(device)

    # weights
    if os.path.exists(args.checkpoint_path):
        print(f"Loading trained weights from {args.checkpoint_path}...")
        state = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print("Model weights loaded successfully.")
    else:
        print(f"Warning: Checkpoint not found at '{args.checkpoint_path}'. Running with random init.")

    model.eval()

    # inputs
    print(f"Processing video: {args.video_path}")
    num_frames = getattr(cfg.DATA, "NUM_INFERENCE_FRAMES", 50)
    resize = getattr(cfg, "INFER_IMG_SIZE", getattr(cfg.DATA, "TRAIN_CROP_SIZE", 224))
    clip, frame_idx, times_sec = process_video_for_inference(args.video_path, num_frames, resize)
    if clip is None:
        return
    clip = clip.to(device)

    text_model_name = getattr(cfg.MODEL, "TEXT_ENCODER_MODEL", "openai/clip-vit-base-patch32")
    max_text_len = getattr(cfg.DATA, "MAX_TEXT_LENGTH", 77)
    print(f"Tokenizing text query with {text_model_name}: '{args.text_query}'")
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    text_inputs = tokenizer(
        args.text_query, padding="max_length", truncation=True,
        max_length=max_text_len, return_tensors="pt"
    ).to(device)

    # forward
    print("Running model forward pass...")
    with torch.no_grad():
        refined_scores, raw_scores, _, _, _, _ = model(clip, text_inputs.input_ids, text_inputs.attention_mask)

    refined_scores = refined_scores.squeeze(0).cpu().numpy()  # [T, 1] -> (T, 1)
    probabilities = 1 / (1 + np.exp(-refined_scores))
    probabilities = probabilities.reshape(-1)  # (T,)

    print("--- Inference Complete ---")

    # outputs
    out_root = Path(cfg.OUTPUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)
    run_name = f"{Path(args.video_path).stem}__{sanitize(args.text_query)}"
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # scores.csv
    scores_csv = run_dir / "scores.csv"
    with open(scores_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "time_sec", "prob"])
        for i, (t, p) in enumerate(zip(times_sec, probabilities)):
            w.writerow([int(frame_idx[i]), float(t), float(p)])

    # segments
    thr = float(getattr(cfg, "SEGMENT_THRESHOLD", 0.5))
    min_dur = float(getattr(cfg, "MIN_SEGMENT_DURATION", 0.4))
    merge_gap = float(getattr(cfg, "MERGE_GAP", 0.2))
    segments = probs_to_segments(probabilities, times_sec, thr=thr, min_dur=min_dur, merge_gap=merge_gap)

    seg_csv = run_dir / "pred_segments.csv"
    with open(seg_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start_sec", "end_sec", "score"])
        for s, e, sc in segments:
            w.writerow([f"{s:.3f}", f"{e:.3f}", f"{sc:.4f}"])

    print("\n--- Results ---")
    print(f"Query: '{args.text_query}'")
    print(f"Saved per-frame scores to: {scores_csv}")
    print(f"Saved predicted segments to: {seg_csv}")
    if len(segments) == 0:
        print("No segments found with current threshold. Consider lowering SEGMENT_THRESHOLD.")

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Run inference for language-guided video localization (single video).")
    ap.add_argument("--video_path", required=True, type=str, help="Path to the input video file.")
    ap.add_argument("--text_query", required=True, type=str, help="Natural-language query.")
    ap.add_argument("--checkpoint_path", required=True, type=str, help="Path to best_model.pth")
    args = ap.parse_args()
    run_inference(args)
