import torch, cv2, numpy as np, argparse, os, sys, csv, re
from transformers import AutoTokenizer
from pathlib import Path
import project_config
from project_config import config as cfg
from models import LocalizationFramework
from torchvision.transforms.functional import normalize
import contextlib
from tqdm import tqdm
import json
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
    for i in tqdm(frame_indices, desc="Processing Video Frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (resize, resize))
        frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Apply ImageNet normalization
        frame = normalize(frame, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        frames.append(frame)
    cap.release()

    if not frames:
        print("❌ Error: Could not extract any frames from the video.", file=sys.stderr)
        return None, None, None

    # Stack along a new dimension to get (T, C, H, W), then permute to (C, T, H, W)
    video_tensor = torch.stack(frames, dim=0).permute(1, 0, 2, 3)
    # Add the batch dimension to get the final (B, C, T, H, W) shape
    video_tensor = video_tensor.unsqueeze(0)

    times_sec = (frame_indices / fps).astype(np.float32)
    return video_tensor, frame_indices, times_sec, fps


def probs_to_segments(probabilities, times_sec, frame_indices, fps,
                      thr=0.5, min_dur=0.4, merge_gap=None):
    """
    Converts sparse frame probabilities into temporal segments by:
    1) Building Voronoi-like bins around each sampled time (midpoints to neighbors).
    2) Marking bins 'active' where prob >= thr.
    3) Merging consecutive active bins and also any gaps <= merge_gap.
    """
    probabilities = np.asarray(probabilities)
    times_sec = np.asarray(times_sec)

    if len(probabilities) == 0:
        return []

    # Build bin edges as midpoints between consecutive sample times
    if len(times_sec) == 1:
        # Single sample: fall back to a tiny bin
        edges = np.array([times_sec[0] - 0.5/fps, times_sec[0] + 0.5/fps], dtype=float)
    else:
        mids = 0.5 * (times_sec[:-1] + times_sec[1:])
        # First edge extrapolates backwards by half the first gap; last edge forwards by half the last gap
        first_gap = times_sec[1] - times_sec[0]
        last_gap  = times_sec[-1] - times_sec[-2]
        edges = np.concatenate([[times_sec[0] - 0.5*first_gap], mids, [times_sec[-1] + 0.5*last_gap]])

    # Choose a sensible default merge_gap from sampling gaps if not provided
    if merge_gap is None:
        if len(times_sec) > 1:
            typical_gap = float(np.median(np.diff(times_sec)))
            merge_gap = typical_gap * 0.1  # small fraction of sampling gap
        else:
            merge_gap = 0.0

    # Build initial bins
    bins = []
    for i, p in enumerate(probabilities):
        start = edges[i]
        end = edges[i+1]
        active = (p >= thr)
        bins.append((start, end, p, active))

    # Merge consecutive active bins
    segments = []
    cur_start, cur_end, ps = None, None, []

    for (start, end, p, active) in bins:
        if active:
            if cur_start is None:
                cur_start, cur_end, ps = start, end, [p]
            else:
                # If touching or with a tiny gap, extend
                if start - cur_end <= merge_gap:
                    cur_end = end
                    ps.append(p)
                else:
                    segments.append((cur_start, cur_end, float(np.mean(ps))))
                    cur_start, cur_end, ps = start, end, [p]
        else:
            # Close any ongoing active run
            if cur_start is not None:
                segments.append((cur_start, cur_end, float(np.mean(ps))))
                cur_start, cur_end, ps = None, None, []

    if cur_start is not None:
        segments.append((cur_start, cur_end, float(np.mean(ps))))

    # Enforce minimum duration
    segments = [(s, e, sc) for (s, e, sc) in segments if (e - s) >= float(min_dur)]
    return segments


def run_inference(args):
    print("--- Starting Inference (sliding-window) ---")

    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    # optional multi-GPU
    if int(args.use_dataparallel) == 1 and torch.cuda.device_count() > 1:
        print(f"Using DataParallel across {torch.cuda.device_count()} GPUs...")
        model = torch.nn.DataParallel(model)

    model.eval()

    # --- Video sampling (dense, constant-rate) ---
    video_path = args.video_path
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found at {video_path}", file=sys.stderr)
        return

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps < 1:
        src_fps = 25.0

    # sample at a target fps to control density & memory
    target_fps = float(args.target_fps)
    step = max(1, int(round(src_fps / max(target_fps, 1e-6))))  # sample every `step` frames
    resize = getattr(cfg, "INFER_IMG_SIZE", getattr(cfg.DATA, "TRAIN_CROP_SIZE", 224))

    sampled_idx = np.arange(0, total_frames, step, dtype=int)
    print(f"[INFO] src_fps={src_fps:.2f}, target_fps={target_fps:.2f}, planned_frames={len(sampled_idx)}")

    # Guardrail: cap sample count to ~6000 frames to avoid huge RAM usage
    if len(sampled_idx) > 6000:
        factor = int(np.ceil(len(sampled_idx) / 6000))
        sampled_idx = sampled_idx[::factor]
        print(f"[INFO] Downsampling further by factor {factor} -> {len(sampled_idx)} frames")

    frames_uint8 = []
    for i in tqdm(sampled_idx, desc="Sampling frames", dynamic_ncols=True):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (resize, resize), interpolation=cv2.INTER_AREA)
        frames_uint8.append(frame)  # store as uint8 to save RAM
    cap.release()

    if len(frames_uint8) == 0:
        print("❌ Error: Could not extract any frames from the video.", file=sys.stderr)
        return

    N = len(frames_uint8)
    times_sec = (sampled_idx / src_fps).astype(np.float32)

    # --- Text encoding (match training CLIP text model) ---
    text_model_name = getattr(cfg.MODEL, "TEXT_ENCODER_MODEL", "openai/clip-vit-base-patch32")
    max_text_len = getattr(cfg.DATA, "MAX_TEXT_LENGTH", 77)
    print(f"Tokenizing text query with {text_model_name}: '{args.text_query}'")
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    text_inputs = tokenizer(
        args.text_query, padding="max_length", truncation=True,
        max_length=max_text_len, return_tensors="pt"
    ).to(device)

    # --- Sliding-window over sampled frames with training T ---
    window_T = int(args.window_T) if args.window_T is not None else int(getattr(cfg.DATA, "NUM_FRAMES", 16))
    window_stride = int(args.window_stride) if args.window_stride is not None else max(1, window_T // 2)
    if window_T > N:
        # fallback: shrink window to available frames
        window_T = N

    sum_probs = np.zeros(N, dtype=np.float32)
    sum_counts = np.zeros(N, dtype=np.float32)

    print("Running sliding-window forward passes (batched)...")
    window_starts = list(range(0, max(1, N - window_T + 1), window_stride))
    if len(window_starts) == 0:
        window_starts = [0]

    B_win = int(args.batch_size_windows)
    total_batches = (len(window_starts) + B_win - 1) // B_win

    # autocast for speed (fp16) on GPU; no-op on CPU
    autocast_ctx = torch.cuda.amp.autocast if device.type == "cuda" else contextlib.nullcontext

    # Force tqdm to stderr; also print a manual fallback every 10 batches
    progress_iter = tqdm(
        range(0, len(window_starts), B_win),
        total=total_batches,
        desc="Scoring windows",
        dynamic_ncols=True,
        mininterval=0.3,
        file=sys.stderr,
        disable=False
    )
    sum_probs = np.zeros(N, dtype=np.float32)
    sum_counts = np.zeros(N, dtype=np.float32)
    sum_unc = np.zeros(N, dtype=np.float32)  # only used if uncertainty available
    sum_alpha = np.zeros(N, dtype=np.float32)
    sum_beta = np.zeros(N, dtype=np.float32)

    with torch.no_grad(), autocast_ctx():
        for bi, b in enumerate(progress_iter):
            batch_starts = window_starts[b:b + B_win]

            # Build a batch of windows: (B, C, T, H, W)
            batch_clips = []
            for s in batch_starts:
                s = min(s, N - window_T)  # guard tail
                clip_imgs = []
                for j in range(s, s + window_T):
                    img = torch.from_numpy(frames_uint8[j]).permute(2, 0, 1).to(torch.float32) / 255.0
                    img = normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    clip_imgs.append(img)  # (C, H, W)
                # stack into (1, C, T, H, W)
                clip_np = torch.stack(clip_imgs, dim=1).unsqueeze(0)
                batch_clips.append(clip_np)
            batch_clips = torch.cat(batch_clips, dim=0).to(device)  # (B, C, T, H, W)

            # Repeat text inputs to match batch
            B = batch_clips.size(0)
            rep_ids = text_inputs.input_ids.repeat(B, 1)
            rep_mask = text_inputs.attention_mask.repeat(B, 1)

            # Forward
            outputs = model(batch_clips, rep_ids, rep_mask)

            # Backward-compatible unpacking:
            alpha_beta = None
            xai_maps = None
            # Try the improved-model structure first
            if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
                # Common: refined_scores, raw_scores are first two
                refined_scores = outputs[0]
                raw_scores = outputs[1]
                # Heuristics: evidential params and attention maps (if present)
                if len(outputs) >= 3 and outputs[2] is not None:
                    alpha_beta = outputs[2]  # shape [B, T, 2] -> (alpha, beta) pre-softplus/ReLU
                if len(outputs) >= 4 and outputs[3] is not None:
                    xai_maps = outputs[3]  # list/tenor of attention maps per frame or [B, T, H', W']
            else:
                # Fallback: older tuple
                refined_scores, raw_scores = outputs, None

            # Convert to probabilities
            def _beta_mean_var(ab):
                # ab: (..., 2) => alpha,beta raw (can be pre-activation)
                a = torch.relu(ab[..., 0]) + 1.0
                b = torch.relu(ab[..., 1]) + 1.0
                mean = a / (a + b)
                var = (a * b) / (((a + b) ** 2) * (a + b + 1.0))
                return mean, var, a, b

            # Convert to probs (+ optional uncertainty)
            if alpha_beta is not None and int(args.save_uncertainty) == 1:
                m, v, a_t, b_t = _beta_mean_var(alpha_beta)  # shapes [B, T]
                probs_batch = m.detach().cpu().numpy()
                unc_batch = v.detach().cpu().numpy()
                alpha_batch = a_t.detach().cpu().numpy()
                beta_batch = b_t.detach().cpu().numpy()
            else:
                probs_batch = torch.sigmoid(refined_scores).squeeze(-1).detach().cpu().numpy()  # (B, T)
                unc_batch = None
                alpha_batch = None
                beta_batch = None


            # Accumulate per-frame probabilities
            for k, s in enumerate(batch_starts):
                s = min(s, N - window_T)
                sum_probs[s:s + window_T] += probs_batch[k]
                sum_counts[s:s + window_T] += 1.0
                if unc_batch is not None:
                    sum_unc[s:s + window_T] += unc_batch[k]
                    sum_alpha[s:s + window_T] += alpha_batch[k]
                    sum_beta[s:s + window_T] += beta_batch[k]

            # Manual fallback progress (stderr) in case bars are suppressed
            if (bi % 10 == 0) or (bi + 1 == total_batches):
                print(f"[Scoring windows] {bi + 1}/{total_batches}", flush=True, file=sys.stderr)

    # Final per-sampled-frame probabilities
    probabilities = sum_probs / np.maximum(sum_counts, 1.0)

    uncertainties = None
    alphas = betas = None
    if np.any(sum_unc):   uncertainties = sum_unc / np.maximum(sum_counts, 1.0)
    if np.any(sum_alpha): alphas = sum_alpha / np.maximum(sum_counts, 1.0)
    if np.any(sum_beta):  betas = sum_beta / np.maximum(sum_counts, 1.0)

    gaps = np.diff(times_sec) if len(times_sec) > 1 else np.array([1.0 / max(1e-6, float(25.0))])
    med_gap = float(np.median(gaps)) if gaps.size else 1.0 / 25.0
    win = max(1, int(round(3.0 / max(med_gap, 1e-6))))
    if win > 1:
        kernel = np.ones(win, dtype=np.float32) / float(win)
        probabilities = np.convolve(probabilities, kernel, mode="same")

    print(f"[DEBUG] USE_UNCERTAINTY={getattr(cfg.MODEL, 'USE_UNCERTAINTY', False)}")
    print(
        f"[DEBUG] probs: min={float(np.min(probabilities)):.4f}, "
        f"max={float(np.max(probabilities)):.4f}, "
        f"mean={float(np.mean(probabilities)):.4f}"
    )

    print("--- Inference Complete ---")

    # --- Save outputs ---
    out_root = Path(cfg.OUTPUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)
    run_name = f"{Path(args.video_path).stem}__{sanitize(args.text_query)}"
    run_dir = out_root / run_name
    sfx = f"_{args.suffix.strip()}" if args.suffix else ""
    run_dir.mkdir(parents=True, exist_ok=True)

    # scores.csv (per-sampled-frame)
    scores_csv = run_dir / f"per_frame{sfx}.csv"
    with open(scores_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["frame_idx", "time_sec", "prob"]
        if alphas is not None and betas is not None:
            header += ["alpha", "beta"]
        if uncertainties is not None:
            header += ["uncertainty"]
        w.writerow(header)
        for i in range(N):
            row = [int(sampled_idx[i]), float(times_sec[i]), float(probabilities[i])]
            if alphas is not None and betas is not None:
                row += [float(alphas[i]), float(betas[i])]
            if uncertainties is not None:
                row += [float(uncertainties[i])]
            w.writerow(row)
    print(f"Saved per-frame scores to: {scores_csv}")

    # --- Segmentation ---
    thr = float(args.segment_threshold)
    min_dur = float(args.segment_min_dur)
    merge_gap = args.segment_merge_gap  # None => auto 10% of median gap

    gaps = np.diff(times_sec) if len(times_sec) > 1 else np.array([])
    typ_gap = float(np.median(gaps)) if gaps.size else 0.0
    auto_merge_gap = (typ_gap * 0.1) if merge_gap is None else merge_gap
    print(f"[DEBUG] frames={N}, src_fps={src_fps:.3f}, target_fps={target_fps:.3f}, "
          f"med_gap={typ_gap:.3f}s, merge_gap={auto_merge_gap}, min_dur={min_dur}")

    segments = probs_to_segments(probabilities, times_sec, sampled_idx, src_fps,
                                 thr=thr, min_dur=min_dur, merge_gap=merge_gap)

    seg_csv = run_dir / f"pred_segments{sfx}.csv"
    with open(seg_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start_sec", "end_sec", "score"])
        for s, e, sc in segments:
            w.writerow([f"{s:.3f}", f"{e:.3f}", f"{sc:.4f}"])

    segments_json  = run_dir / f"segments{sfx}.json"
    with open(segments_json, "w", encoding="utf-8") as f:
        json.dump([{"start_sec": float(s), "end_sec": float(e), "score": float(sc)} for (s, e, sc) in segments],
                  f, indent=2)

    manifest = {
        "experiment_tag": args.experiment_tag,
        "video": str(args.video_path),
        "query": args.text_query,
        "checkpoint": str(args.checkpoint_path),
        "src_fps": float(src_fps),
        "target_fps": float(args.target_fps),
        "window_T": int(window_T),
        "window_stride": int(window_stride),
        "segment_threshold": float(thr),
        "segment_min_dur": float(min_dur),
        "segment_merge_gap": float(auto_merge_gap) if merge_gap is None else float(merge_gap),
        "use_uncertainty": bool(alpha_beta is not None),
    }
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    def _ensure_numpy(x):
        if x is None:
            return None
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        return np.array(x)

    def _save_heatmap_overlay(rgb, attn_0to1, out_png, out_npy):
        H, W = rgb.shape[:2]
        hm = cv2.resize(attn_0to1.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        heat = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = (0.6 * heat + 0.4 * rgb).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(out_png), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        np.save(str(out_npy), hm)

    if int(args.save_xai) == 1 and xai_maps is not None:
        heat_dir = run_dir / "heatmaps"
        heat_dir.mkdir(exist_ok=True, parents=True)
        x_stride = max(1, int(args.xai_stride))

        # Accumulated per-frame attention: mean when a frame appears in multiple windows
        sum_attn = [None] * N
        cnt_attn = np.zeros(N, dtype=np.int32)

        # Re-run a **light** window pass to collect attention only (no probs accumulation)
        # reusing the same batching & windows to align shapes.
        with torch.no_grad(), autocast_ctx():
            for bi, b in enumerate(range(0, len(window_starts), B_win)):
                batch_starts = window_starts[b:b + B_win]
                batch_clips = []
                for s in batch_starts:
                    s = min(s, N - window_T)
                    clip_imgs = []
                    for j in range(s, s + window_T):
                        img = torch.from_numpy(frames_uint8[j]).permute(2, 0, 1).to(torch.float32) / 255.0
                        img = normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        clip_imgs.append(img)
                    batch_clips.append(torch.stack(clip_imgs, dim=1).unsqueeze(0))
                batch_clips = torch.cat(batch_clips, dim=0).to(device)

                B = batch_clips.size(0)
                rep_ids = text_inputs.input_ids.repeat(B, 1)
                rep_mask = text_inputs.attention_mask.repeat(B, 1)

                outs = model(batch_clips, rep_ids, rep_mask)
                # Try to pull attention again
                attn = None
                if isinstance(outs, (list, tuple)) and len(outs) >= 4 and outs[3] is not None:
                    attn = outs[3]  # [B, T, H', W'] or list

                if attn is None:
                    continue

                attn = _ensure_numpy(attn)
                # If attn is [B, T, H', W'], process directly; else if list of length T, stack
                if isinstance(attn, list):
                    attn = np.stack([_ensure_numpy(a) for a in attn], axis=1)  # try [B, T, H', W']

                # Reduce over heads if present: accept [..., Heads, H', W'] or similar
                # Heuristic: take the last 2 dims as spatial; average all preceding non-B,T dims after B,T.
                if attn.ndim >= 4:
                    # attn shape guess: [B, T, ..., H', W']
                    Hp, Wp = attn.shape[-2], attn.shape[-1]
                    spatial = attn.reshape(attn.shape[0], attn.shape[1], -1, Hp, Wp).mean(
                        axis=2)  # mean over heads/chan
                else:
                    # unexpected, skip
                    continue

                # Normalize each frame map to [0,1] before accumulation
                spatial = spatial - spatial.min(axis=(-2, -1), keepdims=True)
                denom = np.maximum(spatial.max(axis=(-2, -1), keepdims=True), 1e-6)
                spatial = spatial / denom

                # Accumulate into per-sampled-frame buckets
                for k, s in enumerate(batch_starts):
                    s = min(s, N - window_T)
                    for j in range(window_T):
                        t = s + j
                        if t >= N: break
                        m = spatial[k, j]  # (H', W')
                        if sum_attn[t] is None:
                            sum_attn[t] = m.copy()
                        else:
                            sum_attn[t] += m
                        cnt_attn[t] += 1

        # Save overlays every x_stride frames
        saved = 0
        for i in range(0, N, x_stride):
            if sum_attn[i] is None or cnt_attn[i] == 0:
                continue
            hm = sum_attn[i] / float(cnt_attn[i])
            out_png = heat_dir / f"frame_{int(sampled_idx[i]):07d}.png"
            out_npy = heat_dir / f"frame_{int(sampled_idx[i]):07d}.npy"
            _save_heatmap_overlay(frames_uint8[i], hm, out_png, out_npy)
            saved += 1
        print(f"[XAI] saved {saved} heatmap overlays to {heat_dir}")

    def _parse_spans(span_list):
        spans = []
        for s in (span_list or []):
            s = s.strip().replace(" ", "")
            if not s:
                continue
            if ";" in s:  # allow "a,b;c,d"
                parts = s.split(";")
            else:
                parts = [s]
            for p in parts:
                a, b = p.split(",")
                spans.append((float(a), float(b)))
        return spans

    def _labels_from_spans(times, spans):
        y = np.zeros_like(times, dtype=np.int32)
        for (a, b) in spans:
            y |= ((times >= float(a)) & (times < float(b))).astype(np.int32)
        return y

    def _basic_metrics(y_true, y_pred):
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        acc = (tp + tn) / max(len(y_true), 1)
        prec = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
        rec = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / max(prec + rec, 1e-12)) if (prec + rec) > 0 else 0.0
        return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    def _try_sklearn_ap_auc(y_true, y_prob):
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
            ap = float(average_precision_score(y_true, y_prob))
            # roc_auc_score needs both classes present; guard:
            if len({0, 1} & set(map(int, set(y_true.tolist())))) == 2:
                auc = float(roc_auc_score(y_true, y_prob))
            else:
                auc = float("nan")
            return ap, auc
        except Exception:
            return float("nan"), float("nan")

    # Only evaluate if GT spans are provided
    if args.gt_span:
        spans = _parse_spans(args.gt_span)
        y_true = _labels_from_spans(times_sec, spans)
        y_prob = probabilities.astype(np.float32)

        # Sweep thresholds if not provided
        if args.eval_threshold is None:
            thr_grid = np.linspace(0.05, 0.95, 19, dtype=np.float32)
            rows = []
            best = {"f1": -1, "thr": None, "metrics": None}
            for thr in thr_grid:
                y_pred = (y_prob >= thr).astype(np.int32)
                m = _basic_metrics(y_true, y_pred)
                rows.append([float(thr), m["accuracy"], m["precision"], m["recall"], m["f1"]])
                if m["f1"] > best["f1"] or (
                        m["f1"] == best["f1"] and m["recall"] > (best["metrics"]["recall"] if best["metrics"] else -1)):
                    best = {"f1": m["f1"], "thr": float(thr), "metrics": m}
            eval_thr = float(best["thr"])
            best_metrics = best["metrics"]
        else:
            eval_thr = float(args.eval_threshold)
            y_pred = (y_prob >= eval_thr).astype(np.int32)
            best_metrics = _basic_metrics(y_true, y_pred)
            thr_grid = None
            rows = None

        # AP / ROC-AUC (sklearn package should be available)
        ap, auc = _try_sklearn_ap_auc(y_true, y_prob)

        # Save per-frame eval CSV
        if int(args.save_eval) == 1:
            pf_csv = run_dir / f"per_frame_eval{sfx}.csv"
            with open(pf_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["frame_idx", "time_sec", "prob", "gt", f"pred@{eval_thr:.2f}"])
                y_pred_eval = (y_prob >= eval_thr).astype(np.int32)
                for i in range(len(y_prob)):
                    w.writerow([int(sampled_idx[i]), float(times_sec[i]), float(y_prob[i]),
                                int(y_true[i]), int(y_pred_eval[i])])

            # Save threshold sweep (if we did it)
            if rows is not None:
                sweep_csv = run_dir / f"metrics_over_thresholds{sfx}.csv"
                with open(sweep_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["threshold", "accuracy", "precision", "recall", "f1"])
                    w.writerows(rows)

        # Print summary
        print("\n--- Per-frame metrics ---")
        print(f"GT spans: {spans}")
        print(f"Chosen eval threshold: {eval_thr:.2f} "
              f"(F1={best_metrics['f1']:.4f}, P={best_metrics['precision']:.4f}, "
              f"R={best_metrics['recall']:.4f}, Acc={best_metrics['accuracy']:.4f})")
        if not np.isnan(ap):
            print(f"Average Precision (AP): {ap:.4f}")
        else:
            print("Average Precision (AP): N/A (sklearn not available or only one class present)")
        if not np.isnan(ap):  # reuse check
            print(f"ROC AUC: {auc:.4f}" if not np.isnan(auc) else "ROC AUC: N/A (single class in GT)")

    print("\n--- Results ---")
    print(f"Query: '{args.text_query}'")
    print(f"Saved per-frame scores to: {scores_csv}")
    print(f"Saved predicted segments to: {seg_csv}")
    if len(segments) == 0:
        print("No segments found with current threshold. Consider lowering --segment_threshold.")

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Run inference for language-guided video localization (single video).")
    ap.add_argument("--video_path", required=True, type=str, help="Path to the input video file.")
    ap.add_argument("--text_query", required=True, type=str, help="Natural-language query.")
    ap.add_argument("--checkpoint_path", required=True, type=str, help="Path to best_model.pth")
    ap.add_argument("--segment_threshold", type=float, default=0.5)
    ap.add_argument("--segment_min_dur", type=float, default=0.4)
    ap.add_argument("--gt_span", type=str, action="append", default=None,
                    help="One or more GT intervals 'start,end' in seconds (repeat to add multiple).")
    ap.add_argument("--suffix", type=str, default="",
                    help="Optional suffix to append to output filenames. Example: 'improved'")
    ap.add_argument("--eval_threshold", type=float, default=None,
                    help="Threshold for per-frame metrics. If omitted, we auto-sweep to best F1.")
    ap.add_argument("--save_eval", type=int, default=1,
                    help="If 1, saves per-frame eval CSV and threshold-sweep metrics.")
    ap.add_argument("--segment_merge_gap", type=float, default=None)
    ap.add_argument("--use_dataparallel", type=int, default=0,
                    help="Set 1 to use torch.nn.DataParallel over all visible GPUs.")
    ap.add_argument("--batch_size_windows", type=int, default=8,
                    help="How many windows to score at once.")
    ap.add_argument("--window_T", type=int, default=None,
                    help="Temporal length per window (defaults to cfg.DATA.NUM_FRAMES).")
    ap.add_argument("--window_stride", type=int, default=None,
                    help="Stride in frames between windows (defaults to T//2).")
    ap.add_argument("--target_fps", type=float, default=0.5, help="Sampling rate for inference (frames per second).")
    ap.add_argument("--save_uncertainty", type=int, default=1,
                    help="If 1 and model returns evidential params (alpha,beta), save uncertainty.")
    ap.add_argument("--save_xai", type=int, default=1,
                    help="If 1 and model returns attention maps, save heatmap overlays + raw maps.")
    ap.add_argument("--xai_stride", type=int, default=10,
                    help="Save a heatmap every N sampled frames to keep storage low.")
    ap.add_argument("--experiment_tag", type=str, default="exp",
                    help="Used to label the run directory/manifest.")

    args = ap.parse_args()
    run_inference(args)
