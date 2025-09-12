import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XCLIPProcessor
import pandas as pd
from tqdm import tqdm
import random
from xclip.model import XCLIPWrapper
from xclip.losses import clip_style_infonce
from xclip.utils import set_seed, ensure_dir, load_project_config, RunPaths
from xclip.data import load_triplets, get_frame_paths, sample_positive_window, load_frames_by_indices


class PosPairDataset(Dataset):
    def __init__(self, config, csv_path: str, num_frames: int, stride: int, frame_glob: str = "*.jpg"):
        self.config = config
        self.triplets = load_triplets(csv_path, default_fps=config.DATA.FRAME_RATE)
        self.num_frames = num_frames
        self.stride = stride
        self.frame_glob = frame_glob
        self.video_cache = {}  # video_id -> frame_paths

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        tri = self.triplets[idx]
        if tri.video not in self.video_cache:
            self.video_cache[tri.video] = get_frame_paths(self.config.EXTRACTED_FRAMES_DIR, tri.video, self.frame_glob)
        frame_paths = self.video_cache[tri.video]
        _, indices = sample_positive_window(tri, self.num_frames, self.stride)
        frames = load_frames_by_indices(frame_paths, indices)
        return {"frames": frames, "text": tri.query}


def collate(batch):
    frames_list = [b["frames"] for b in batch]
    texts = [b["text"] for b in batch]
    return frames_list, texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--frame-glob", type=str, default="*.jpg")
    parser.add_argument("--model-name", type=str, default="microsoft/xclip-base-patch32")
    parser.add_argument("--tag", type=str, default="xclip_ft")
    parser.add_argument("--train-csv-path", type=str, default="", help="Override the training CSV path")
    parser.add_argument("--val-csv-path", type=str, default="", help="Override the validation CSV path")
    parser.add_argument("--subset-ratio", type=float, default=1.0, help="Fraction of the training data to use")
    # --- NEW: Argument to specify a checkpoint to resume from ---
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_project_config()
    paths = RunPaths.from_config(config, tag=args.tag)
    ensure_dir(paths.out_dir);
    ensure_dir(paths.logs_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XCLIPWrapper(model_name=args.model_name).to(device)
    processor = model.processor
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    start_epoch = 1
    best_val = 1e9
    ckpt_path = Path(config.CHECKPOINT_DIR) / f"{args.tag}_xclip.pt"

    if args.resume_from:
        if Path(args.resume_from).exists():
            print(f"🔄 Resuming training from checkpoint: {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location=device)
            model.load_state_dict(checkpoint['model'])
            # The optimizer state is loaded if it exists in the checkpoint
            if 'optimizer' in checkpoint:
                optim.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            best_val = checkpoint['val_loss']
            print(f"   -> Resuming from Epoch {start_epoch}, Best Val Loss: {best_val:.4f}")
        else:
            print(f"⚠️  Warning: Checkpoint not found at {args.resume_from}. Starting from scratch.")
    # --- End of checkpoint loading logic ---

    train_csv_path = args.train_csv_path if args.train_csv_path else config.TRAIN_TRIPLETS_CSV_PATH
    val_csv_path = args.val_csv_path if args.val_csv_path else config.VAL_TRIPLETS_CSV_PATH

    print(f"[INFO] Loading training data from: {train_csv_path}")
    train_ds = PosPairDataset(config, train_csv_path, args.num_frames, args.stride, args.frame_glob)

    if args.subset_ratio < 1.0:
        print(f"[INFO] Using a random {args.subset_ratio * 100:.0f}% subset of the training data.")
        num_train_samples = len(train_ds)
        subset_size = int(num_train_samples * args.subset_ratio)
        indices = list(range(num_train_samples))
        random.shuffle(indices)
        train_ds = torch.utils.data.Subset(train_ds, indices[:subset_size])
        print(f"[INFO] Subset size: {len(train_ds)} samples.")

    print(f"[INFO] Loading validation data from: {val_csv_path}")
    val_ds = PosPairDataset(config, val_csv_path, args.num_frames, args.stride, args.frame_glob)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=config.DATA.NUM_WORKERS,
                              collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=config.DATA.NUM_WORKERS,
                            collate_fn=collate)

    from torch.cuda.amp import GradScaler, autocast
    scaler = GradScaler()

    def run_epoch(loader, epoch, train=True):
        # (run_epoch function remains the same)
        total_loss = 0.0
        mode = "Train" if train else "Val"
        pbar = tqdm(loader, desc=f"Epoch {epoch} [{mode}]")
        for frames_list, texts in pbar:
            inputs = processor(videos=frames_list, text=texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with autocast():
                video_embeds, text_embeds = model(**inputs)
                loss = clip_style_infonce(video_embeds, text_embeds, temperature=1.0)
            if train:
                optim.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{total_loss / (pbar.n + 1):.4f}")
        if not loader: return 0.0
        return total_loss / len(loader)

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    epoch_pbar = tqdm(range(start_epoch, args.epochs + 1), desc="Overall Training Progress")
    for epoch in epoch_pbar:
        model.train()
        train_loss = run_epoch(train_loader, epoch, train=True)
        model.eval()
        with torch.no_grad():
            val_loss = run_epoch(val_loader, epoch, train=False)
        epoch_pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "model_name": args.model_name
            }, ckpt_path)
            print(f"🎉 New best model saved to {ckpt_path} with val_loss: {best_val:.4f}")

if __name__ == "__main__":
    main()