\
import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XCLIPProcessor
import pandas as pd
from xclip.model import XCLIPWrapper
from xclip.losses import clip_style_infonce
from xclip.utils import set_seed, ensure_dir, load_project_config, RunPaths

from xclip.data import load_triplets, get_frame_paths, sample_positive_window, load_frames_by_indices

class PosPairDataset(Dataset):
    def __init__(self, config, csv_path: str, num_frames: int, stride: int, frame_glob: str="*.jpg"):
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--frame-glob", type=str, default="*.jpg")
    parser.add_argument("--model-name", type=str, default="microsoft/xclip-base-patch32")
    parser.add_argument("--tag", type=str, default="xclip_ft")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_project_config()
    paths = RunPaths.from_config(config, tag=args.tag)
    ensure_dir(paths.out_dir); ensure_dir(paths.logs_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XCLIPWrapper(model_name=args.model_name).to(device)
    model.train()

    processor = model.processor

    train_ds = PosPairDataset(config, config.TRAIN_TRIPLETS_CSV_PATH, args.num_frames, args.stride, args.frame_glob)
    val_ds   = PosPairDataset(config, config.VAL_TRIPLETS_CSV_PATH, args.num_frames, args.stride, args.frame_glob)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=config.DATA.NUM_WORKERS, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=config.DATA.NUM_WORKERS, collate_fn=collate)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def run_epoch(loader, train=True):
        total = 0.0
        steps = 0
        for frames_list, texts in loader:
            inputs = processor(videos=frames_list, text=texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            v, t = model(**inputs)
            loss = clip_style_infonce(v, t, temperature=1.0)
            if train:
                optim.zero_grad(); loss.backward(); optim.step()
            total += loss.item(); steps += 1
        return total / max(steps, 1)

    best_val = 1e9
    ckpt_path = Path(config.CHECKPOINT_DIR) / f"{args.tag}_xclip.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        tr = run_epoch(train_loader, train=True)
        vl = run_epoch(val_loader, train=False)
        print(f"[Epoch {epoch}] train {tr:.4f} | val {vl:.4f}")
        # save best
        if vl < best_val:
            best_val = vl
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": vl, "model_name": args.model_name},
                       ckpt_path)
            print(f"Saved best to {ckpt_path}")

if __name__ == "__main__":
    main()
