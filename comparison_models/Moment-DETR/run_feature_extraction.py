"""
Moment-DETR Baseline: Step 2 - Visual Feature Extraction (Server-Ready)
"""
import torch, torch.nn as nn, numpy as np
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from tqdm import tqdm

class ServerConfig:
    EXTRACTED_FRAMES_DIR = "/home/240331715/data/unified_medical_videos/extracted_frames"
    NUM_WORKERS = 4
cfg = ServerConfig()
FRAMES_DIR, FEATURES_DIR = Path(cfg.EXTRACTED_FRAMES_DIR), Path("./extracted_features_resnet50")
BATCH_SIZE, DEVICE = 128, "cuda" if torch.cuda.is_available() else "cpu"

class ResNet50FeatureExtractor(nn.Module):
    def __init__(self): super().__init__(); resnet = models.resnet50(pretrained=True); self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    def forward(self, x): return self.feature_extractor(x)
class FrameDataset(Dataset):
    def __init__(self, p, t): self.paths, self.transform = p, t
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        try: return self.transform(Image.open(self.paths[i]).convert("RGB"))
        except: return torch.zeros((3, 224, 224))

def main():
    print("[INFO] Starting visual feature extraction for Moment-DETR.")
    FEATURES_DIR.mkdir(exist_ok=True); device = torch.device(DEVICE)
    model = ResNet50FeatureExtractor().to(device); model.eval()
    if torch.cuda.device_count() > 1: print(f"[INFO] Using {torch.cuda.device_count()} GPUs."); model = nn.DataParallel(model)
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    video_dirs = sorted([d for d in FRAMES_DIR.iterdir() if d.is_dir()])
    print(f"[INFO] Found {len(video_dirs)} videos to process in {FRAMES_DIR}")
    for video_dir in tqdm(video_dirs, desc="Extracting Features"):
        output_path = FEATURES_DIR / f"{video_dir.name}.npz"
        if output_path.exists(): continue
        frame_paths = sorted(video_dir.glob("*.jpg"))
        if not frame_paths: continue
        dataloader = DataLoader(FrameDataset(frame_paths, transform), batch_size=BATCH_SIZE, num_workers=cfg.NUM_WORKERS, pin_memory=True)
        with torch.no_grad(): features = torch.cat([model(frames.to(device)) for frames in dataloader]).squeeze().cpu().numpy()
        np.savez_compressed(output_path, features=features, num_frames=len(frame_paths))
    print("\n[SUCCESS] Feature extraction complete.")

if __name__ == "__main__": main()

