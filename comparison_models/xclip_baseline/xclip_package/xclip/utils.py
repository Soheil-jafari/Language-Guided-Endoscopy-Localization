\
import os, sys, json, time, random, math, hashlib
from pathlib import Path
from dataclasses import dataclass
import torch
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def now_str():
    return time.strftime("%Y%m%d_%H%M%S")

def sha1_short(s: str):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def load_project_config():
    """
    Import the user's project_config.py that defines Config().
    We assume `project_config.py` is either in PYTHONPATH or alongside the scripts when run.
    """
    try:
        import project_config as pc
    except Exception as e:
        raise RuntimeError(
            "Could not import project_config.py. Place your uploaded file next to these scripts or add its directory to PYTHONPATH."
        ) from e
    return pc.Config()

@dataclass
class RunPaths:
    out_dir: Path
    logs_dir: Path
    per_video_dir: Path

    @classmethod
    def from_config(cls, config, tag="xclip"):
        base = Path(config.OUTPUT_DIR) / tag / now_str()
        return cls(out_dir=base, logs_dir=base / "logs", per_video_dir=base / "per_video")
