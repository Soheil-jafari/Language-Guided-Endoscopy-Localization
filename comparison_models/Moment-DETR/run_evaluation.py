"""
Moment-DETR Baseline: Step 4 - Evaluation
"""
import os, argparse, torch, json
from torch.utils.data import DataLoader
from moment_detr_module.configs import Config
from moment_detr_module.modeling import MomentDETR
from moment_detr_module.dataset import MomentDETRDataset, collate_fn
from moment_detr_module.engine import evaluate

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()

    # --- build model ---
    model = MomentDETR(cfg).to(device)

    # --- load checkpoint robustly ---
    if not args.resume:
        raise ValueError("A checkpoint path must be provided with --resume")
    print(f"Loading checkpoint from: {args.resume}")
    state = torch.load(args.resume, map_location="cpu")
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # --- dataset/loader ---
    split = args.split  # "test" by default
    dataset = MomentDETRDataset(cfg, split=split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size if args.batch_size > 0 else getattr(cfg, "batch_size", 32),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_fn,
    )

    # --- evaluate ---
    print(f"Starting evaluation on the {split} split...")
    stats = evaluate(model, loader, device)

    # --- save outputs next to the checkpoint ---
    out_dir = os.path.dirname(os.path.abspath(args.resume))
    txt_path = os.path.join(out_dir, f"{split}_metrics.txt")
    json_path = os.path.join(out_dir, f"{split}_metrics.json")

    lines = [ "\n" + "="*40, f"   Moment-DETR Evaluation Results ({split})", "="*40 ]
    for k, v in sorted(stats.items()):
        if isinstance(v, (float, int)):
            lines.append(f"{k:<18}: {float(v):.6f}")
    lines.append("="*40)
    report = "\n".join(lines)

    print(report)
    with open(txt_path, "w") as f: f.write(report)
    with open(json_path, "w") as f: json.dump(stats, f, indent=2)
    print(f"\n[SUCCESS] Saved: {txt_path} and {json_path}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=str, required=True,
                    help="Path to .ckpt (best_checkpoint.ckpt or epoch_X.ckpt)")
    ap.add_argument("--split", type=str, default="test", choices=["val","test"],
                    help="Which split to evaluate")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=0,
                    help="Override batch size (0 = use cfg.batch_size)")
    args = ap.parse_args()
    main(args)
