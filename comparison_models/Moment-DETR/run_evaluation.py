"""
Moment-DETR Baseline: Step 4 - Evaluation
"""
import os, argparse, torch, json
from torch.utils.data import DataLoader, SequentialSampler
from moment_detr_module.configs import Config
from moment_detr_module.modeling import MomentDETR
from moment_detr_module.dataset import MomentDETRDataset, collate_fn
from moment_detr_module.engine import evaluate

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); cfg = Config()
    model = MomentDETR(cfg).to(device)
    if not args.resume: raise ValueError("A checkpoint path must be provided with --resume")
    print(f"Loading checkpoint from: {args.resume}"); checkpoint = torch.load(args.resume, map_location=device); model.load_state_dict(checkpoint['model']); model.eval()
    test_dataset = MomentDETRDataset(cfg, split='test')
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn, sampler=SequentialSampler(test_dataset))
    print("Starting evaluation on the test set...")
    eval_stats = evaluate(model, test_loader, device)
    output_dir = os.path.dirname(args.resume)
    summary = f"\n{'='*40}\n          Moment-DETR Evaluation Results\n{'='*40}\n"
    for k, v in eval_stats.items(): summary += f"{k:<15}: {v:.4f}\n" if isinstance(v, float) else ""
    summary += "="*40
    print(summary)
    with open(os.path.join(output_dir, "test_metrics.txt"), "w") as f: f.write(summary)
    with open(os.path.join(output_dir, "test_metrics.json"), "w") as f: json.dump(eval_stats, f, indent=4)
    print(f"\n[SUCCESS] Evaluation complete. Results saved in {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, required=True, help="Path to the trained model checkpoint (.ckpt)"); parser.add_argument("--num_workers", type=int, default=8); args = parser.parse_args(); main(args)

