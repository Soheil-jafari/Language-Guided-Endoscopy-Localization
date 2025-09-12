import os
import json
import argparse
import datetime
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from moment_detr_module.configs import Config
from moment_detr_module.modeling import MomentDETR
from moment_detr_module.dataset import MomentDETRDataset, collate_fn
from moment_detr_module.engine import train_one_epoch, evaluate
from moment_detr_module.utils import setup_seed, get_logger
from tqdm import tqdm


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=0)

    # === DEBUG OVERFIT FLAGS ===
    parser.add_argument("--debug_overfit", action="store_true",
                        help="Train on a tiny subset and verify the model can overfit.")
    parser.add_argument("--debug_k", type=int, default=8,
                        help="How many samples to overfit on.")
    parser.add_argument("--debug_epochs", type=int, default=200,
                        help="Epochs to run during overfit debug.")
    parser.add_argument("--debug_lr", type=float, default=1e-3,
                        help="LR during overfit debug.")
    return parser


def main(args):
    # ----- distributed init -----
    if args.dist:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)

    device = torch.device("cuda", args.local_rank)
    cfg = Config()
    setup_seed(cfg.seed)

    # If we are in debug-overfit mode, shorten epochs
    if args.debug_overfit:
        cfg.epochs = args.debug_epochs

    run_name = f"moment_detr_{cfg.dataset_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = os.path.join(args.ckpt_dir, run_name)
    if args.local_rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Logger
    logger = get_logger(os.path.join(checkpoint_dir, "train.log")) if args.local_rank == 0 else None
    if args.local_rank == 0 and logger is not None:
        logger.info(f"Saving checkpoints to: {checkpoint_dir}")
        logger.info(f"Config: {vars(cfg)}")
        # Also dump config to JSON for record
        with open(os.path.join(checkpoint_dir, "config_dump.json"), "w") as f:
            json.dump(vars(cfg), f, indent=2)

    # ----- model -----
    model = MomentDETR(cfg).to(device)
    if args.dist:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    # Optimizer + LR
    params = [p for p in model.parameters() if p.requires_grad]
    assert len(params) > 0, "No trainable parameters found!"
    base_lr = getattr(cfg, "lr", 1e-4)
    lr_use = (args.debug_lr if args.debug_overfit else base_lr)
    optimizer = torch.optim.AdamW(params, lr=lr_use, weight_decay=getattr(cfg, "weight_decay", 1e-4))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, getattr(cfg, "lr_drop", 40))

    # ----- datasets -----
    train_dataset = MomentDETRDataset(cfg, 'train')
    val_dataset = MomentDETRDataset(cfg, 'val')

    # Optional: overfit on K items
    if args.debug_overfit:
        from torch.utils.data import Subset, Dataset
        idxs = list(range(min(args.debug_k, len(train_dataset))))
        train_dataset = Subset(train_dataset, idxs)

        class _RepeatDS(Dataset):
            def __init__(self, base, times=128): self.base, self.times = base, times
            def __len__(self): return len(self.base) * self.times
            def __getitem__(self, i): return self.base[i % len(self.base)]
        train_dataset = _RepeatDS(train_dataset, times=128)

    # Samplers
    train_sampler = DistributedSampler(train_dataset) if (args.dist and not args.debug_overfit) else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if args.dist else None

    # ----- loaders (use collate_fn for both) -----
    if args.debug_overfit:
        train_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=False, collate_fn=collate_fn
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=getattr(cfg, "batch_size", 32),
            shuffle=(train_sampler is None), num_workers=args.num_workers,
            pin_memory=True, drop_last=False, sampler=train_sampler,
            collate_fn=collate_fn
        )

    val_loader = DataLoader(
        val_dataset, batch_size=getattr(cfg, "batch_size", 32), shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False, sampler=val_sampler,
        collate_fn=collate_fn
    )

    # Optional single-batch sanity (only when debug_overfit true)
    one_batch_sanity = args.debug_overfit and (not args.dist)
    if one_batch_sanity:
        if args.local_rank == 0:
            print("\n" + "=" * 40)
            print("!!! RUNNING OVERFITTING SANITY CHECK (single GPU, one batch) !!!")
            print("=" * 40 + "\n")
        single_batch = next(iter(train_loader))
        train_loader = [single_batch]
        val_loader = [single_batch]

    if args.local_rank == 0 and logger is not None:
        logger.info("Start training")

    # Track best by R1@0.5 (fall back if missing)
    def _score(d): return d.get('R1@0.5', d.get('mAP@0.5', -1))

    best_metric = -1.0
    best_epoch = -1

    for epoch in tqdm(range(cfg.epochs), desc="Training Epochs"):
        if args.dist and (train_sampler is not None):
            train_sampler.set_epoch(epoch)

        train_one_epoch(model, train_loader, optimizer, device, epoch, cfg.clip_max_norm, logger, args.local_rank)
        lr_scheduler.step()

        if args.local_rank == 0:
            # DDP unwrap for saving
            state_dict = model.module.state_dict() if args.dist else model.state_dict()

            # Save epoch checkpoint
            torch.save(
                {
                    'model': state_dict,
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch
                },
                os.path.join(checkpoint_dir, f"epoch_{epoch}.ckpt")
            )

            # Evaluate (and track best)
            if (epoch + 1) % args.eval_every == 0 or epoch == cfg.epochs - 1:
                eval_stats = evaluate(model, val_loader, device)
                if logger is not None:
                    logger.info(f"Validation - Epoch {epoch}: {eval_stats}")

                # Save per-epoch metrics JSON
                with open(os.path.join(checkpoint_dir, f"metrics_epoch_{epoch}.json"), "w") as f:
                    json.dump({"epoch": epoch, **eval_stats}, f, indent=2)

                # Update "best"
                current_metric = _score(eval_stats)
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_epoch = epoch
                    # Save best weights (model only)
                    torch.save({'model': state_dict}, os.path.join(checkpoint_dir, "best_checkpoint.ckpt"))
                    # Save best metrics JSON
                    with open(os.path.join(checkpoint_dir, "best_metrics.json"), "w") as f:
                        json.dump({"epoch": epoch, **eval_stats}, f, indent=2)

    if args.local_rank == 0 and logger is not None:
        logger.info(f"Training finished. Best epoch: {best_epoch}, best metric: {best_metric:.4f}")


if __name__ == '__main__':
    parser = build_argparser()
    args = parser.parse_args()

    # Determinism for debug-overfit
    if args.debug_overfit:
        import random, numpy as np
        torch.manual_seed(0); np.random.seed(0); random.seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Decide distributed after parsing
    args.dist = "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1
    main(args)
