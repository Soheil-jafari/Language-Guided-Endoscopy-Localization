"""Real-size GPU rehearsal of the two new trained comparison models (paper preset; about a minute).

Before any long job depends on them, run two optimiser steps of one model at the size the paper protocol trains
it (random weights and random inputs; nothing is saved), with the same cuDNN settings as their training:
  --part closedset   the closed-set model: our TimeSformer backbone and temporal head, batch 8 x 16 frames at
                     the configured crop, in the configured mixed precision;
  --part tecno_cnn   TeCNO stage 1: the two-head ResNet-50 at batch 80, float32 (as the authors);
  --part tecno_tcn   TeCNO stage 2: the causal MS-TCN on a 6,000-step sequence (a long Cholec80 video at 1 fps).
main.py runs each part in its own process (so no other model is in GPU memory while one is measured) with a
deadline, once per GPU type, and only reports a failure; the run it concerns would then fail on its own and be
retried by the next job. Writes {part: {ok, peak_gb, reserved_gb, seconds, what}} to --output.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'comparison_models' / 'closedset'))
sys.path.insert(0, str(REPO / 'comparison_models' / 'tecno'))
PARTS = ('closedset', 'tecno_cnn', 'tecno_tcn')


def closedset_steps(a, device):
    import closedset
    from project_config import config as cfg
    closedset.apply_overrides(cfg, json.loads(Path(a.config).read_text()))
    cfg.TRAIN.DEVICE = device
    amp = torch.bfloat16 if cfg.TRAIN.AMP_DTYPE == 'bf16' else torch.float16
    model = closedset.ClosedSetModel(cfg, initialize_backbone=False).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda' and amp == torch.float16)
    size, length = cfg.DATA.TRAIN_CROP_SIZE, cfg.DATA.CLIP_LENGTH
    for _ in range(2):
        clip = torch.randn(a.closedset_batch, 3, length, size, size, device=device)
        labels = torch.randint(0, 2, (a.closedset_batch, length, closedset.N_CONCEPTS), device=device).float()
        with torch.autocast(device, dtype=amp, enabled=device == 'cuda'):
            raw, refined = model(clip)
        loss, _ = closedset.loss_terms(raw, refined, labels, torch.tensor(1.0, device=device),
                                       cfg.TRAIN.TEMPORAL_LOSS_WEIGHT)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    return f'closed-set model, batch {a.closedset_batch} x {length} frames, {size} px, {cfg.TRAIN.AMP_DTYPE}'


def tecno_cnn_steps(a, device):
    import tecno
    model = tecno.TwoHeadResNet50(pretrained=False).to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    ce = torch.nn.CrossEntropyLoss(weight=torch.tensor(tecno.CLASS_WEIGHTS, device=device))
    for _ in range(2):
        x = torch.randn(a.tecno_batch, 3, a.tecno_size, a.tecno_size, device=device)
        _, phase, tool = model(x)
        loss = tecno.cnn_loss(model, phase, tool, torch.randint(0, 7, (a.tecno_batch,), device=device),
                              torch.randint(0, 2, (a.tecno_batch, 7), device=device).float(), ce)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return f'TeCNO stage 1 (ResNet-50), batch {a.tecno_batch}, {a.tecno_size} px, float32'


def tecno_tcn_steps(a, device):
    import tecno
    model = tecno.MultiStageTCN(**tecno.TCN).to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=7e-5)
    ce = torch.nn.CrossEntropyLoss(weight=torch.tensor(tecno.CLASS_WEIGHTS, device=device))
    for _ in range(2):
        out = model(torch.randn(1, 2048, a.tcn_steps, device=device))
        loss = tecno.tcn_loss(out, torch.randint(0, 7, (a.tcn_steps,), device=device), ce)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return f'TeCNO stage 2 (MS-TCN), {a.tcn_steps} steps'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--part', choices=PARTS, required=True)
    p.add_argument('--config', required=True, help='overrides of project_config (crop, clip, AMP) as JSON')
    p.add_argument('--tecno-batch', type=int, default=80)
    p.add_argument('--tecno-size', type=int, default=224)
    p.add_argument('--closedset-batch', type=int, default=8)
    p.add_argument('--tcn-steps', type=int, default=6000)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda', help='(cpu: for the tests only)')
    a = p.parse_args()
    device = a.device
    if device == 'cuda' and not torch.cuda.is_available():
        print('comparison rehearsal SKIPPED: no CUDA GPU', flush=True)
        return
    torch.backends.cudnn.benchmark = False                  # as both models' training
    torch.backends.cudnn.deterministic = True
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    try:
        what = dict(closedset=closedset_steps, tecno_cnn=tecno_cnn_steps, tecno_tcn=tecno_tcn_steps)[a.part](a, device)
        result = dict(ok=True, what=what, seconds=round(time.time() - started, 1))
        if device == 'cuda':
            torch.cuda.synchronize()
            result.update(peak_gb=round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
                          reserved_gb=round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2),
                          gpu_memory_gb=round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1))
    except Exception as e:                                       # noqa: BLE001
        result = dict(ok=False, error=repr(e)[:500])
    print(f'{a.part}: {result}', flush=True)
    Path(a.output).write_text(json.dumps({a.part: result}, indent=2))
    if not result['ok']:
        sys.exit(1)


if __name__ == '__main__':
    main()
