"""Real-size GPU rehearsal of the two new trained comparison models (paper preset; about a minute).

Before any long job depends on them, run two optimiser steps of each at the size the paper protocol trains it
(random weights and random inputs; nothing is saved):
  * the closed-set model: our TimeSformer backbone and temporal head, batch 8 x 16 frames at the configured crop,
    in the configured mixed precision;
  * TeCNO: the two-head ResNet-50 at batch 80 (float32, as the authors), and the causal MS-TCN on a 6,000-step
    sequence (a long Cholec80 video at 1 fps).
Prints the peak GPU memory of each and writes them to --output (JSON). main.py runs it once per GPU type and
only reports a failure; the run it concerns would then fail on its own and be retried by the next job.
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


def measure(name, step, results, device):
    cuda = device == 'cuda'
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    try:
        for _ in range(2):
            step()
        results[name] = dict(ok=True, seconds=round(time.time() - started, 1))
        if cuda:
            torch.cuda.synchronize()
            results[name].update(peak_gb=round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
                                 reserved_gb=round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2))
    except Exception as e:                                       # noqa: BLE001
        results[name] = dict(ok=False, error=repr(e)[:500])
    print(f'{name}: {results[name]}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True, help="our training configuration (main.train_config) as JSON")
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
    results = dict(gpu=torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu',
                   gpu_memory_gb=round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
                   if device == 'cuda' else None)
    import closedset
    import tecno
    from project_config import config as cfg
    closedset.apply_overrides(cfg, json.loads(Path(a.config).read_text()))
    cfg.TRAIN.DEVICE = device
    amp = torch.bfloat16 if cfg.TRAIN.AMP_DTYPE == 'bf16' else torch.float16

    def closed_step(state={}):
        if not state:
            state['model'] = closedset.ClosedSetModel(cfg, initialize_backbone=False).to(device).train()
            state['opt'] = torch.optim.AdamW(state['model'].parameters(), lr=1e-5)
            state['scaler'] = torch.amp.GradScaler('cuda', enabled=device == 'cuda' and amp == torch.float16)
        m, opt, scaler = state['model'], state['opt'], state['scaler']
        size, length = cfg.DATA.TRAIN_CROP_SIZE, cfg.DATA.CLIP_LENGTH
        clip = torch.randn(a.closedset_batch, 3, length, size, size, device=device)
        labels = torch.randint(0, 2, (a.closedset_batch, length, closedset.N_CONCEPTS), device=device).float()
        with torch.autocast(device, dtype=amp, enabled=device == 'cuda'):
            raw, refined = m(clip)
        loss, _ = closedset.loss_terms(raw, refined, labels, torch.tensor(1.0, device=device),
                                       cfg.TRAIN.TEMPORAL_LOSS_WEIGHT)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    measure(f'closed-set model, batch {a.closedset_batch} x {cfg.DATA.CLIP_LENGTH} frames, '
            f'{cfg.DATA.TRAIN_CROP_SIZE} px, {cfg.TRAIN.AMP_DTYPE}', closed_step, results, device)

    def cnn_step(state={}):
        if not state:
            state['model'] = tecno.TwoHeadResNet50(pretrained=False).to(device).train()
            state['opt'] = torch.optim.Adam(state['model'].parameters(), lr=5e-4)
            state['ce'] = torch.nn.CrossEntropyLoss(weight=torch.tensor(tecno.CLASS_WEIGHTS, device=device))
        m, opt = state['model'], state['opt']
        x = torch.randn(a.tecno_batch, 3, a.tecno_size, a.tecno_size, device=device)
        _, phase, tool = m(x)
        loss = tecno.cnn_loss(m, phase, tool, torch.randint(0, 7, (a.tecno_batch,), device=device),
                              torch.randint(0, 2, (a.tecno_batch, 7), device=device).float(), state['ce'])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    measure(f'TeCNO stage 1 (ResNet-50), batch {a.tecno_batch}, {a.tecno_size} px, float32', cnn_step, results, device)

    def tcn_step(state={}):
        if not state:
            state['model'] = tecno.MultiStageTCN(**tecno.TCN).to(device).train()
            state['opt'] = torch.optim.Adam(state['model'].parameters(), lr=7e-5)
            state['ce'] = torch.nn.CrossEntropyLoss(weight=torch.tensor(tecno.CLASS_WEIGHTS, device=device))
        m, opt = state['model'], state['opt']
        out = m(torch.randn(1, 2048, a.tcn_steps, device=device))
        loss = tecno.tcn_loss(out, torch.randint(0, 7, (a.tcn_steps,), device=device), state['ce'])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    measure(f'TeCNO stage 2 (MS-TCN), {a.tcn_steps} steps', tcn_step, results, device)
    Path(a.output).write_text(json.dumps(results, indent=2))
    if not all(v['ok'] for v in results.values() if isinstance(v, dict)):
        sys.exit(1)


if __name__ == '__main__':
    main()
