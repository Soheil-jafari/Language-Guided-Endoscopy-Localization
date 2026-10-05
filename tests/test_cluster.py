"""Cluster integration: Slurm scripts, CPU allocation, report bundle, weights export (no Slurm needed)."""
import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

import main as pipeline

REPO = Path(pipeline.__file__).resolve().parent
posix_only = pytest.mark.skipif(os.name == 'nt', reason='Linux cluster scripts')


def test_cpu_count_never_exceeds_the_slurm_allocation(monkeypatch):
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '1')
    assert pipeline.cpu_count() == 1
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '100000')
    assert pipeline.cpu_count() == len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else True
    monkeypatch.delenv('SLURM_CPUS_PER_TASK')
    assert pipeline.cpu_count() >= 1


def test_every_job_leaves_a_small_report_bundle_without_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.FileLock, 'disabled', False)          # restored after the test
    root = tmp_path / 'root'
    p = pipeline.Paths(root, 'smoke_seed42')
    p.run.mkdir(parents=True)
    (p.run / 'training_metrics.jsonl').write_text(json.dumps(dict(epoch=1)) + '\n')
    (p.run / 'latest_model.pth').write_bytes(b'x' * 100)
    p.results.mkdir(parents=True)
    (p.results / 'model_weights.pth').write_bytes(b'x' * 100)
    args = ['--preset', 'smoke', '--root', str(root), '--stages', 'summary', '--allow-cpu', '--single-job']
    assert pipeline.main(args) == 0
    latest = (p.outbox / 'LATEST.txt').read_text().strip()
    assert latest.endswith('-OK.tar.gz')
    with tarfile.open(p.outbox / latest) as tar:
        names = tar.getnames()
    assert 'logs/main.log' in names and 'runs/smoke_seed42/training_metrics.jsonl' in names
    assert 'results/smoke_seed42/SUMMARY.md' in names
    assert not [n for n in names if n.endswith('.pth')]                 # checkpoints stay on the cluster


def test_best_checkpoint_is_exported_without_optimizer_state(tmp_path):
    import torch
    p = pipeline.Paths(tmp_path, 'r')
    p.run.mkdir(parents=True)
    p.results.mkdir(parents=True)
    torch.save(dict(schema_version=2, config={}, model_state_dict={'w': torch.ones(3)}, epoch=5,
                    validation_threshold=0.4, optimizer={'big': torch.ones(100)}, scheduler={}, scaler={}, rng={}),
               p.run / 'best_model.pth')
    pipeline.export_model_weights(p)
    slim = torch.load(p.results / 'model_weights.pth', weights_only=False)
    assert set(slim) == {'schema_version', 'config', 'model_state_dict', 'epoch', 'validation_threshold'}


def _stub_bin(tmp_path, squeue_output=''):
    stub = tmp_path / 'bin'
    stub.mkdir()
    (stub / 'sbatch').write_text('#!/bin/bash\necho "$@" >> "$STUB_LOG"\necho "$((1000 + $(wc -l < "$STUB_LOG")))"\n')
    (stub / 'squeue').write_text(f'#!/bin/bash\nprintf "%s" "{squeue_output}"\n')
    for f in stub.iterdir():
        f.chmod(0o755)
    return stub


def _submit(tmp_path, *args, squeue_output=''):
    stub = _stub_bin(tmp_path, squeue_output)
    env = dict(os.environ, PATH=f'{stub}:{os.environ["PATH"]}', LGEL_BASE=str(tmp_path / 'base'),
               STUB_LOG=str(tmp_path / 'sbatch.log'), USER='tester')
    (tmp_path / 'sbatch.log').write_text('')
    out = subprocess.run(['bash', str(REPO / 'hpc' / 'submit.sh'), *args], env=env, capture_output=True, text=True)
    calls = [l for l in (tmp_path / 'sbatch.log').read_text().splitlines() if l]
    return out, calls


@posix_only
def test_submit_script_builds_the_right_slurm_jobs(tmp_path):
    out, calls = _submit(tmp_path, 'smoke')
    assert out.returncode == 0, out.stderr
    assert len(calls) == 1
    call = calls[0]
    base = tmp_path / 'base'
    assert '--job-name=lgel' in call and '--time=0-04:00:00' in call
    assert f'--output={base}/lgel_smoke/logs/slurm-%j.out' in call
    assert f'LGEL_ROOT={base}/lgel_smoke' in call and 'LGEL_PRESET=smoke' in call
    assert call.endswith('hpc/lgel.sbatch')
    assert (base / 'lgel_smoke' / 'logs').is_dir()                    # Slurm needs the log folder to exist


@posix_only
def test_submit_script_refuses_a_second_job_and_missing_links(tmp_path):
    out, calls = _submit(tmp_path, 'full', squeue_output='123 lgel RUNNING')
    assert out.returncode == 1 and 'already queued or running' in out.stderr and not calls
    out, calls = _submit(tmp_path / 'b', 'pilot') if (tmp_path / 'b').mkdir() is None else (None, None)
    assert out.returncode == 1 and 'links.env is missing' in out.stderr and not calls


@posix_only
def test_submit_script_chains_follow_up_jobs_and_passes_options(tmp_path):
    (tmp_path / 'base').mkdir()
    (tmp_path / 'base' / 'links.env').write_text("LGEL_DATA_URL='https://data.invalid/d.zip'\n"
                                                 "LGEL_WEIGHTS_URL='https://weights.invalid/w.pth'\n")
    out, calls = _submit(tmp_path, 'full', '--repeat', '2', '--batch-size', '4', '--accum', '48')
    assert out.returncode == 0, out.stderr
    assert len(calls) == 2
    assert '--dependency' not in calls[0] and '--dependency=afterany:1001' in calls[1]
    assert all(c.endswith('hpc/lgel.sbatch --batch-size 4 --accum 48') for c in calls)
    assert all('--time=3-00:00:00' in c for c in calls)


@posix_only
def test_job_script_runs_the_pipeline_with_the_cluster_settings(tmp_path):
    stub = tmp_path / 'bin'
    stub.mkdir()
    (stub / 'module').write_text('#!/bin/bash\necho "module $*" >> "$STUB_LOG"\n')
    (stub / 'module').chmod(0o755)
    base = tmp_path / 'base'
    base.mkdir()
    (base / 'links.env').write_text("LGEL_DATA_URL='https://example.invalid/data.zip'\n")
    env = dict(os.environ, PATH=f'{stub}:{os.environ["PATH"]}', STUB_LOG=str(tmp_path / 'module.log'),
               LGEL_REPO=str(REPO), LGEL_BASE=str(base), LGEL_PRESET='smoke', LGEL_ROOT=str(base / 'lgel_smoke'),
               LGEL_SKIP_ENV='1')
    env.pop('LGEL_WEIGHTS_URL', None)
    out = subprocess.run(['bash', str(REPO / 'hpc' / 'lgel.sbatch'), '--dry-run'], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert 'module load apps/miniconda3' in (tmp_path / 'module.log').read_text()
    assert f'root={base / "lgel_smoke"}' in out.stdout
    for d in ('conda/envs', 'conda/pkgs', 'conda/pip-cache', 'tmp'):
        assert (base / d).is_dir()                                     # env and caches kept off $HOME
