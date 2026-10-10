"""The full preset trains the baseline, then the advanced model; the advanced add-ons are checked early."""
import argparse
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

import main as pipeline

REPO = Path(pipeline.__file__).resolve().parent


@pytest.mark.parametrize('argv,expected', [
    (['--preset', 'full'], [('baseline', 'full_baseline_seed42'), ('advanced', 'full_advanced_seed42')]),
    (['--preset', 'pilot'], [('baseline', 'pilot_seed42')]),
    (['--preset', 'smoke'], [('baseline', 'smoke_seed42')]),
    (['--preset', 'full', '--variant', 'advanced'], [('advanced', 'full_advanced_seed42')]),
    (['--preset', 'full', '--variant', 'baseline'], [('baseline', 'full_baseline_seed42')]),
    (['--preset', 'pilot', '--variant', 'advanced'], [('advanced', 'pilot_advanced_seed42')]),
    (['--preset', 'full', '--run-name', 'x'], [('baseline', 'x_baseline'), ('advanced', 'x_advanced')]),
])
def test_which_models_each_preset_trains(tmp_path, monkeypatch, argv, expected):
    monkeypatch.setattr(pipeline.FileLock, 'disabled', False)
    seen = []
    monkeypatch.setattr(pipeline, '_drive', lambda a, p, s, r: seen.append((a.variant, a.run_name)) or 0)
    monkeypatch.setattr(pipeline, 'write_report_bundle', lambda *a, **k: None)
    monkeypatch.setattr(pipeline, 'write_comparison', lambda *a, **k: None)
    assert pipeline.main(['--root', str(tmp_path), '--single-job', *argv]) == 0
    assert seen == expected


def test_the_advanced_model_is_trained_only_after_the_baseline_succeeded(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(pipeline, '_drive', lambda a, p, s, r: seen.append(a.variant) or 1)
    monkeypatch.setattr(pipeline, 'write_report_bundle', lambda *a, **k: None)
    assert pipeline.main(['--root', str(tmp_path), '--single-job', '--preset', 'full']) == 1
    assert seen == ['baseline']


def test_only_the_advanced_run_switches_the_add_ons_on(tmp_path):
    p = pipeline.Paths(tmp_path, 'r')
    base = argparse.Namespace(seed=1, epochs=2, warmup=1, batch_size=2, accum=2, subset_ratio=0.1, sample_fps=1.0,
                              text_model='t', random_init=True, weights_path=None, variant='baseline')
    cfg = pipeline.train_config(base, p, 0, 'bf16')
    assert 'TEMPORAL_HEAD_TYPE' not in cfg['MODEL'] and 'USE_BILEVEL_CONSISTENCY' not in cfg['TRAIN']
    adv = argparse.Namespace(**dict(vars(base), variant='advanced'))
    cfg = pipeline.train_config(adv, p, 0, 'bf16')
    assert cfg['MODEL']['TEMPORAL_HEAD_TYPE'] == 'SSM' and cfg['MODEL']['SSM_USE_OFFICIAL_MAMBA'] is True
    assert cfg['MODEL']['USE_UNCERTAINTY'] is True and cfg['MODEL']['USE_CONFIDENCE_FUSION'] is True
    assert cfg['TRAIN']['USE_BILEVEL_CONSISTENCY'] is True
    assert cfg['TRAIN']['NUM_EPOCHS'] == 2 and cfg['TRAIN']['BATCH_SIZE'] == 2      # same training settings


def test_every_advanced_flag_exists_in_the_project_configuration():
    from project_config import Config
    c = Config()
    for section, values in pipeline.ADVANCED_FLAGS.items():
        for key in values:
            assert hasattr(getattr(c, section), key), f'{section}.{key}'


@pytest.mark.parametrize('preset,variant,checked', [('smoke', None, True), ('pilot', None, False), ('full', None, True),
                                                     ('pilot', 'advanced', True), ('full', 'baseline', False)])
def test_the_advanced_add_ons_are_checked_whenever_they_will_be_needed(tmp_path, monkeypatch, preset, variant, checked):
    seen = []
    monkeypatch.setattr(pipeline, '_drive', lambda a, p, s, r: seen.append(a.check_advanced) or 0)
    monkeypatch.setattr(pipeline, 'write_report_bundle', lambda *a, **k: None)
    monkeypatch.setattr(pipeline, 'write_comparison', lambda *a, **k: None)
    pipeline.main(['--root', str(tmp_path), '--single-job', '--preset', preset] + (['--variant', variant] if variant else []))
    assert seen and all(c == checked for c in seen)


def test_advanced_check_is_skipped_without_a_gpu_and_reports_a_missing_mamba(tmp_path):
    cfg = tmp_path / 'cfg.json'
    cfg.write_text(json.dumps(dict(MODEL=pipeline.ADVANCED_FLAGS['MODEL'], TRAIN=pipeline.ADVANCED_FLAGS['TRAIN'])))
    out = subprocess.run([sys.executable, str(REPO / 'main.py'), '_check_advanced', str(cfg)], capture_output=True,
                         text=True, cwd=REPO)
    import torch
    if not torch.cuda.is_available():                          # (the weights download may be blocked offline)
        assert (out.returncode == 0 and 'SKIPPED: no CUDA GPU' in out.stdout) or \
               (out.returncode == pipeline.FLOW_WEIGHTS_MISSING and 'OPTICAL-FLOW WEIGHTS UNAVAILABLE' in out.stdout)


def test_comparison_and_report_cover_both_models(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.FileLock, 'disabled', False)
    a = argparse.Namespace(run_name='full_seed42', preset='full')
    plans = []
    for variant, ap in (('baseline', 0.41), ('advanced', 0.47)):
        sub = argparse.Namespace(variant=variant, run_name=f'full_{variant}_seed42')
        sp = pipeline.Paths(tmp_path, sub.run_name)
        sp.results.mkdir(parents=True)
        (sp.results / 'SUMMARY.md').write_text(f'# {variant}\n')
        (sp.results / 'proposed_test_metrics.json').write_text(json.dumps(dict(average_precision=ap, auroc=0.8, f1=0.5,
                                                                               precision=0.5, recall=0.5, brier=0.2, ece=0.1)))
        plans.append((sub, sp))
    pipeline.write_comparison(a, plans)
    table = (tmp_path / 'results' / 'full_seed42_COMPARISON.md').read_text()
    assert '| baseline | full_baseline_seed42 | 0.4100' in table and '| advanced | full_advanced_seed42 | 0.4700' in table
    pipeline.write_report_bundle(a, [sp for _, sp in plans], 0)
    name = (tmp_path / 'outbox' / 'LATEST.txt').read_text().strip()
    assert name.startswith('full_seed42-') and name.endswith('-OK.tar.gz')
    with tarfile.open(tmp_path / 'outbox' / name) as tar:
        names = tar.getnames()
    assert {'results/full_seed42_COMPARISON.md', 'results/full_baseline_seed42/SUMMARY.md',
            'results/full_advanced_seed42/SUMMARY.md'} <= set(names)


def test_setup_installs_and_verifies_mamba():
    assert 'bash "$REPO/install_mamba.sh" "$PY" || true' in (REPO / 'setup_env.sh').read_text()
    text = (REPO / 'install_mamba.sh').read_text()
    assert 'releases/download/v$MAMBA_VERSION/$MAMBA_ASSET' in text and '--max-time' in text
    assert "import mamba_ssm, selective_scan_cuda" in text and '--no-deps' in text
    assert 'mamba-ssm==$MAMBA_VERSION' in text                     # compile fallback when nvcc exists


def test_a_finished_baseline_is_left_alone_so_the_advanced_model_can_use_other_settings(tmp_path, monkeypatch):
    base = pipeline.Paths(tmp_path, 'full_baseline_seed42')
    base.results.mkdir(parents=True)
    base.state.mkdir(parents=True)
    (base.results / 'proposed_test_metrics.json').write_text('{}')
    pipeline.write_json(base.marker('evaluate'), dict(signature={}))
    seen = []
    monkeypatch.setattr(pipeline, '_drive', lambda a, p, s, r: seen.append((a.run_name, a.batch_size, s[-1], len(s))) or 0)
    monkeypatch.setattr(pipeline, 'write_report_bundle', lambda *a, **k: None)
    monkeypatch.setattr(pipeline, 'write_comparison', lambda *a, **k: None)
    assert pipeline.main(['--root', str(tmp_path), '--single-job', '--preset', 'full', '--batch-size', '4',
                          '--accum', '48']) == 0
    # the baseline only gets its summary redone (data check, weights export); the advanced model trains
    assert seen == [('full_baseline_seed42', 4, 'summary', 1), ('full_advanced_seed42', 4, 'summary', len(pipeline.STAGES))]
    seen.clear()
    pipeline.main(['--root', str(tmp_path), '--single-job', '--preset', 'full', '--force', 'train'])
    assert [(r, n) for r, _, _, n in seen] == [('full_baseline_seed42', len(pipeline.STAGES)),
                                               ('full_advanced_seed42', len(pipeline.STAGES))]   # unless redone


def test_a_finished_baseline_whose_data_changed_stops_the_advanced_model(tmp_path, monkeypatch):
    base = pipeline.Paths(tmp_path, 'full_baseline_seed42')
    base.results.mkdir(parents=True)
    base.state.mkdir(parents=True)
    (base.results / 'proposed_test_metrics.json').write_text('{}')
    pipeline.write_json(base.marker('evaluate'), dict(signature={}))
    seen = []
    def drive(a, p, stages, redo):                               # its summary refuses the changed data
        seen.append(a.run_name)
        return 1 if stages == ['summary'] else 0
    monkeypatch.setattr(pipeline, '_drive', drive)
    monkeypatch.setattr(pipeline, 'write_report_bundle', lambda *a, **k: None)
    assert pipeline.main(['--root', str(tmp_path), '--single-job', '--preset', 'full']) == 1
    assert seen == ['full_baseline_seed42']


def test_comparison_lists_each_models_own_training_settings(tmp_path):
    a = argparse.Namespace(run_name='full_seed42')
    plans = []
    for variant, batch in (('baseline', 8), ('advanced', 4)):
        sub = argparse.Namespace(variant=variant, run_name=f'full_{variant}_seed42')
        sp = pipeline.Paths(tmp_path, sub.run_name)
        sp.run_cfg.parent.mkdir(parents=True, exist_ok=True)
        sp.run_cfg.write_text(json.dumps(dict(TRAIN=dict(NUM_EPOCHS=20, BATCH_SIZE=batch, GRADIENT_ACCUMULATION_STEPS=192 // batch,
                                                         AMP_DTYPE='bf16'))))
        plans.append((sub, sp))
    pipeline.write_comparison(a, plans)
    table = (tmp_path / 'results' / 'full_seed42_COMPARISON.md').read_text()
    assert '| 20 | 8 x 24 | bf16 |' in table and '| 20 | 4 x 48 | bf16 |' in table


def test_optical_flow_weights_are_fetched_even_without_a_gpu(tmp_path, monkeypatch):
    """The offline flow: the advanced loss's weights must be cached by the online (possibly GPU-less) job."""
    import torch
    if torch.cuda.is_available():
        pytest.skip('checks the GPU-less path')
    calls = []
    import losses
    monkeypatch.setattr(losses.MasterLoss, '__init__', lambda self, config: calls.append(config.TRAIN.USE_BILEVEL_CONSISTENCY) or None)
    cfg = tmp_path / 'cfg.json'
    cfg.write_text(json.dumps(dict(MODEL=pipeline.ADVANCED_FLAGS['MODEL'], TRAIN=pipeline.ADVANCED_FLAGS['TRAIN'])))
    pipeline._check_advanced(cfg)
    assert calls == [True]


def test_each_summary_records_what_that_model_was_trained_with(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'export_model_weights', lambda p: None)
    p = pipeline.Paths(tmp_path, 'full_advanced_seed42')
    p.run_cfg.parent.mkdir(parents=True)
    p.run_cfg.write_text(json.dumps(dict(MODEL=dict(pipeline.ADVANCED_FLAGS['MODEL']),
                                         TRAIN=dict(NUM_EPOCHS=20, BATCH_SIZE=4, GRADIENT_ACCUMULATION_STEPS=48, AMP_DTYPE='bf16'))))
    a = argparse.Namespace(run_name='full_advanced_seed42', preset='full', variant='advanced', batch_size=8, subset_ratio=1.0)
    pipeline.stage_summary(a, p)
    text = (p.results / 'SUMMARY.md').read_text()
    assert 'Model: **advanced** (Mamba temporal head' in text and 'batch 4 x 48 accumulation' in text
    summary = json.loads((p.results / 'summary.json').read_text())
    assert summary['trained_with']['TRAIN']['BATCH_SIZE'] == 4 and summary['latest_command_args']['batch_size'] == '8'


def test_offline_jobs_refuse_missing_optical_flow_weights_without_touching_the_network(tmp_path, monkeypatch):
    p = pipeline.Paths(tmp_path, 'full_advanced_seed42')
    a = argparse.Namespace(offline=True, amp_dtype='bf16', text_model='t', variant='advanced')
    launched = []
    monkeypatch.setattr(pipeline, 'run', lambda *x, **k: launched.append(x))
    monkeypatch.setattr(pipeline, '_ADVANCED_CHECKED', False)
    with pytest.raises(pipeline.StageError, match='--offline'):
        pipeline.stage_check_advanced(a, p)
    assert not launched                                          # nothing was started that could download
    f = pipeline.flow_weights_file(p)
    f.parent.mkdir(parents=True)
    f.write_bytes(b'cached')
    pipeline.stage_check_advanced(a, p)                          # cached: the check runs
    assert len(launched) == 1


def test_no_comparison_is_written_until_both_models_have_results(tmp_path, monkeypatch):
    written = []
    monkeypatch.setattr(pipeline, '_drive', lambda a, p, s, r: 0)
    monkeypatch.setattr(pipeline, 'write_report_bundle', lambda *a, **k: None)
    monkeypatch.setattr(pipeline, 'write_comparison', lambda *a, **k: written.append(1))
    assert pipeline.main(['--root', str(tmp_path), '--single-job', '--preset', 'full', '--to-stage', 'fetch_data']) == 0
    assert not written
