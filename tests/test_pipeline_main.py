"""Orchestrator (main.py) logic that needs no GPU, no model download and no video decoding."""
import argparse
import json
import random
from pathlib import Path

import pandas as pd
import pytest

import main as pipeline


def _args(**kw):
    base = dict(max_phantom_rows=3, max_videos=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_find_inventory_handles_nested_folders_and_macos_junk(tmp_path):
    for v in ('video01', 'video02', 'video10'):
        (tmp_path / 'cholec80' / 'videos').mkdir(parents=True, exist_ok=True)
        (tmp_path / 'cholec80' / 'phase_annotations').mkdir(exist_ok=True)
        (tmp_path / 'cholec80' / 'tool_annotations').mkdir(exist_ok=True)
        (tmp_path / 'cholec80' / 'videos' / f'{v}.mp4').write_bytes(b'x')
        (tmp_path / 'cholec80' / 'videos' / f'{v}-timestamp.txt').write_text('unused')
        (tmp_path / 'cholec80' / 'phase_annotations' / f'{v}-phase.txt').write_text('Frame\tPhase\n')
        (tmp_path / 'cholec80' / 'tool_annotations' / f'{v}-tool.txt').write_text('Frame\n')
    (tmp_path / '__MACOSX').mkdir()
    (tmp_path / '__MACOSX' / '._video01.mp4').write_bytes(b'junk')
    inv = pipeline.find_inventory(tmp_path, None)
    assert list(inv) == ['video01', 'video02', 'video10']            # natural order, junk ignored
    assert list(pipeline.find_inventory(tmp_path, 2)) == ['video01', 'video02']
    (tmp_path / 'cholec80' / 'tool_annotations' / 'video02-tool.txt').unlink()
    with pytest.raises(pipeline.StageError, match='complete set'):
        pipeline.find_inventory(tmp_path, None)


def _sanitize_setup(tmp_path, phase_rows, frame_count=10):
    (tmp_path / 'v').mkdir()
    phase = tmp_path / 'v' / 'video01-phase.txt'
    tool = tmp_path / 'v' / 'video01-tool.txt'
    pd.DataFrame(phase_rows, columns=['Frame', 'Phase']).to_csv(phase, sep='\t', index=False)
    tools = ['Grasper', 'Bipolar', 'Hook', 'Scissors', 'Clipper', 'Irrigator', 'SpecimenBag']
    pd.DataFrame([[0] + [0] * 7], columns=['Frame'] + tools).to_csv(tool, sep='\t', index=False)
    paths = pipeline.Paths(tmp_path / 'root', 'r')
    paths.data.mkdir(parents=True)
    paths.raw.mkdir(parents=True)
    paths.metadata.write_text(json.dumps({'CHOLEC80__video01': dict(frame_count=frame_count, source_fps=25.0)}))
    paths.inventory.write_text(json.dumps(dict(base=str(tmp_path), videos={'video01': dict(
        video=str(tmp_path / 'v.mp4'), phase=str(phase), tool=str(tool))})))
    return paths


def test_sanitize_drops_a_phantom_row_that_repeats_the_last_phase(tmp_path):
    rows = [[i, 'Preparation'] for i in range(10)] + [[10, 'Preparation']]
    paths = _sanitize_setup(tmp_path, rows)
    pipeline.stage_sanitize_annotations(_args(), paths)
    cleaned = pd.read_csv(paths.clean / 'phase' / 'video01-phase.txt', sep='\t')
    assert cleaned.Frame.max() == 9 and len(cleaned) == 10
    report = json.loads((paths.data / 'annotations_sanitized.json').read_text())
    assert report['video01']['phase']['dropped_frames'] == [10]


def test_sanitize_refuses_a_row_that_changes_phase_or_too_many_rows(tmp_path):
    rows = [[i, 'Preparation'] for i in range(10)] + [[10, 'GallbladderRetraction']]
    with pytest.raises(pipeline.StageError, match='change the phase'):
        pipeline.stage_sanitize_annotations(_args(), _sanitize_setup(tmp_path, rows))
    tmp2 = tmp_path / 'two'
    tmp2.mkdir()
    rows = [[i, 'Preparation'] for i in range(15)]
    with pytest.raises(pipeline.StageError, match='refusing to guess'):
        pipeline.stage_sanitize_annotations(_args(max_phantom_rows=3), _sanitize_setup(tmp2, rows))


def test_new_split_matches_create_splits_for_80_videos_and_is_disjoint(tmp_path):
    ids = [f'CHOLEC80__video{i:02d}' for i in range(1, 81)]
    paths = pipeline.Paths(tmp_path, 'r')
    paths.data.mkdir(parents=True)
    paths.metadata.write_text(json.dumps({i: {} for i in ids}))
    pipeline.stage_splits(_args(splits_json=None, split_seed=42, val_ratio=.1, test_ratio=.1, max_videos=None), paths)
    split = json.loads(paths.splits.read_text())
    assert [len(split[k]) for k in ('train', 'val', 'test')] == [64, 8, 8]
    assert not (set(split['train']) & set(split['val']) | set(split['train']) & set(split['test']) | set(split['val']) & set(split['test']))
    shuffled = sorted(ids)
    random.Random(42).shuffle(shuffled)                               # exactly dataset_preprocessing/create_splits.py
    assert split == dict(train=shuffled[:64], val=shuffled[64:72], test=shuffled[72:])


def test_stage_signature_changes_with_video_limit_but_not_with_link():
    a = argparse.Namespace(text_model='m', weights_url='u', weights_path=None, random_init=False, data_url='d',
                           data_zip=None, data_dir=None, synthetic=None, max_videos=None, sample_fps=1.0, max_phantom_rows=3,
                           split_seed=1, val_ratio=.1, test_ratio=.1, splits_json=None, audit='all', run_name='r')
    before = {s: pipeline.stage_signature(s, a) for s in ('fetch_data', 'fetch_weights', 'extract_frames')}
    # an expired/rotated link, or an --offline re-run without any link, must still resume
    a.data_url, a.weights_url = 'https://new.example/fresh-token', None
    assert {s: pipeline.stage_signature(s, a) for s in before} == before
    a.max_videos = 6
    assert pipeline.stage_signature('fetch_data', a) != before['fetch_data']
    a.max_videos = None
    a.random_init = True                                              # switching to random weights IS a change
    assert pipeline.stage_signature('fetch_weights', a) != before['fetch_weights']
    a.random_init = False
    a.synthetic = 6                                                   # fake data must never mix with real data
    assert pipeline.stage_signature('fetch_data', a) != before['fetch_data']


def test_weights_location_is_remembered_between_invocations(tmp_path):
    p = pipeline.Paths(tmp_path, 'r')
    a = argparse.Namespace(random_init=False, weights_path=None)
    assert pipeline.read_weights_path(p, a) == str(p.weights)            # nothing known yet -> default location
    p.state.mkdir(parents=True)
    ckpt = tmp_path / 'elsewhere' / 'm2crl.pth'
    pipeline.write_json(p.weights_ref, dict(path=str(ckpt), random_init=False))
    assert pipeline.read_weights_path(p, a) == str(ckpt)                 # later job given no --weights-path
    a.weights_path = str(tmp_path / 'explicit.pth')
    assert pipeline.read_weights_path(p, a) == str((tmp_path / 'explicit.pth').resolve())
    a.random_init = True
    assert pipeline.read_weights_path(p, a) == ''


def test_runs_sharing_one_root_keep_separate_run_markers(tmp_path):
    pilot, full = pipeline.Paths(tmp_path, 'pilot_seed42'), pipeline.Paths(tmp_path, 'full_seed42')
    for stage in ('train', 'predict', 'evaluate'):                      # per run: pilot then full must not collide
        assert pilot.marker(stage) != full.marker(stage)
    for stage in ('fetch_data', 'extract_frames', 'splits', 'triplets', 'audit'):   # data is shared and reused
        assert pilot.marker(stage) == full.marker(stage)


@pytest.mark.parametrize('capability,expected', [((7, 0), False), ((7, 5), False), ((8, 0), True), ((9, 0), True)])
def test_auto_amp_uses_bf16_only_on_native_hardware(monkeypatch, capability, expected):
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda *a, **k: capability)
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda *a, **k: True)   # emulated support must not count
    assert pipeline.native_bf16() is expected


def test_real_runs_require_an_explicit_root(monkeypatch):
    monkeypatch.delenv('LGEL_ROOT', raising=False)
    for preset in ('pilot', 'full'):
        with pytest.raises(SystemExit) as e:
            pipeline.main(['--preset', preset, '--dry-run'])
        assert e.value.code == 2


def test_run_sh_keeps_relative_paths_relative_to_the_callers_directory(tmp_path):
    import os
    import subprocess
    repo = Path(pipeline.__file__).resolve().parent
    env = dict(os.environ, LGEL_SKIP_ENV='1')
    env.pop('LGEL_ROOT', None)
    out = subprocess.run(['bash', str(repo / 'run.sh'), '--preset', 'smoke', '--root', 'rel_root', '--dry-run'],
                         cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert f'root={tmp_path.resolve() / "rel_root"}' in out.stdout
    assert not (repo / 'rel_root').exists()
