"""Orchestrator (main.py) logic that needs no GPU, no model download and no video decoding."""
import argparse
import json
import os
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


@pytest.mark.skipif(os.name == 'nt', reason='run.sh is a Linux/HPC wrapper')
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


# ----------------------------------------------------------------- review round 2 (Codex findings)
def test_dry_run_creates_and_deletes_nothing(tmp_path, capsys):
    root = tmp_path / 'root'
    (root / 'state').mkdir(parents=True)
    marker = root / 'state' / 'fetch_data.done.json'
    marker.write_text('{"signature": {}}')
    before = sorted(str(x) for x in root.rglob('*'))
    assert pipeline.main(['--preset', 'smoke', '--root', str(root), '--force', 'fetch_data', '--dry-run']) == 0
    assert marker.exists()                                            # --force must not act during a dry run
    assert sorted(str(x) for x in root.rglob('*')) == before          # no logs/, tmp/ ... created
    assert 'run (forced)' in capsys.readouterr().out
    assert pipeline.main(['--preset', 'smoke', '--root', str(tmp_path / 'new'), '--dry-run']) == 0
    assert not (tmp_path / 'new').exists()


def _dl_args(**kw):
    return argparse.Namespace(**{'offline': False, **kw})


def test_download_continues_partial_files_reuses_complete_ones_and_respects_offline(tmp_path, monkeypatch):
    calls = []
    def fake_download(url, dest, a, p, name):
        calls.append(url)
        Path(dest).write_bytes(b'x' * 10)
    monkeypatch.setattr(pipeline, '_download', fake_download)
    p = pipeline.Paths(tmp_path, 'r')
    dest = tmp_path / 'weights' / 'checkpoint.pth'
    dest.parent.mkdir()
    dest.write_bytes(b'x' * 3)                                        # partial file left by a killed job
    with pytest.raises(pipeline.StageError, match='--offline'):
        pipeline.download('https://example.invalid/w', dest, _dl_args(offline=True), p, 'w')
    assert calls == []                                                # offline: no network attempt at all
    pipeline.download('https://example.invalid/w', dest, _dl_args(), p, 'w')
    assert calls == ['https://example.invalid/w']                     # partial file -> downloader runs (resumes)
    pipeline.download(None, dest, _dl_args(offline=True), p, 'w')     # complete -> reused, even offline / no URL
    assert calls == ['https://example.invalid/w']
    dest.write_bytes(b'x' * 4)                                        # changed size -> not trusted any more
    with pytest.raises(pipeline.StageError, match='no download link'):
        pipeline.download(None, dest, _dl_args(), p, 'w')


def test_find_inventory_rejects_duplicate_video_names(tmp_path):
    for sub in ('a', 'b'):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / 'video01.mp4').write_bytes(sub.encode())
    (tmp_path / 'video01-phase.txt').write_text('Frame\tPhase\n')
    (tmp_path / 'video01-tool.txt').write_text('Frame\tGrasper\n')
    with pytest.raises(pipeline.StageError, match='Duplicate'):
        pipeline.find_inventory(tmp_path, None)


def test_metrics_journal_is_repaired_to_match_the_checkpoint(tmp_path):
    p = pipeline.Paths(tmp_path, 'r')
    p.run.mkdir(parents=True)
    f = p.run / 'training_metrics.jsonl'
    rows = [dict(epoch=1, train_loss=1.0), dict(epoch=2, train_loss=0.9), dict(epoch=2, train_loss=0.8)]
    f.write_text(''.join(json.dumps(r) + '\n' for r in rows) + '{"epoch": 3, "train_lo')   # replay + torn line
    assert pipeline.repair_metrics_journal(p, 3) == [3]                 # epoch 3 saved, but its row never written
    kept = [json.loads(l) for l in f.read_text().splitlines()]
    assert kept == [dict(epoch=1, train_loss=1.0), dict(epoch=2, train_loss=0.8)]
    assert f.read_text().endswith('\n')                               # train.py can append safely again
    assert list(p.run.glob('training_metrics.jsonl.before-repair-*'))
    assert pipeline.repair_metrics_journal(p, 1) == []                # rows beyond the checkpoint are dropped
    assert [json.loads(l)['epoch'] for l in f.read_text().splitlines()] == [1]


def test_completed_epochs_come_from_the_checkpoint_not_the_log(tmp_path):
    import torch
    p = pipeline.Paths(tmp_path, 'r')
    assert pipeline.completed_epochs(p) == 0
    p.run.mkdir(parents=True)
    torch.save(dict(epoch=4), p.run / 'latest_model.pth')
    (p.run / 'training_metrics.jsonl').write_text(json.dumps(dict(epoch=3)) + '\n')
    assert pipeline.completed_epochs(p) == 4


def test_a_run_killed_in_its_first_epoch_is_restarted_without_deleting_checkpoints(tmp_path):
    p = pipeline.Paths(tmp_path, 'r')
    p.run.mkdir(parents=True)
    (p.run / 'best_model.pth').write_bytes(b'x')                     # best saved, latest not yet
    aside = pipeline.recover_unresumable_run(p)
    assert aside is not None and (aside / 'best_model.pth').exists() and not p.run.exists()
    p.run.mkdir()
    (p.run / 'run_config.json').write_text('{}')                     # died before any checkpoint
    assert pipeline.recover_unresumable_run(p) is None and not p.run.exists()
    p.run.mkdir()
    (p.run / 'latest_model.pth').write_bytes(b'x')                   # resumable: untouched
    assert pipeline.recover_unresumable_run(p) is None and (p.run / 'latest_model.pth').exists()


def test_runs_refuse_data_that_changed_after_they_started(tmp_path):
    p = pipeline.Paths(tmp_path, 'full_seed42')
    with pytest.raises(pipeline.StageError, match='incomplete'):     # no data yet: refuse, record nothing
        pipeline.check_run_inputs(p, 'full_seed42')
    assert not p.run_inputs.exists()
    p.triplets.mkdir(parents=True)
    for f in (p.splits, p.metadata, p.parsed, *(p.triplet_csv(s) for s in ('train', 'val', 'test'))):
        f.write_text('original')
    pipeline.check_run_inputs(p, 'full_seed42')                       # first launch records the data
    pipeline.check_run_inputs(p, 'full_seed42')                       # unchanged: fine
    p.splits.write_text('{"test": ["video01"]}')
    pipeline.check_run_inputs(p, 'full_seed42')                       # nothing produced yet: adopts the new data
    p.run.mkdir(parents=True)
    (p.run / 'latest_model.pth').write_bytes(b'x')                   # now the run has a checkpoint ...
    p.splits.write_text('{"test": ["a video the model was trained on"]}')
    with pytest.raises(pipeline.StageError, match='changed since run'):
        pipeline.check_run_inputs(p, 'full_seed42')                   # ... so changed data is refused
    other = pipeline.Paths(tmp_path, 'full_seed43')                   # a NEW run on the new data is fine
    pipeline.check_run_inputs(other, 'full_seed43')


def test_a_link_that_returns_a_web_page_is_rejected_and_not_remembered(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, '_download',
                        lambda url, dest, a, p, name: Path(dest).write_bytes(b'\n  <!DOCTYPE html><html>share page'))
    p = pipeline.Paths(tmp_path, 'r')
    dest = tmp_path / 'weights' / 'checkpoint.pth'
    with pytest.raises(pipeline.StageError, match='web page'):
        pipeline.download('https://pan.example/s/abc', dest, _dl_args(), p, 'w')
    assert not dest.exists() and not dest.with_name('checkpoint.pth.complete').exists()


def test_reextraction_keeps_valid_frames_when_raw_videos_are_gone(tmp_path):
    p = pipeline.Paths(tmp_path, 'r')
    p.shards.mkdir(parents=True)
    inv = {}
    for v in ('video01', 'video02'):
        (p.frames / f'CHOLEC80__{v}').mkdir(parents=True)
        (p.shards / f'CHOLEC80__{v}.json').write_text(json.dumps({f'CHOLEC80__{v}': dict(sample_fps=1.0, frame_count=50)}))
        inv[v] = dict(video=str(tmp_path / 'deleted' / f'{v}.mp4'), phase='x', tool='x')
    pipeline.write_json(p.inventory, dict(base=str(tmp_path), videos=inv))
    a = argparse.Namespace(sample_fps=2.0, workers=1, cleanup_raw=False, data_dir=None, data_zip=None)
    with pytest.raises(pipeline.StageError, match='raw videos'):
        pipeline.stage_extract_frames(a, p)
    assert len(list(p.shards.glob('*.json'))) == 2                    # the 1 fps metadata is still usable


@pytest.mark.skipif(pipeline.fcntl is None, reason='POSIX file locks only')
def test_locks_detect_another_live_job(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.FileLock, 'mode', 'fcntl')
    import subprocess
    import sys
    p = pipeline.Paths(tmp_path, 'full_seed42')
    p.state.mkdir(parents=True)
    holder = subprocess.Popen([sys.executable, '-c',
                               'import fcntl, sys, time; f = open(sys.argv[1], "a+"); fcntl.lockf(f, fcntl.LOCK_EX); '
                               'print("locked", flush=True); time.sleep(60)', str(p.run_lock)],
                              stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == 'locked'
        assert pipeline.FileLock(p.run_lock).acquire(blocking=False) is False
        assert pipeline.busy_runs(p) == ['full_seed42']
    finally:
        holder.kill()
        holder.wait()
    lock = pipeline.FileLock(p.run_lock)
    assert lock.acquire(blocking=False) is True                       # released when the holder died
    lock.release()
    assert pipeline.busy_runs(p) == []


def test_only_one_gpu_is_exposed_when_a_job_sees_several(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,3')
    assert pipeline.pin_single_gpu(2) == '2'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '2'
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    assert pipeline.pin_single_gpu(4) == '0'
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    assert pipeline.pin_single_gpu(1) is None and 'CUDA_VISIBLE_DEVICES' not in os.environ


@pytest.mark.parametrize('rc,discarded', [(pipeline.CHECKPOINT_REJECTED, True), (-9, False), (1, False)])
def test_only_a_rejected_checkpoint_is_set_aside(tmp_path, monkeypatch, rc, discarded):
    p = pipeline.Paths(tmp_path, 'r')
    p.weights.parent.mkdir(parents=True)
    p.weights.write_bytes(b'PK\x03\x04 checkpoint')
    pipeline.write_json(p.weights.with_name('checkpoint.pth.complete'), dict(size=p.weights.stat().st_size))
    def failing_run(cmd, a, p, name, **kw):
        err = pipeline.StageError(f'exited with code {rc}')
        err.rc = rc
        raise err
    monkeypatch.setattr(pipeline, 'run', failing_run)
    a = argparse.Namespace(random_init=False, weights_path=None, weights_url='https://x.invalid/w', offline=False,
                           text_model='m')
    with pytest.raises(pipeline.StageError):
        pipeline.stage_fetch_weights(a, p)
    assert p.weights.exists() is (not discarded)                     # out of memory / killed: keep the good file
    assert bool(list(p.weights_dir.glob('checkpoint.pth.rejected-*'))) is discarded


# ----------------------------------------------------------------- review round 4 (second Codex pass)
def test_disk_check_asks_only_for_what_the_remaining_stages_need(tmp_path):
    p = pipeline.Paths(tmp_path, 'r')
    a = argparse.Namespace(min_free_gb=200, synthetic=None, data_dir=None, data_zip=None)
    stages = list(pipeline.STAGES)
    assert pipeline.disk_needed_gb(a, p, stages)[0] == 200                     # nothing downloaded yet
    p.state.mkdir(parents=True)
    video = tmp_path / 'v.mp4'
    video.write_bytes(b'x' * 1000)
    pipeline.write_json(p.inventory, dict(base=str(tmp_path), videos={'video01': dict(video=str(video))}))
    pipeline.write_json(p.marker('fetch_data'), {})                             # downloaded and unpacked
    need, what = pipeline.disk_needed_gb(a, p, stages)
    assert what == 'frame extraction' and need < 11                           # only the frames are still to come
    for s in pipeline.STAGES[:pipeline.STAGES.index('audit') + 1]:
        pipeline.write_json(p.marker(s), {})
    assert pipeline.disk_needed_gb(a, p, stages) == (10, 'checkpoints and results')
    a.min_free_gb = 5
    assert pipeline.disk_needed_gb(a, p, stages)[0] == 5             # --min-free-gb still overrides
    a.min_free_gb = 200
    assert pipeline.disk_needed_gb(a, p, ['preflight', 'summary']) == (0, '')


@pytest.fixture
def lease_mode(monkeypatch):
    monkeypatch.setattr(pipeline.FileLock, 'mode', 'lease')
    monkeypatch.setattr(pipeline.FileLock, '_held_leases', [])
    monkeypatch.setattr(pipeline, 'LEASE_STALE', 1.0)
    monkeypatch.setattr(pipeline, 'LEASE_BEAT', 0.1)
    import signal
    monkeypatch.setattr(signal, 'signal', lambda *a, **k: None)


def test_lease_locks_exclude_each_other_and_expire_when_the_holder_dies(tmp_path, lease_mode):
    holder, other = pipeline.FileLock(tmp_path / 'prepare.lock'), pipeline.FileLock(tmp_path / 'prepare.lock')
    assert holder.acquire(blocking=False)
    assert pipeline.FileLock.busy(tmp_path / 'prepare.lock')
    assert other.acquire(blocking=False) is False                     # a live holder keeps its heartbeat going
    holder._stop.set()                                                # holder "killed": heartbeat stops, file stays
    import time
    time.sleep(1.5)
    assert not pipeline.FileLock.busy(tmp_path / 'prepare.lock')
    assert other.acquire(blocking=True)                               # abandoned lease is taken over
    other.release()
    assert not (tmp_path / 'prepare.lock.lease').exists()


def test_unsupported_locks_switch_to_leases_instead_of_being_ignored(tmp_path, monkeypatch, lease_mode):
    import errno
    monkeypatch.setattr(pipeline.FileLock, 'mode', None)
    def no_locks(*a, **k):
        raise OSError(errno.ENOLCK, 'No locks available')
    monkeypatch.setattr(pipeline.fcntl, 'lockf', no_locks)
    first, second = pipeline.FileLock(tmp_path / 'l'), pipeline.FileLock(tmp_path / 'l')
    assert first.acquire(blocking=False) and pipeline.FileLock.mode == 'lease'
    assert second.acquire(blocking=False) is False                    # previously both "acquired"
    first.release()


def test_a_corrupt_downloaded_archive_is_discarded_not_reused(tmp_path):
    import io
    import zipfile
    p = pipeline.Paths(tmp_path, 'r')
    p.raw.mkdir(parents=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('cholec80/videos/video01.mp4', os.urandom(200000))
    data = bytearray(buf.getvalue())
    start = data.find(b'PK\x03\x04') + 30 + len('cholec80/videos/video01.mp4')
    data[start:start + 5000] = b'\x00' * 5000                         # damaged compressed data
    p.zip.write_bytes(bytes(data))
    pipeline.write_json(p.zip.with_name('dataset.zip.complete'), dict(size=len(data)))
    a = argparse.Namespace(data_url='https://x.invalid/d.zip', data_zip=None, data_dir=None, synthetic=None,
                           offline=False, skip_zip_check=False, keep_zip=False, max_videos=None)
    with pytest.raises(pipeline.StageError, match='corrupt'):
        pipeline.stage_fetch_data(a, p)
    assert not p.zip.exists() and not p.zip.with_name('dataset.zip.complete').exists()
    assert not (p.raw / 'extracted.partial').exists()


def test_frame_folders_with_missing_frames_are_extracted_again(tmp_path):
    from data_contract import sample_indices
    p = pipeline.Paths(tmp_path, 'r')
    p.shards.mkdir(parents=True)
    meta = dict(sample_fps=1.0, source_fps=25.0, frame_count=1125)
    inv = {}
    for v in ('video01', 'video02'):
        folder = p.frames / f'CHOLEC80__{v}'
        folder.mkdir(parents=True)
        (p.shards / f'CHOLEC80__{v}.json').write_text(json.dumps({f'CHOLEC80__{v}': meta}))
        inv[v] = dict(video=str(tmp_path / 'gone' / f'{v}.mp4'), phase='x', tool='x')
    for i in sample_indices(1125, 25.0, 1.0):                         # video01 complete, video02 empty
        (p.frames / 'CHOLEC80__video01' / f'frame_{int(i):07d}.jpg').write_bytes(b'j')
    assert pipeline.missing_frames(meta, p.frames / 'CHOLEC80__video01') == 0
    assert pipeline.missing_frames(meta, p.frames / 'CHOLEC80__video02') == 45
    pipeline.write_json(p.inventory, dict(base=str(tmp_path), videos=inv))
    a = argparse.Namespace(sample_fps=1.0, workers=1, cleanup_raw=False, data_dir=None, data_zip=None)
    with pytest.raises(pipeline.StageError, match=r"raw videos.*video02"):
        pipeline.stage_extract_frames(a, p)                           # video02 must be redone (video01 is fine)


# ----------------------------------------------------------------- review round 5
def test_an_archive_python_cannot_unpack_is_kept_and_explained(tmp_path):
    import io
    import struct
    import zipfile
    p = pipeline.Paths(tmp_path, 'r')
    p.raw.mkdir(parents=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_STORED) as z:
        z.writestr('cholec80/videos/video01.mp4', b'v' * 1000)
    data = bytearray(buf.getvalue())
    for sig, off in ((b'PK\x03\x04', 8), (b'PK\x01\x02', 10)):    # relabel as Deflate64 (method 9)
        i = data.find(sig)
        data[i + off:i + off + 2] = struct.pack('<H', 9)
    p.zip.write_bytes(bytes(data))
    pipeline.write_json(p.zip.with_name('dataset.zip.complete'), dict(size=len(data)))
    a = argparse.Namespace(data_url='https://x.invalid/d.zip', data_zip=None, data_dir=None, synthetic=None,
                           offline=False, skip_zip_check=False, keep_zip=False, max_videos=None)
    with pytest.raises(pipeline.StageError, match='compression method'):
        pipeline.stage_fetch_data(a, p)
    assert list(p.raw.glob('dataset.zip.rejected-*'))                 # a valid 70 GB download is kept aside


def test_disk_check_counts_a_partial_download_and_local_inputs(tmp_path):
    p = pipeline.Paths(tmp_path, 'r')
    p.raw.mkdir(parents=True)
    a = argparse.Namespace(min_free_gb=200, synthetic=None, data_dir=None, data_zip=None)
    with open(p.zip, 'wb') as f:                                      # 60 GB partial download (sparse file)
        f.truncate(60 * 1024 ** 3)
    need, _ = pipeline.disk_needed_gb(a, p, list(pipeline.STAGES))
    assert 139 < need < 141
    p.zip.unlink()
    videos = tmp_path / 'local'
    videos.mkdir()
    (videos / 'video01.mp4').write_bytes(b'x' * 1024)
    a.data_dir = str(videos)
    need, what = pipeline.disk_needed_gb(a, p, list(pipeline.STAGES))
    assert what == 'frame extraction' and need < 11                   # --data-dir downloads nothing


def test_breaking_a_stale_lease_never_removes_a_fresh_one(tmp_path, monkeypatch, lease_mode):
    lock = tmp_path / 'prepare.lock'
    lease = tmp_path / 'prepare.lock.lease'
    lease.write_text(json.dumps(dict(pid=1, token='old')))
    os.utime(lease, (1, 1))                                           # abandoned long ago
    me, other = pipeline.FileLock(lock), pipeline.FileLock(lock)
    real_age, calls = pipeline.FileLock._lease_age, []
    def racing_age(self, path=None):
        if not calls:                                                 # between our check and our rename,
            calls.append(1)                                           # another job breaks it and takes it
            lease.unlink()
            assert other.acquire(blocking=False)
            return 10_000.0
        return real_age(self, path)
    monkeypatch.setattr(pipeline.FileLock, '_lease_age', racing_age)
    assert me.acquire(blocking=False) is False                        # we must not take the other job's lock
    assert json.loads(lease.read_text())['token'] == pipeline.FileLock._token(lease) != 'old'
    assert other.lease is not None
    other.release()


def test_busy_runs_handles_dots_in_run_names(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.FileLock, 'mode', 'fcntl')
    p = pipeline.Paths(tmp_path, 'x')
    p.state.mkdir(parents=True)
    (p.state / 'run.exp.lock.v2.lock').write_text('')
    monkeypatch.setattr(pipeline.FileLock, 'busy', staticmethod(lambda path: Path(path).name == 'run.exp.lock.v2.lock'))
    assert pipeline.busy_runs(p) == ['exp.lock.v2']


def test_heartbeat_survives_a_lease_that_is_briefly_missing(tmp_path, lease_mode):
    import time
    holder = pipeline.FileLock(tmp_path / 'l')
    assert holder.acquire(blocking=False)
    lease = tmp_path / 'l.lease'
    moved = tmp_path / 'moved'
    os.rename(lease, moved)                                           # another job checking it, for a moment
    time.sleep(0.3)                                                   # heartbeat fires while it is absent
    os.link(moved, lease)
    moved.unlink()
    os.utime(lease, (1, 1))
    time.sleep(0.3)
    assert pipeline.FileLock.busy(tmp_path / 'l')                     # refreshed again: still alive
    holder.release()


def test_a_corrected_link_is_downloaded_after_an_unpackable_archive(tmp_path):
    import io
    import struct
    import zipfile
    p = pipeline.Paths(tmp_path, 'r')
    p.raw.mkdir(parents=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_STORED) as z:
        z.writestr('cholec80/videos/video01.mp4', b'v' * 1000)
    data = bytearray(buf.getvalue())
    for sig, off in ((b'PK\x03\x04', 8), (b'PK\x01\x02', 10)):
        i = data.find(sig)
        data[i + off:i + off + 2] = struct.pack('<H', 9)
    p.zip.write_bytes(bytes(data))
    pipeline.write_json(p.zip.with_name('dataset.zip.complete'), dict(size=len(data)))
    a = argparse.Namespace(data_url='https://x.invalid/d.zip', data_zip=None, data_dir=None, synthetic=None,
                           offline=True, skip_zip_check=False, keep_zip=False, max_videos=None)
    with pytest.raises(pipeline.StageError, match='kept as'):
        pipeline.stage_fetch_data(a, p)
    assert list(p.raw.glob('dataset.zip.rejected-*'))                 # kept for manual unzipping
    with pytest.raises(pipeline.StageError, match='--offline'):      # ... and a fresh download is required next
        pipeline.stage_fetch_data(a, p)
