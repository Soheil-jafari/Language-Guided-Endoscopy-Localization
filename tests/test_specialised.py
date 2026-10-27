"""The two comparison models without a text input (paper protocol): our backbone without language (closed set)
and the TeCNO re-implementation - training, exact resume after a hard kill, prediction, and their dispatch
from main.py. Synthetic data, CPU, random initial weights (no network)."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

import main as pipeline
import paper_protocol as protocol
from data_contract import PHRASES
from paper_fixtures import make_prepared

REPO = Path(pipeline.__file__).resolve().parent
CLOSEDSET = REPO / 'comparison_models' / 'closedset' / 'closedset.py'
TECNO = REPO / 'comparison_models' / 'tecno' / 'tecno.py'
sys.path.insert(0, str(REPO / 'comparison_models' / 'tecno'))
import tecno  # noqa: E402


def prepared(tmp_path, seconds=24):
    from comparison_models import frame_scores as fs
    d = make_prepared(tmp_path / 'data_root', n_videos=6, seconds=seconds, size=(48, 32))
    work = tmp_path / 'work'
    orders = {}
    for split in ('train', 'val', 'test'):
        fs.write_work_order(d['triplets'](split), d['metadata'], d['frames'], d['annotations'],
                            work / f'{split}.order.json', 1.0, labels=work / f'{split}.labels.npz')
        orders[split] = (work / f'{split}.order.json', work / f'{split}.labels.npz')
    return d, orders


def env_for(tmp_path):
    # tests/e2e_hook: a 1-block backbone (checkpoints of the real 12-block one are 1.5 GB with the optimiser)
    return dict({k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}, LGEL_E2E_TINY='1',
                PYTHONPATH=os.pathsep.join([str(REPO / 'tests' / 'e2e_hook'), str(REPO)]),
                TORCH_HOME=str(tmp_path / 'torch_cache'))


def run(cmd, env, cwd):
    out = subprocess.run([sys.executable, *map(str, cmd)], env=env, cwd=cwd, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    return out.stdout


def kill_after_first_epoch(cmd, env, cwd, journal):
    """Start a training and kill it hard (SIGKILL, as a node failure) once its first epoch is logged."""
    proc = subprocess.Popen([sys.executable, *map(str, cmd)], env=env, cwd=cwd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    deadline = time.time() + 600
    while time.time() < deadline and proc.poll() is None:
        if journal.exists() and journal.read_text().count('\n') >= 1:
            os.killpg(proc.pid, signal.SIGKILL)
            break
        time.sleep(0.02)
    proc.wait()
    return proc.returncode


def same_state(a, b):
    sa = torch.load(a, map_location='cpu', weights_only=False)['model']
    sb = torch.load(b, map_location='cpu', weights_only=False)['model']
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def score_matrix(path):
    z = np.load(path, allow_pickle=False)
    return z, [str(q) for q in z['queries']]


# --------------------------------------------------------------------------- closed set
def closedset_config(path, **train):
    cfg = dict(MODEL=dict(M2CRL_WEIGHTS_PATH='', HEAD_NUM_LAYERS=1),
               DATA=dict(TRAIN_CROP_SIZE=32, CLIP_LENGTH=4, NUM_FRAMES=4, NUM_WORKERS=0),
               TRAIN=dict(NUM_EPOCHS=2, WARMUP_EPOCHS=1, BATCH_SIZE=2, GRADIENT_ACCUMULATION_STEPS=2, SEED=42,
                          SUBSET_RATIO=1.0, **train))
    path.write_text(json.dumps(cfg))
    return path


def test_closed_set_model_trains_resumes_exactly_after_a_kill_and_predicts(tmp_path):
    d, orders = prepared(tmp_path)
    env = env_for(tmp_path)
    cfg = closedset_config(tmp_path / 'cfg.json')
    train = lambda out: [CLOSEDSET, 'train', '--config', cfg, '--train-order', orders['train'][0], '--train-labels',
                         orders['train'][1], '--val-order', orders['val'][0], '--val-labels', orders['val'][1],
                         '--output', out, '--workers', 0, '--parameter-counts', tmp_path / f'{out.name}_counts.json']
    straight = tmp_path / 'straight'
    run(train(straight), env, tmp_path)
    rows = [json.loads(x) for x in (straight / 'training_metrics.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in rows] == [1, 2] and all(np.isfinite(r['val_frame_nll']) for r in rows)
    counts = json.loads((tmp_path / 'straight_counts.json').read_text())
    assert set(counts) == {'vision_backbone', 'frame_head', 'temporal_head'}
    # killed during epoch 2, then resumed: exactly the same model as without the interruption
    resumed = tmp_path / 'resumed'
    kill_after_first_epoch(train(resumed), env, tmp_path, resumed / 'training_metrics.jsonl')
    assert torch.load(resumed / 'latest.pt', map_location='cpu', weights_only=False)['epoch'] == 1
    assert 'resuming after epoch 1' in run(train(resumed), env, tmp_path)
    assert same_state(straight / 'latest.pt', resumed / 'latest.pt')
    assert same_state(straight / 'best.pt', resumed / 'best.pt')
    assert [json.loads(x)['epoch'] for x in (resumed / 'training_metrics.jsonl').read_text().splitlines()] == [1, 2]
    # other settings: refused, never mixed
    other = closedset_config(tmp_path / 'other.json', LEARNING_RATE=1e-3)
    bad = train(resumed)
    bad[bad.index('--config') + 1] = other
    out = subprocess.run([sys.executable, *map(str, bad)], env=env, cwd=tmp_path, capture_output=True, text=True)
    assert out.returncode != 0 and 'different settings' in out.stderr
    # prediction: every frame of the work order, one column per query (its concept's output)
    run([CLOSEDSET, 'predict', '--checkpoint', straight / 'best.pt', '--work-order', orders['test'][0], '--output',
         tmp_path / 'test.npz', '--workers', 0], env, tmp_path)
    z, queries = score_matrix(tmp_path / 'test.npz')
    order = json.loads(orders['test'][0].read_text())
    assert queries == [q['text'] for q in order['queries']] and set(queries) == set(PHRASES)
    assert z['scores'].shape == (sum(len(v['frame_idx']) for v in order['videos'].values()), 14)
    assert ((z['scores'] >= 0) & (z['scores'] <= 1)).all()
    manifest = json.loads((tmp_path / 'test.npz.manifest.json').read_text())
    assert manifest['frames'] == len(z['frame_idx']) and manifest['parameters'] > 0
    from comparison_models import frame_scores as fs
    df = fs.assemble(tmp_path / 'test.npz', d['triplets']('test'), d['metadata'], d['annotations'], tmp_path / 'p.csv')
    assert len(df) == 14 * len(z['frame_idx'])


def test_closed_set_answers_a_query_with_its_concepts_output():
    sys.path.insert(0, str(CLOSEDSET.parent))
    import closedset
    assert [closedset.column('phase', i) for i in range(7)] == list(range(7))
    assert [closedset.column('tool', i) for i in range(7)] == list(range(7, 14))


# --------------------------------------------------------------------------- TeCNO
def imagenet_stand_in(tmp_path):
    """Random ResNet-50 weights under the file name torchvision caches the ImageNet weights as."""
    import torchvision
    target = tmp_path / 'torch_cache' / 'hub' / 'checkpoints' / pipeline.RESNET50_IMAGENET
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.save(torchvision.models.resnet50().state_dict(), target)
    assert Path(torchvision.models.ResNet50_Weights.IMAGENET1K_V1.url).name == target.name


def test_tecno_trains_both_stages_resumes_exactly_and_predicts(tmp_path):
    d, orders = prepared(tmp_path)
    imagenet_stand_in(tmp_path)
    env = env_for(tmp_path)
    common = ['--train-order', orders['train'][0], '--train-labels', orders['train'][1], '--val-order', orders['val'][0],
              '--val-labels', orders['val'][1], '--seed', 42]
    cnn = lambda out: [TECNO, 'cnn', *common, '--output', out, '--epochs', 3, '--min-epochs', 3, '--lr', 5e-4,
                       '--batch-size', 8, '--input-size', 32, '--workers', 0]
    straight = tmp_path / 'cnn'
    run(cnn(straight), env, tmp_path)
    resumed = tmp_path / 'cnn_resumed'
    kill_after_first_epoch(cnn(resumed), env, tmp_path, resumed / 'training_metrics.jsonl')
    assert 'resuming stage 1 after epoch 1' in run(cnn(resumed), env, tmp_path)
    assert same_state(straight / 'latest.pt', resumed / 'latest.pt') and same_state(straight / 'best.pt', resumed / 'best.pt')
    assert 'already finished' in run(cnn(straight), env, tmp_path)               # stopped: never trained further
    # features of every video (train, validation, test), extracted once
    feats = tmp_path / 'features'
    extract = [TECNO, 'extract', '--checkpoint', straight / 'best.pt', '--work-order', *(orders[s][0] for s in
               ('train', 'val', 'test')), '--output', feats, '--workers', 0]
    run(extract, env, tmp_path)
    stamps = {f: f.stat().st_mtime_ns for f in feats.glob('*.npz')}
    assert len(stamps) == 6
    run(extract, env, tmp_path)
    assert {f: f.stat().st_mtime_ns for f in feats.glob('*.npz')} == stamps
    z = np.load(sorted(feats.glob('*.npz'))[0])
    assert z['features'].shape[1] == 2048 and z['tool_logits'].shape[1] == 7
    tcn = [TECNO, 'tcn', *common, '--features', feats, '--output', tmp_path / 'tcn', '--epochs', 3, '--min-epochs', 1,
           '--lr', 7e-5, '--parameter-counts', tmp_path / 'counts.json']
    run(tcn, env, tmp_path)
    assert set(json.loads((tmp_path / 'counts.json').read_text())) == {'resnet50_two_heads', 'mstcn'}
    run([TECNO, 'predict', '--checkpoint', tmp_path / 'tcn' / 'best.pt', '--features', feats, '--work-order',
         orders['test'][0], '--output', tmp_path / 'test.npz'], env, tmp_path)
    z, queries = score_matrix(tmp_path / 'test.npz')
    phase_cols = [queries.index(q) for q in PHRASES[:7]]
    np.testing.assert_allclose(z['scores'][:, phase_cols].sum(1), 1, atol=1e-5)   # one softmax over the 7 phases
    order = json.loads(orders['test'][0].read_text())
    video = next(iter(order['videos']))
    stage1 = np.load(feats / f'{video}.npz')
    n = len(stage1['frame_idx'])
    np.testing.assert_allclose(z['scores'][:n, queries.index(PHRASES[7])], 1 / (1 + np.exp(-stage1['tool_logits'][:, 0])),
                               rtol=1e-5)
    assert json.loads((tmp_path / 'test.npz.manifest.json').read_text())['frames_per_second'] > 0
    # features of another stage-1 network are refused
    other = tmp_path / 'other_best.pt'
    ck = torch.load(straight / 'best.pt', map_location='cpu', weights_only=False)
    ck['epoch'] = 99
    torch.save(ck, other)
    with pytest.raises(ValueError, match='different stage-1'):
        tecno.load_features(feats, video, stage1['frame_idx'], tecno.file_sha(other))


def test_tecno_temporal_network_is_causal():
    torch.manual_seed(0)
    model = tecno.MultiStageTCN(**tecno.TCN).eval()
    x = torch.randn(1, 2048, 60)
    y = model(x)
    assert y.shape == (2, 1, 7, 60)
    x2 = x.clone()
    x2[..., 40:] = torch.randn(1, 2048, 20)
    y2 = model(x2)
    assert torch.allclose(y[..., :40], y2[..., :40]) and not torch.allclose(y[..., 40:], y2[..., 40:])


def test_tecno_early_stopping_follows_pytorch_lightning():
    stop = tecno.EarlyStopping(patience=3, min_epochs=6, max_epochs=15)
    values = [0.5, 0.6, 0.6, 0.55, 0.58, 0.59, 0.7]
    for epoch, v in enumerate(values, 1):
        stop.update(epoch, v)
        if stop.stopped:
            break
    assert epoch == 6 and stop.best_epoch == 2 and stop.best == 0.6      # patience reached at 5, min epochs at 6
    stop = tecno.EarlyStopping(3, 6, 15)
    for epoch in range(1, 16):
        stop.update(epoch, epoch / 100)
    assert stop.stopped and stop.best_epoch == 15
    state = tecno.EarlyStopping(3, 6, 15, stop.state())
    assert state.state() == stop.state()


def test_tecno_phase_targets_and_class_weights():
    labels = np.full((3, 14), -100)
    labels[0, :7] = [0, 0, 1, 0, 0, 0, 0]
    labels[1, :7] = [1, 0, 0, 0, 0, 0, 0]
    assert tecno.phase_targets(labels).tolist() == [2, 0, -100]
    labels[2, :7] = [0, 0, 0, -100, -100, -100, -100]
    with pytest.raises(ValueError):
        tecno.phase_targets(labels)
    assert len(tecno.CLASS_WEIGHTS) == 7 and tecno.CLASS_WEIGHTS[1] == pytest.approx(0.19090963801041133)


# --------------------------------------------------------------------------- dispatch from main.py
def _plans(tmp_path):
    a = pipeline.resolve_settings(pipeline.build_parser().parse_args(['--preset', 'paper', '--root', str(tmp_path),
                                                                      '--single-job']))
    plans = pipeline.experiment_plans(a)
    out = {s.kind: s for s in plans if s.kind != 'proposed'}
    out['baseline'] = next(s for s in plans if s.run_name == 'paper_baseline_s42')
    return out


def _record(monkeypatch):
    launched = []
    monkeypatch.setattr(pipeline, 'run', lambda cmd, a, p, name, **k: launched.append((name, [str(c) for c in cmd])))
    monkeypatch.setattr(pipeline, 'check_run_inputs', lambda *x: None)
    return launched


def test_closed_set_uses_our_training_configuration_with_its_own_batch(tmp_path, monkeypatch):
    subs = _plans(tmp_path)
    sub = subs['closedset']
    assert not sub.paraphrase_eval and sub.flags == {}
    p = pipeline.Paths(tmp_path, sub.run_name)
    launched = _record(monkeypatch)
    monkeypatch.setattr(pipeline, 'read_weights_path', lambda p, a: '/weights/m2crl.pth')
    (p.run / 'closedset').mkdir(parents=True)
    (p.run / 'closedset' / 'best.pt').write_bytes(b'x')
    pipeline.stage_train(sub, p)
    assert [n for n, _ in launched] == ['workorder_train', 'workorder_val_labels', 'train']
    cmd = launched[-1][1]
    assert cmd[1].endswith('closedset.py') and cmd[cmd.index('--parameter-counts') + 1] == str(p.run / 'parameter_counts.json')
    cfg = json.loads((p.run / 'closedset_config.json').read_text())
    ours = pipeline.train_config(subs['baseline'], p, cfg['DATA']['NUM_WORKERS'], cfg['TRAIN']['AMP_DTYPE'])
    assert cfg['TRAIN'].pop('BATCH_SIZE') == 8 and cfg['TRAIN'].pop('GRADIENT_ACCUMULATION_STEPS') == 2
    assert ours['TRAIN'].pop('BATCH_SIZE') == 8 and ours['TRAIN'].pop('GRADIENT_ACCUMULATION_STEPS') == 24
    assert cfg == ours                                       # everything else exactly as our baseline's training
    assert cfg['MODEL']['M2CRL_WEIGHTS_PATH'] == '/weights/m2crl.pth' and cfg['TRAIN']['NUM_EPOCHS'] == 10
    assert pipeline.finished_marker(p.run / 'closedset').exists()
    launched.clear()
    pipeline.stage_train(sub, p)
    assert launched == []                                    # finished: nothing again
    # predictions on the validation and test videos only (no text input: no unseen wordings)
    monkeypatch.setattr(pipeline.shutil, 'copy2', lambda *x: None)
    pipeline.COMPARISON_METHODS['closedset']['predict'](sub, p, pipeline.prediction_sets(sub, p))
    assert [n for n, _ in launched if n.startswith('closedset_')] == ['closedset_val', 'closedset_test']
    # a resumable run started with other settings is never continued with new ones
    pipeline.finished_marker(p.run / 'closedset').unlink()
    (p.run / 'closedset' / 'latest.pt').write_bytes(b'x')
    sub.epochs = 3
    with pytest.raises(pipeline.StageError, match='different settings'):
        pipeline.stage_train(sub, p)


def test_tecno_is_dispatched_with_the_authors_settings(tmp_path, monkeypatch):
    sub = _plans(tmp_path)['tecno']
    assert not sub.paraphrase_eval
    p = pipeline.Paths(tmp_path, sub.run_name)
    launched = _record(monkeypatch)
    for stage in ('tecno_cnn', 'tecno_tcn'):
        (p.run / stage).mkdir(parents=True)
        (p.run / stage / 'best.pt').write_bytes(b'x')
    sub.offline = True
    with pytest.raises(pipeline.StageError, match='ImageNet ResNet-50'):
        pipeline.stage_train(sub, p)
    sub.offline = False
    launched.clear()
    pipeline.stage_train(sub, p)
    assert [n for n, _ in launched] == ['workorder_train', 'workorder_val_labels', 'train_cnn', 'workorder_test',
                                        'tecno_features', 'train_tcn']
    args = {n: c for n, c in launched}
    get = lambda name, flag: args[name][args[name].index(flag) + 1]
    assert (get('train_cnn', '--epochs'), get('train_cnn', '--min-epochs'), get('train_cnn', '--batch-size'),
            float(get('train_cnn', '--lr')), get('train_cnn', '--input-size'), get('train_cnn', '--patience')) == \
        ('15', '6', '80', 5e-4, '224', '3')
    assert (get('train_tcn', '--epochs'), get('train_tcn', '--min-epochs'), float(get('train_tcn', '--lr'))) == \
        ('16', '12', 7e-5)
    assert get('train_tcn', '--parameter-counts') == str(p.run / 'parameter_counts.json')
    feats = args['tecno_features']
    assert sum(c.endswith('.order.json') for c in feats) == 3          # train, validation and test videos
    assert pipeline.finished_marker(p.run / 'tecno_tcn').exists() and pipeline.finished_marker(p.run / 'tecno_cnn').exists()
    launched.clear()
    pipeline.stage_train(sub, p)
    assert launched == []
    monkeypatch.setattr(pipeline.shutil, 'copy2', lambda *x: None)
    pipeline.COMPARISON_METHODS['tecno']['predict'](sub, p, pipeline.prediction_sets(sub, p))
    assert [n for n, _ in launched if n.startswith('tecno_')] == ['tecno_val', 'tecno_test']
    # once stage 1 has finished it is not repeated (only extraction, which skips finished videos, and stage 2)
    pipeline.finished_marker(p.run / 'tecno_tcn').unlink()
    launched.clear()
    pipeline.stage_train(sub, p)
    assert 'train_cnn' not in [n for n, _ in launched] and 'train_tcn' in [n for n, _ in launched]


def test_the_protocol_settings_of_the_specialised_models():
    t = protocol.COMPARISONS['tecno']
    assert (t['cnn_epochs'], t['cnn_min_epochs'], t['cnn_lr'], t['cnn_batch_size'], t['tcn_epochs'],
            t['tcn_min_epochs'], t['tcn_lr'], t['patience']) == (15, 6, 5e-4, 80, 16, 12, 7e-5, 3)
    assert tecno.TCN == dict(stages=2, layers=8, f_maps=32, dim=2048, classes=7)
    c = protocol.COMPARISONS['closedset']
    assert c['batch_size'] * c['accum'] == 16
