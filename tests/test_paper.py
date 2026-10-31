"""The paper protocol: run plan, unseen wordings, comparison-method plumbing, the analysis' metrics and the
job driver's failure isolation. Everything runs offline on synthetic data and tiny models."""
import argparse
import json
import math
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import main as pipeline
import paper_protocol as protocol
import paper_analysis as pa
from data_contract import PHRASES, query_spec
from paper_fixtures import make_prepared, tiny_xclip

REPO = Path(pipeline.__file__).resolve().parent


# --------------------------------------------------------------------------- protocol
def test_the_protocol_lists_every_run_once_in_priority_order():
    plans = protocol.plans('paper')
    names = [p['name'] for p in plans]
    assert len(names) == len(set(names)) == 20
    assert names[:2] == ['paper_baseline_s42', 'paper_advanced_s42']           # the main comparison first
    assert names[-5:-1] == ['paper_baseline_s43', 'paper_advanced_s43', 'paper_baseline_s44', 'paper_advanced_s44']
    assert names[-1] == 'paper_closedset_matched_s42'                           # the longest comparison runs last
    assert {p['kind'] for p in plans} - {'proposed'} == set(pipeline.COMPARISON_METHODS) == set(protocol.COMPARISONS)
    assert protocol.ADVANCED == pipeline.ADVANCED_FLAGS                          # one definition of "advanced"
    ablations = [p for p in plans if p['group'] == 'ablation']
    assert len(ablations) == 4 and not any(p['paraphrases'] for p in ablations)
    for p, (label, flags) in zip(ablations, protocol.ADDONS.values()):
        assert p['flags'] == flags and protocol.describe_flags(p['flags']) == f'baseline + {label}'
    assert protocol.describe_flags({}) == 'baseline (Transformer temporal head)'
    assert protocol.describe_flags(protocol.ADVANCED).startswith('advanced (')
    seeds = sorted((p['seed'], protocol.describe_flags(p['flags'])[:8]) for p in plans
                   if p['kind'] == 'proposed' and p['group'] in ('main', 'seed'))
    assert [s for s, _ in seeds] == [42, 42, 43, 43, 44, 44]
    assert all(p['seed'] == protocol.MAIN_SEED for p in plans if p['kind'] != 'proposed')
    comparisons = [p for p in plans if p['group'] == 'comparison']
    assert [p['kind'] for p in comparisons] == list(protocol.ZERO_SHOT + protocol.TRAINED)
    # models without a text input are not asked in other words; every other method is
    assert {p['kind'] for p in comparisons if not p['paraphrases']} == set(protocol.CLOSED_SET) == \
        {'tecno', 'closedset', 'closedset_matched'}
    assert {protocol.COMPARISONS[k]['model'] for k in protocol.SURGVLP_ZERO_SHOT} == set(protocol.SURGVLP['weights_urls'])
    assert [p['name'] for p in protocol.plans('x')][0] == 'x_baseline_s42'


def test_standard_split_file_is_40_8_32_and_disjoint():
    split = json.loads(protocol.SPLIT_FILE.read_text())
    assert [len(split[k]) for k in ('train', 'val', 'test')] == [40, 8, 32]
    assert split['train'][0] == 'CHOLEC80__video01' and split['val'][0] == 'CHOLEC80__video41'
    assert split['test'] == [f'CHOLEC80__video{i:02d}' for i in range(49, 81)]
    allv = [v for vs in split.values() for v in vs]
    assert len(set(allv)) == 80


def test_paraphrases_cover_every_concept_and_never_reuse_training_wordings(tmp_path):
    sets = protocol.load_paraphrases()
    assert set(sets) == {'A', 'B'} and all(len(m) == 14 for m in sets.values())
    assert protocol.canonical_queries() == PHRASES and len(protocol.phase_queries()) == 7
    for words, prompt in zip(('preparation', 'calot', 'clip', 'dissection', 'packaging', 'coagulation', 'retraction'),
                             protocol.PESKAVLP_PHASE_PROMPTS):
        assert words in prompt.lower()
    data = json.loads(protocol.PARAPHRASE_FILE.read_text())
    # how the wordings were made is recorded with them; the tool labels mean "visible", so no tool wording
    # claims that the tool is being used
    assert len(data['sources']) >= 3 and 'not reviewed by a clinician' in data['_comment']
    assert not any('being used' in q['text'] or 'is used' in q['text']
                   for s in data['sets'].values() for q in s['queries'] if q['concept'] == 'tool')
    bad = json.loads(json.dumps(data))
    bad['sets']['A']['queries'][0]['text'] = PHRASES[0]                          # a training sentence
    (tmp_path / 'bad.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='training sentence'):
        protocol.load_paraphrases(tmp_path / 'bad.json')
    bad = json.loads(json.dumps(data))
    first = bad['sets']['A']['queries'][0]
    assert first['concept'] == 'phase'
    first['text'] = 'grasper'                                   # a known word of another concept (a tool)
    (tmp_path / 'bad.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='contradicts'):
        protocol.load_paraphrases(tmp_path / 'bad.json')


def _args(argv):
    a = pipeline.build_parser().parse_args(argv)
    return pipeline.resolve_settings(a)


def test_paper_preset_settings_and_runs(tmp_path):
    a = _args(['--preset', 'paper', '--root', str(tmp_path)])
    assert a.splits_json == str(protocol.SPLIT_FILE) and a.paraphrases and a.epochs == 10
    subs = pipeline.experiment_plans(a)
    assert [s.run_name for s in subs] == [p['name'] for p in protocol.plans('paper')]
    by = {s.run_name: s for s in subs}
    assert by['paper_advanced_s42'].flags == pipeline.ADVANCED_FLAGS and by['paper_advanced_s42'].variant == 'advanced'
    assert by['paper_baseline_s43'].seed == 43 and by['paper_baseline_s43'].flags == {}
    assert by['paper_ablation_mamba_s42'].variant == 'ablation' and not by['paper_ablation_mamba_s42'].paraphrase_eval
    assert by['paper_univtg_s42'].kind == 'univtg' and by['paper_univtg_s42'].paraphrase_eval
    cfg = pipeline.train_config(by['paper_ablation_bilevel_s42'], pipeline.Paths(tmp_path, 'x'), 4, 'bf16')
    assert cfg['TRAIN']['USE_BILEVEL_CONSISTENCY'] is True and 'TEMPORAL_HEAD_TYPE' not in cfg['MODEL']
    assert cfg['TRAIN']['SEED'] == 42 and cfg['TRAIN']['NUM_EPOCHS'] == 10
    assert pipeline.uses_flow(by['paper_advanced_s42']) and not pipeline.uses_flow(by['paper_baseline_s42'])
    # the data stages are the same for every run (and include the unseen-wording manifests)
    sigs = {json.dumps(pipeline.stage_signature('triplets', s), sort_keys=True) for s in subs}
    assert len(sigs) == 1 and 'paraphrases' in sigs.pop()
    with pytest.raises(SystemExit, match='variant'):
        _args(['--preset', 'paper', '--root', str(tmp_path), '--variant', 'baseline'])
    # other presets are unchanged
    full = pipeline.experiment_plans(_args(['--preset', 'full', '--root', str(tmp_path)]))
    assert [(s.variant, s.run_name, s.kind) for s in full] == [('baseline', 'full_baseline_seed42', 'proposed'),
                                                              ('advanced', 'full_advanced_seed42', 'proposed')]
    assert full[1].flags == pipeline.ADVANCED_FLAGS and full[0].flags == {}


def test_paper_dry_run_lists_the_protocol(tmp_path, capsys):
    assert pipeline.main(['--preset', 'paper', '--root', str(tmp_path / 'r'), '--dry-run']) == 0
    out = capsys.readouterr().out
    assert 'paper protocol: 20 runs' in out and 'paper_peskavlp_finetune_s42' in out and 'unseen wordings' in out
    assert 'paper_tecno_s42' in out and 'paper_closedset_s42' in out and 'paper_hecvl_zeroshot' in out
    assert not (tmp_path / 'r').exists()


# --------------------------------------------------------------------------- the job driver
def _fake_drive(fail):
    calls = []

    def drive(a, p, stages, redo):
        calls.append(a.run_name)
        a.failed_stage = fail.get(a.run_name)
        return 1 if a.run_name in fail else 0
    return drive, calls


@pytest.mark.parametrize('stage,continues', [('train', True), ('predict', True), ('evaluate', True), ('summary', True),
                                             ('duplicate', True), ('fetch_data', False), ('preflight', False),
                                             ('check_advanced', False), ('locks', False), ('data', False)])
def test_a_failed_run_stops_the_protocol_only_if_others_are_affected(tmp_path, monkeypatch, stage, continues):
    drive, calls = _fake_drive({'paper_clip_zeroshot': stage})
    monkeypatch.setattr(pipeline, '_drive', drive)
    monkeypatch.setattr(pipeline, 'write_report_bundle', lambda *a, **k: None)
    outputs = []
    monkeypatch.setattr(pipeline, 'paper_outputs', lambda a, plans: outputs.append(len(plans)) or 0)
    monkeypatch.setattr(pipeline, 'data_prepared', lambda p: True)
    rc = pipeline.main(['--preset', 'paper', '--root', str(tmp_path), '--single-job'])
    assert rc == 1
    names = [p['name'] for p in protocol.plans('paper')]
    if continues:
        assert calls == names and outputs == [20]
    else:
        assert calls == names[:names.index('paper_clip_zeroshot') + 1]


def test_the_paper_analysis_runs_after_every_complete_job(tmp_path, monkeypatch):
    drive, calls = _fake_drive({})
    monkeypatch.setattr(pipeline, '_drive', drive)
    monkeypatch.setattr(pipeline, 'write_report_bundle', lambda *a, **k: None)
    launched = []
    monkeypatch.setattr(pipeline, 'run', lambda cmd, a, p, name, **k: launched.append((name, [str(c) for c in cmd])))
    monkeypatch.setattr(pipeline, 'data_prepared', lambda p: True)
    assert pipeline.main(['--preset', 'paper', '--root', str(tmp_path), '--single-job']) == 0
    assert [n for n, _ in launched] == ['paper_analysis']            # no finished model yet: no explanation maps
    cmd = launched[0][1]
    assert cmd[1] == 'paper_analysis.py' and cmd[cmd.index('--prefix') + 1] == 'paper'
    # an analysis failure is reported (exit code 1) but does not touch the runs
    def failing(cmd, a, p, name, **k):
        raise pipeline.StageError('boom')
    monkeypatch.setattr(pipeline, 'run', failing)
    assert pipeline.main(['--preset', 'paper', '--root', str(tmp_path), '--single-job']) == 1
    # a partial command (--to-stage) does not run it
    monkeypatch.setattr(pipeline, 'run', lambda *x, **k: launched.append('again'))
    assert pipeline.main(['--preset', 'paper', '--root', str(tmp_path), '--single-job', '--to-stage', 'audit']) == 0
    assert 'again' not in launched


def test_failed_stage_is_recorded_by_the_real_driver(tmp_path, monkeypatch):
    a = _args(['--preset', 'paper', '--root', str(tmp_path), '--single-job', '--allow-cpu'])
    sub = pipeline.experiment_plans(a)[0]
    p = pipeline.Paths(tmp_path, sub.run_name)
    for d in (p.state, p.logs):
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(pipeline, 'stage_preflight', lambda *x: None)
    monkeypatch.setattr(pipeline, 'stage_train', lambda a, p: (_ for _ in ()).throw(pipeline.StageError('no')))
    for s in pipeline.STAGES[1:pipeline.STAGES.index('train')]:
        if s not in pipeline.ALWAYS_RUN:
            pipeline.write_json(p.marker(s), dict(signature=pipeline.stage_signature(s, sub)))
    sub.check_advanced = False
    assert pipeline._drive(sub, p, ['train'], []) == 1 and sub.failed_stage == 'train'
    assert p.log_scope is None


def test_run_stage_logs_go_to_a_folder_per_run(tmp_path):
    a = argparse.Namespace(offline=True)
    p = pipeline.Paths(tmp_path, 'r1')
    p.log_scope = 'r1'
    pipeline.run([sys.executable, '-c', 'print("hello")'], a, p, 'predict_test')
    assert 'hello' in (p.logs / 'r1' / 'predict_test.log').read_text()


# --------------------------------------------------------------------------- shared comparison plumbing
def test_work_orders_and_score_matrices_round_trip_to_the_standard_prediction_files(tmp_path):
    d = make_prepared(tmp_path, n_videos=4, seconds=12)
    from comparison_models import frame_scores as fs
    order = fs.write_work_order(d['triplets']('test'), d['metadata'], d['frames'], d['annotations'],
                                tmp_path / 'o.json', 1.0, labels=tmp_path / 'l.npz',
                                prompts={PHRASES[0]: 'In preparation phase I insert trocars'})
    texts = [q['text'] for q in order['queries']]
    assert len(texts) == 14 and order['queries'][texts.index(PHRASES[0])]['prompt'].startswith('In preparation')
    z = np.load(tmp_path / 'l.npz')
    video = d['split']['test'][0]
    assert z['labels'].shape == (len(order['videos'][video]['frame_idx']), 14)
    assert list(z['canonical']) == list(PHRASES)
    # a scorer's matrix (frames x scored sentences) -> the standard CSV, exactly the manifest's pairs
    sentences = [q['prompt'] for q in order['queries']]
    frame_idx = [i for v in order['videos'].values() for i in v['frame_idx']]
    offsets = np.cumsum([0] + [len(v['frame_idx']) for v in order['videos'].values()])
    scores = np.random.default_rng(0).random((len(frame_idx), len(sentences)))
    np.savez(tmp_path / 's.npz', video_ids=np.asarray(list(order['videos'])), offsets=offsets,
             frame_idx=np.asarray(frame_idx), queries=np.asarray(sentences), scores=scores)
    df = fs.assemble(tmp_path / 's.npz', d['triplets']('test'), d['metadata'], d['annotations'], tmp_path / 'p.csv',
                     1.0, prompts={PHRASES[0]: 'In preparation phase I insert trocars'})
    assert set(df.text_query) == set(PHRASES) and len(df) == 14 * len(frame_idx)
    first = df[(df.text_query == PHRASES[0])].score.to_numpy()
    np.testing.assert_allclose(first, scores[:, sentences.index('In preparation phase I insert trocars')])
    ref = tmp_path / 'ref.csv'
    subprocess.run([sys.executable, 'build_reference.py', '--triplets', d['triplets']('test'), '--metadata',
                    d['metadata'], '--frames', d['frames'], '--annotations', d['annotations'], '--output', ref],
                   cwd=REPO, check=True)
    from evaluate import checked_frames
    merged = checked_frames(tmp_path / 'p.csv', ref)                 # exactly the reference's frames
    ours = df.set_index(['video_id', 'text_query', 'frame_idx']).label
    theirs = merged.set_index(['video_id', 'text_query', 'frame_idx']).label
    assert len(merged) == len(df) and (ours.sort_index() == theirs.sort_index()).all()
    with pytest.raises(FileExistsError):
        fs.assemble(tmp_path / 's.npz', d['triplets']('test'), d['metadata'], d['annotations'], tmp_path / 'p.csv')
    bad = scores.copy(); bad[0, 0] = 1.5
    np.savez(tmp_path / 'bad.npz', video_ids=np.asarray(list(order['videos'])), offsets=offsets,
             frame_idx=np.asarray(frame_idx), queries=np.asarray(sentences), scores=bad)
    with pytest.raises(ValueError, match='probabilities'):
        fs.read_scores(tmp_path / 'bad.npz')


def test_unseen_wording_manifests_keep_frames_and_labels(tmp_path):
    d = make_prepared(tmp_path, n_videos=4, seconds=10)
    written = protocol.write_paraphrase_triplets(d['triplets']('test'), tmp_path)
    test = pd.read_csv(d['triplets']('test'))
    for name, path in written.items():
        para = pd.read_csv(path)
        assert len(para) == len(test) and (para.frame_path == test.frame_path).all()
        assert (para.relevance_label == test.relevance_label).all()
        assert not set(para.text_query) & set(PHRASES)
        from data_contract import validate_triplets
        specs = validate_triplets(para)
        assert specs == validate_triplets(test)                       # same concept on every row


def test_peskavlp_archive_is_unpacked_safely(tmp_path):
    good = tmp_path / 'g.zip'
    with zipfile.ZipFile(good, 'w') as z:
        z.writestr('PeskaVLP.pth', b'w')
        z.writestr('__MACOSX/._PeskaVLP.pth', b'junk')
    a = argparse.Namespace(offline=True)
    p = pipeline.Paths(tmp_path / 'root', 'x')
    target = p.root / 'comparisons' / 'peskavlp' / 'PeskaVLP.download'
    target.parent.mkdir(parents=True)
    target.write_bytes(good.read_bytes())
    pipeline.write_json(target.with_name(target.name + '.complete'), dict(size=target.stat().st_size))
    found = pipeline.peskavlp_weights(a, p)                         # offline: the finished download is reused
    assert found.name == 'PeskaVLP.pth' and found.read_bytes() == b'w'
    for model in ('SurgVLP', 'HecVL'):                                # each model has its own download
        with pytest.raises(pipeline.StageError, match=f'--offline: {model}.download'):
            pipeline.peskavlp_weights(a, p, model)
        target = p.root / 'comparisons' / model.lower() / f'{model}.download'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'plain checkpoint')
        pipeline.write_json(target.with_name(target.name + '.complete'), dict(size=target.stat().st_size))
        assert pipeline.peskavlp_weights(a, p, model) == target
    with pytest.raises(pipeline.StageError, match='unknown'):
        pipeline.peskavlp_weights(a, p, 'CLIP')
    evil = tmp_path / 'e.zip'
    with zipfile.ZipFile(evil, 'w') as z:
        z.writestr('../escape.pth', b'x')
    with pytest.raises(pipeline.StageError, match='unsafe'):
        pipeline.safe_extract(evil, tmp_path / 'out')


def _record_runs(monkeypatch):
    launched = []

    def fake_run(cmd, a, p, name, **k):
        launched.append((name, [str(c) for c in cmd]))
    monkeypatch.setattr(pipeline, 'run', fake_run)
    return launched


def test_comparison_methods_are_dispatched_with_their_own_settings(tmp_path, monkeypatch):
    a = _args(['--preset', 'paper', '--root', str(tmp_path), '--single-job'])
    subs = {s.kind: s for s in pipeline.experiment_plans(a) if s.kind != 'proposed'}
    launched = _record_runs(monkeypatch)
    monkeypatch.setattr(pipeline, 'check_run_inputs', lambda *x: None)
    # X-CLIP: the repository's fine-tuning with the protocol's settings, resumed from latest.pt
    sub = subs['xclip_finetune']
    p = pipeline.Paths(tmp_path, sub.run_name)
    (p.run / 'xclip').mkdir(parents=True)
    (p.run / 'xclip' / 'latest.pt').write_bytes(b'x')
    (p.run / 'xclip' / 'best.pt').write_bytes(b'x')
    pipeline.stage_train(sub, p)
    names = [n for n, _ in launched]
    assert names == ['fetch_xclip', 'train']
    cmd = launched[1][1]
    assert cmd[1] == 'train_xclip.py' and cmd[cmd.index('--epochs') + 1] == '5' and '--resume-from' in cmd
    assert pipeline.finished_marker(p.run / 'xclip').exists()
    launched.clear()
    pipeline.stage_train(sub, p)                                     # finished: only the model check-out
    assert [n for n, _ in launched] == ['fetch_xclip']
    # UniVTG: features (once per backbone), text features (incl. unseen wordings), training
    sub = subs['univtg']
    p = pipeline.Paths(tmp_path, sub.run_name)
    p.weights_ref.parent.mkdir(parents=True, exist_ok=True)
    pipeline.write_json(p.weights_ref, dict(path='', random_init=True))
    sub.random_init = True
    launched.clear()
    pipeline.stage_train(sub, p)
    assert [n for n, _ in launched] == ['univtg_features', 'univtg_text', 'train']
    text_cmd = launched[1][1]
    assert any('test_paraA' in c for c in text_cmd) and any('test_paraB' in c for c in text_cmd)
    feat = launched[0][1]
    assert feat[feat.index('--weights') + 1] == '' and feat[feat.index('--output') + 1].endswith('m2crl_random-init')
    train = launched[2][1]
    assert train[train.index('--epochs') + 1] == '200' and train[train.index('--clip-frames') + 1] == '4'
    # predictions for every evaluation set, with the native windows kept
    launched.clear()
    p.results.mkdir(parents=True, exist_ok=True)
    pipeline.univtg_predict(sub, p, pipeline.prediction_sets(sub, p))
    assert [n for n, _ in launched] == ['predict_val', 'predict_test', 'predict_test_paraA', 'predict_test_paraB']
    assert all('--segments' in c for _, c in launched)


@pytest.mark.parametrize('kind', protocol.SURGVLP_ZERO_SHOT)
def test_surgical_vlm_zero_shot_also_scores_the_authors_phase_prompts(tmp_path, monkeypatch, kind):
    a = _args(['--preset', 'paper', '--root', str(tmp_path), '--single-job'])
    sub = [s for s in pipeline.experiment_plans(a) if s.kind == kind][0]
    model = protocol.COMPARISONS[kind]['model']
    tag = model.lower()
    p = pipeline.Paths(tmp_path, sub.run_name)
    launched = _record_runs(monkeypatch)
    asked = []
    monkeypatch.setattr(pipeline, 'peskavlp_weights', lambda a, p, m='PeskaVLP': asked.append(m) or tmp_path / f'{m}.pth')
    copied = []
    monkeypatch.setattr(pipeline.shutil, 'copy2', lambda src, dst: copied.append(str(dst)))
    p.results.mkdir(parents=True)
    pipeline.COMPARISON_METHODS[kind]['predict'](sub, p, pipeline.prediction_sets(sub, p))
    assert set(asked) == {model}
    scored = [n for n, _ in launched if n.startswith(tag + '_')]
    assert scored == [f'{tag}_val', f'{tag}_test', f'{tag}_test_paraA', f'{tag}_test_paraB',
                      f'{tag}_authorprompts_val', f'{tag}_authorprompts_test']
    for n, c in launched:
        if n.startswith(tag + '_'):
            assert c[c.index('--model') + 1] == model and c[c.index('--weights') + 1].endswith(f'{model}.pth')
            assert '--checkpoint' not in c
    prompts = json.loads((p.run / 'work' / 'authorprompts_test.prompts.json').read_text())
    assert prompts == dict(zip(PHRASES[:7], protocol.PESKAVLP_PHASE_PROMPTS))
    assemble = [c for n, c in launched if n == 'assemble_authorprompts_test'][0]
    assert assemble[assemble.index('--output') + 1].endswith('authorprompts_test.csv')    # not a proposed_* file
    assert any(c.endswith('authorprompts_test.csv.manifest.json') for c in copied)
    # the zero-shot "training": environment and a strict construction with this model's weights
    launched.clear()
    sub.offline = True
    (pipeline.peskavlp_prefix(p) / '.lgel_peskavlp_ready').parent.mkdir(parents=True, exist_ok=True)
    (pipeline.peskavlp_prefix(p) / '.lgel_peskavlp_ready').write_text('ok')
    pipeline.COMPARISON_METHODS[kind]['train'](sub, p)
    assert [n for n, _ in launched] == [f'{tag}_check']
    assert launched[0][1][launched[0][1].index('--model') + 1] == model


def test_peskavlp_fine_tuned_scores_with_its_checkpoint(tmp_path, monkeypatch):
    a = _args(['--preset', 'paper', '--root', str(tmp_path), '--single-job'])
    launched = _record_runs(monkeypatch)
    monkeypatch.setattr(pipeline, 'peskavlp_weights', lambda a, p, m='PeskaVLP': tmp_path / f'{m}.pth')
    monkeypatch.setattr(pipeline.shutil, 'copy2', lambda src, dst: None)
    sub2 = [s for s in pipeline.experiment_plans(a) if s.kind == 'peskavlp_finetune'][0]
    pipeline.COMPARISON_METHODS['peskavlp_finetune']['predict'](sub2, pipeline.Paths(tmp_path, sub2.run_name),
                                                                  [('test', tmp_path / 't.csv')])
    score = [c for n, c in launched if n == 'peskavlp_test'][0]
    assert score[score.index('--checkpoint') + 1].endswith('peskavlp/best.pt')
    assert score[score.index('--model') + 1] == 'PeskaVLP'
    assert not any(n.startswith('peskavlp_authorprompts') for n, _ in launched)


def test_the_peskavlp_environment_never_sees_this_repository(tmp_path):
    a = argparse.Namespace(offline=False)
    p = pipeline.Paths(tmp_path, 'x')
    env = pipeline.peskavlp_env(a, p, online=True)
    assert 'PYTHONPATH' not in env and 'HF_HUB_OFFLINE' not in env and env['HF_HOME'] == str(p.hf)


# --------------------------------------------------------------------------- X-CLIP scoring
def test_xclip_scores_all_queries_of_a_window_at_once_exactly_like_one_at_a_time(tmp_path):
    folder = tiny_xclip(tmp_path / 'xclip')
    from transformers import XCLIPModel, XCLIPProcessor
    model = XCLIPModel.from_pretrained(folder).eval()
    proc = XCLIPProcessor.from_pretrained(folder)
    clip = [np.random.default_rng(i).integers(0, 255, (16, 24, 3), dtype=np.uint8) for i in range(4)]
    texts = list(PHRASES[:3])
    with torch.no_grad():
        together = model(**proc(videos=[clip], text=texts, padding=True, return_tensors='pt')).logits_per_video[0]
        alone = [model(**proc(videos=[clip], text=[t], padding=True, return_tensors='pt')).logits_per_video[0, 0]
                 for t in texts]
    torch.testing.assert_close(together, torch.stack(alone), rtol=1e-4, atol=1e-5)
    d = make_prepared(tmp_path / 'data', n_videos=4, seconds=10)
    out = tmp_path / 'pred.csv'
    subprocess.run([sys.executable, 'benchmark.py', '--model', 'xclip', '--model-name', folder, '--triplets',
                    d['triplets']('test'), '--metadata', d['metadata'], '--frames', d['frames'], '--annotations',
                    d['annotations'], '--output', out, '--device', 'cpu'], cwd=REPO, check=True, capture_output=True)
    df = pd.read_csv(out)
    assert set(df.text_query) == set(PHRASES) and df.score.between(0, 1).all()
    manifest = json.loads(Path(str(out) + '.manifest.json').read_text())
    assert manifest['frame_scores'] == len(df) and manifest['seconds'] >= 0 and manifest['parameters'] > 0


# --------------------------------------------------------------------------- UniVTG adapter
def test_univtg_and_peskavlp_journals_are_cleaned_when_training_resumes(tmp_path):
    from comparison_models.univtg import adapter
    sys.path.insert(0, str(Path(pa.__file__).resolve().parent / 'comparison_models' / 'peskavlp'))
    import peska_runner
    for clean in (adapter.clean_journal, peska_runner.clean_journal):
        j = tmp_path / f'{clean.__module__}.jsonl'
        rows = [json.dumps(dict(epoch=e, train_loss=1.0 / e)) for e in (1, 2)]
        torn = json.dumps(dict(epoch=3, train_loss=0.3))[:12] + json.dumps(dict(epoch=3, train_loss=0.3))
        j.write_text('\n'.join(rows + [torn, '[1, 2]', json.dumps(dict(epoch=4, train_loss=0.2))]) + '\n')
        clean(j, 2)                                 # resumed after epoch 2: epochs 3 and 4 are redone
        assert [json.loads(x)['epoch'] for x in j.read_text().splitlines()] == [1, 2]
        clean(j, 0)                                 # a fresh start keeps nothing
        assert j.read_text() == ''
        clean(tmp_path / 'absent.jsonl', 3)         # no journal yet: nothing to do
        assert not (tmp_path / 'absent.jsonl').exists() and not list(tmp_path.glob('*.tmp'))


def test_tecno_records_the_epochs_it_trained(tmp_path):
    torch.save(dict(epoch=7, model={}), tmp_path / 'latest.pt')
    assert pipeline.tecno_epochs(tmp_path) == 7
    assert pipeline.tecno_epochs(tmp_path / 'missing') is None
    (tmp_path / 'bad').mkdir()
    (tmp_path / 'bad' / 'latest.pt').write_text('not a checkpoint')
    assert pipeline.tecno_epochs(tmp_path / 'bad') is None


def test_univtg_batches_never_repeat_a_query_and_use_every_sample_once():
    import random
    from comparison_models.univtg import adapter
    samples = [dict(query=q) for q in PHRASES for _ in range(5)] + [dict(query=PHRASES[0])] * 7
    batches = adapter.distinct_query_batches(samples, 8, random.Random(1))
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(len(samples)))
    assert all(len({samples[i]['query'] for i in b}) == len(b) <= 8 for b in batches)
    assert sum(len(b) == 8 for b in batches) >= len(batches) - 4
    assert adapter.distinct_query_batches(samples, 8, random.Random(1)) == batches          # deterministic


def test_univtg_targets_follow_the_official_dense_formulation():
    import random
    from comparison_models.univtg import adapter
    s = dict(vfeat=np.zeros((10, 4)), windows=[[2, 4], [7, 9]])
    t = adapter.targets_for(s, random.Random(0))
    inside = t['timestamp_window'].numpy()
    assert inside.tolist() == [0, 0, 1, 1, 0, 0, 0, 1, 1, 0]          # clips 2,3 and 7,8
    np.testing.assert_allclose(t['timestamp'][:, 0].numpy(), (np.arange(10) + 0.5) / 10)
    np.testing.assert_allclose(t['span_labels_nn'][2].numpy(), [0.2, 0.4])
    np.testing.assert_allclose(t['span_labels_nn'][8].numpy(), [0.7, 0.9])
    assert t['saliency_pos_labels'][0] in (2, 3, 7, 8)
    # the official fall-back when no clip centre lies inside a window: the window's first clip
    t = adapter.targets_for(dict(vfeat=np.zeros((10, 4)), windows=[[5, 5.4]]), random.Random(0))
    assert t['timestamp_window'].numpy().tolist() == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]


def test_univtg_trains_resumes_and_predicts_on_synthetic_features(tmp_path, monkeypatch):
    from comparison_models.univtg import adapter
    monkeypatch.setitem(adapter.OFFICIAL, 'hidden_dim', 32)
    monkeypatch.setitem(adapter.OFFICIAL, 'dim_feedforward', 32)
    monkeypatch.setitem(adapter.OFFICIAL, 'enc_layers', 1)
    monkeypatch.setitem(adapter.OFFICIAL, 'nheads', 4)
    d = make_prepared(tmp_path, n_videos=4, seconds=24)
    meta = json.loads(d['metadata'].read_text())
    feats = tmp_path / 'features'
    feats.mkdir()
    rng = np.random.default_rng(0)
    for v, m in meta.items():
        n = len(range(0, m['frame_count'], 25))
        np.save(feats / f'{v}.npy', rng.normal(size=(n, adapter.VIDEO_DIM)).astype(np.float16))
    texts = {q: rng.normal(size=(5, adapter.TEXT_DIM)).astype(np.float16) for q in PHRASES}
    torch.save(texts, tmp_path / 'text.pt')
    out = tmp_path / 'univtg'
    args = argparse.Namespace(train=str(d['triplets']('train')), val=str(d['triplets']('val')), metadata=str(d['metadata']),
                              annotations=str(d['annotations']), features=str(feats), text_features=str(tmp_path / 'text.pt'),
                              output=str(out), clip_frames=4, epochs=3, eval_every=1, batch_size=4, lr=1e-4, seed=1,
                              sample_fps=1.0, device='cpu')
    adapter.train(args)
    rows = [json.loads(l) for l in (out / 'training_metrics.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in rows] == [1, 2, 3] and all('val_window_map' in r for r in rows)
    best = torch.load(out / 'best.pt', weights_only=False)
    adapter.train(args)                                                 # finished: nothing is trained again
    assert len((out / 'training_metrics.jsonl').read_text().splitlines()) == 3
    with pytest.raises(ValueError, match='different settings'):
        adapter.train(argparse.Namespace(**{**vars(args), 'lr': 2e-4}))
    pred = tmp_path / 'pred.csv'
    adapter.predict(argparse.Namespace(checkpoint=str(out / 'best.pt'), triplets=str(d['triplets']('test')),
                                       metadata=str(d['metadata']), annotations=str(d['annotations']), features=str(feats),
                                       text_features=str(tmp_path / 'text.pt'), output=str(pred),
                                       segments=str(tmp_path / 'seg.jsonl'), batch_size=4, device='cpu'))
    df = pd.read_csv(pred)
    assert set(df.text_query) == set(PHRASES) and df.score.between(0, 1).all()
    test_video = d['split']['test'][0]
    assert len(df[df.video_id == test_video]) == 14 * len(range(0, meta[test_video]['frame_count'], 25))
    segs = [json.loads(l) for l in (tmp_path / 'seg.jsonl').read_text().splitlines()]
    assert len(segs) == 14 * len(d['split']['test'])
    assert all(0 <= w[0] < w[1] <= s['duration'] and 0 <= w[2] <= 1 for s in segs for w in s['predictions'])
    man = json.loads(Path(str(pred) + '.manifest.json').read_text())
    assert man['frame_scores'] == len(df) and man['checkpoint_epoch'] == best['epoch']


# --------------------------------------------------------------------------- analysis metrics
def test_weighted_ap_and_auroc_match_scikit_learn_with_ties_and_weights():
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(3)
    for _ in range(20):
        n = int(rng.integers(5, 300))
        y = rng.integers(0, 2, n)
        if y.sum() in (0, n):
            continue
        s = np.round(rng.random(n), int(rng.integers(1, 3)))           # many ties
        video = rng.integers(0, 6, n)
        w = rng.integers(0, 3, 6).astype(float)
        if not ((w[video] > 0) & (y == 1)).any() or not ((w[video] > 0) & (y == 0)).any():
            continue
        r = pa.Ranked(y, s, video)
        assert r.ap() == pytest.approx(average_precision_score(y, s))
        assert r.ap(w) == pytest.approx(average_precision_score(y, s, sample_weight=w[video]))
        assert r.auroc(w) == pytest.approx(roc_auc_score(y, s, sample_weight=w[video]))


def test_segment_metrics_match_the_repository_definitions():
    from metrics import temporal_ap
    from evaluate import segment_metrics
    rng = np.random.default_rng(5)
    pairs, records = [], []
    for k in range(25):
        dur = 100.0
        gt = []
        t = 0.0
        while rng.random() < 0.7 and t < 80:
            a = t + rng.uniform(0, 10); b = a + rng.uniform(1, 10); gt.append([a, b]); t = b + 1
        pred = []
        for _ in range(int(rng.integers(0, 6))):
            a = rng.uniform(0, 90); b = a + rng.uniform(0.5, 10)
            pred.append([a, min(b, dur), float(np.round(rng.random(), 1))])
        key = (f'v{k % 5}', f'q{k}')
        pairs.append(dict(video=k % 5, key=str(key), gt=gt, pred=pred, frame_iou=0.5 if gt else None))
        records.append(dict(video_id=key[0], text_query=key[1], duration=dur, ground_truth=gt, predictions=pred))
    match = pa.SegmentMatch(pairs, 0.5, 5)
    ours = match.metrics()
    theirs = segment_metrics(records, 0.5)
    for t in pa.IOUS:
        key = lambda r: (r['video_id'], r['text_query'])                                    # noqa: E731
        assert ours[f'mAP@{t}'] == pytest.approx(temporal_ap({key(r): r['predictions'] for r in records},
                                                             {key(r): r['ground_truth'] for r in records}, t))
        assert ours[f'mAP@{t}'] == pytest.approx(theirs[f'pooled_AP@{t}'])
        assert ours[f'R1@{t}'] == pytest.approx(theirs[f'R1@{t}'])
    assert ours['absent_false_alarm'] == pytest.approx(theirs['absent_query_false_positive_rate'])
    # weights = how often each video is drawn; duplicating the pairs of a video gives the same numbers
    w = np.asarray([2.0, 1, 0, 1, 1])
    dup = [dict(p, video=i) for i, p in enumerate([q for q in pairs for _ in range(int(w[q['video']]))])]
    dup = [dict(p, key=str((p['key'], i))) for i, p in enumerate(dup)]
    for t in pa.IOUS:
        expect = pa.SegmentMatch(dup, 0.5, len(dup)).metrics()[f'R1@{t}']
        assert match.metrics(w)[f'R1@{t}'] == pytest.approx(expect)


def test_phase_recognition_metrics_on_a_hand_example():
    gt = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    pred = np.array([0, 0, 1, 1, 1, 2, 2, 0])
    t = pa.phase_tables({0: (gt, pred)})
    s = pa.phase_summary(t)
    assert s['accuracy'] == pytest.approx(6 / 8)
    assert s['per_phase']['recall'][:3] == pytest.approx([2 / 3, 1.0, 2 / 3])
    assert s['per_phase']['precision'][:3] == pytest.approx([2 / 3, 2 / 3, 1.0])
    assert s['per_phase']['jaccard'][:3] == pytest.approx([2 / 4, 2 / 3, 2 / 3])
    assert all(math.isnan(x) for x in s['per_phase']['recall'][3:])         # phases absent from the video


def test_phase_summary_uses_sample_standard_deviations_like_the_cholec80_script():
    gt = np.array([0, 0, 1, 1])
    t = pa.phase_tables({0: (gt, np.array([0, 0, 1, 1])), 1: (gt, np.array([0, 1, 1, 1]))})
    s = pa.phase_summary(t)
    assert s['accuracy'] == pytest.approx(0.875) and s['accuracy_std'] == pytest.approx(np.std([1, 0.75], ddof=1))
    assert math.isnan(pa.std1([0.5, math.nan])) and pa.std1([1, 3, math.nan]) == pytest.approx(math.sqrt(2))


def test_relaxed_phase_metrics_reproduce_the_cholec80_evaluation_script():
    # Hand-computed with the MATLAB semantics of Evaluate.m (phases 1-based there; fps 0.2 -> 2-frame boundary).
    # 1) a too-early transition at the END of a phase is not relaxed: the script's end-of-phase statement
    #    indexes the first t frames (a logical mask of length t), which this port reproduces.
    jac, prec, rec, acc = pa.relaxed_evaluate(np.array([0] * 5 + [1] * 5), np.array([0] * 4 + [1] * 6), 0.2)
    assert acc == pytest.approx(90)
    np.testing.assert_allclose(jac[:2], [80, 100 * 5 / 6]); np.testing.assert_allclose(prec[:2], [100, 100 * 5 / 6])
    np.testing.assert_allclose(rec[:2], [80, 100])
    assert np.isnan(jac[2:]).all() and np.isnan(prec[2:]).all() and np.isnan(rec[2:]).all()
    # 2) a late transition (the previous phase still predicted at the start of the next) is relaxed; relaxed
    #    true positives can exceed the frames of a phase (133 %), and Main.m clips to 100
    gt, pred = np.array([0] * 3 + [1] * 5), np.array([0] * 4 + [1] * 4)
    jac, prec, rec, acc = pa.relaxed_evaluate(gt, pred, 0.2)
    assert acc == pytest.approx(100)
    np.testing.assert_allclose(rec[:2], [400 / 3, 100]); np.testing.assert_allclose(prec[:2], [100, 125])
    r = pa.relaxed_phase_metrics({0: (gt, pred)}, 0.2)
    assert r['per_phase']['recall'][:2] == pytest.approx([1, 1]) and r['per_phase']['precision'][:2] == pytest.approx([1, 1])
    # 3) Gallbladder dissection/packaging (4, 5) and cleaning/retraction (6, 7) also accept two phases away
    gt, pred = np.array([4] * 6 + [5] * 6), np.array([4] * 6 + [3] * 1 + [5] * 5)  # dissection at the start of cleaning
    assert pa.relaxed_evaluate(gt, pred, 0.2)[3] == pytest.approx(100)              # (-2 at the start of phase 6)
    assert pa.relaxed_evaluate(gt + 0, np.array([4] * 6 + [3] * 3 + [5] * 3), 0.2)[3] == pytest.approx(100 * 11 / 12)
    gt, pred = np.array([2] * 4 + [3] * 4), np.array([2] * 4 + [1] + [3] * 3)       # -2 is not relaxed for phase 4
    assert pa.relaxed_evaluate(gt, pred, 0.2)[3] == pytest.approx(100 * 7 / 8)
    # 4) a phase never predicted: precision inf (tp > 0 from relaxed frames) -> clipped to 100 by Main.m
    gt, pred = np.array([0] * 3 + [1] * 2 + [2] * 3), np.array([0] * 5 + [2] * 3)
    prec = pa.relaxed_evaluate(gt, pred, 0.2)[1]
    assert np.isinf(prec[1]) and pa.relaxed_phase_metrics({0: (gt, pred)}, 0.2)['per_phase']['precision'][1] == 1
    # 5) Main.m's means: Jaccard/recall plain mean over phases (NaN if a phase is in no video), precision nanmean
    r = pa.relaxed_phase_metrics({0: (np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]))}, 1.0)
    assert math.isnan(r['jaccard']) and r['precision'] == 1 and r['accuracy'] == 1
    gt = np.repeat(np.arange(7), 3)
    r = pa.relaxed_phase_metrics({0: (gt, gt), 1: (gt, np.roll(gt, 1))}, 0.2)
    second = pa.relaxed_evaluate(gt, np.roll(gt, 1), 0.2)[3]
    assert r['jaccard'] <= 1 and r['accuracy_std'] == pytest.approx(np.std([100, second], ddof=1) / 100)


def test_relaxed_metrics_keep_annotation_gaps_on_the_time_grid():
    # fps 0.2 -> a 2-frame boundary. Phase 1 (0-based 0) for 6 frames, then phase 2; the prediction switches 1 frame
    # late (relaxed). Unknown frames (-1) inside, between and around phases must never be merged away.
    gt = np.array([0] * 6 + [1] * 6)
    pred = np.array([0] * 7 + [1] * 5)
    assert pa.relaxed_evaluate(gt, pred, 0.2)[3] == pytest.approx(100)
    # a 3-frame gap at the transition: the late frame is now the first frame of phase 2 AFTER the gap - still the
    # first 2 frames of that run, so still relaxed; nothing else changes and the gap is not counted
    gt_gap = np.r_[gt[:6], [-1] * 3, gt[6:]]
    pred_gap = np.r_[pred[:6], [0] * 3, pred[6:]]
    jac, prec, rec, acc = pa.relaxed_evaluate(gt_gap, pred_gap, 0.2)
    assert acc == pytest.approx(100) and rec[1] == pytest.approx(100)
    # a gap inside a phase splits it into two runs: an error right after the gap is "late" for the second run
    gt_in = np.array([1] * 4 + [-1] * 3 + [1] * 4)
    pred_in = np.array([1] * 4 + [1] * 3 + [0] + [1] * 3)               # phase 1 predicted as phase 0 after the gap
    assert pa.relaxed_evaluate(gt_in, pred_in, 0.2)[3] == pytest.approx(100)     # 0 - 1 = -1: late, relaxed
    # unknown frames at the start and the end are ignored completely
    a = pa.relaxed_evaluate(np.r_[[-1] * 2, gt, [-1] * 2], np.r_[[3] * 2, pred, [5] * 2], 0.2)
    b = pa.relaxed_evaluate(gt, pred, 0.2)
    for x, z in zip(a, b):
        np.testing.assert_allclose(x, z, equal_nan=True)
    # phase_recognition keeps the grid position of every annotated frame; the metrics use it
    per_video = {0: (gt, pred, np.r_[np.arange(6), np.arange(9, 15)])}
    assert pa.phase_gaps(per_video) == {0: 3}
    r = pa.relaxed_phase_metrics(per_video, 0.2)
    assert r['accuracy'] == pytest.approx(1)


def test_symmetric_relaxation_also_forgives_early_transitions():
    gt, pred = np.array([0] * 5 + [1] * 5), np.array([0] * 4 + [1] * 6)    # phase 2 predicted 1 frame early
    assert pa.relaxed_evaluate(gt, pred, 0.2)[3] == pytest.approx(90)               # legacy script: not forgiven
    assert pa.relaxed_evaluate(gt, pred, 0.2, legacy=False)[3] == pytest.approx(100)  # as intended: forgiven
    late = np.array([0] * 6 + [1] * 4)                                               # late transitions: both forgive
    assert pa.relaxed_evaluate(gt, late, 0.2)[3] == pa.relaxed_evaluate(gt, late, 0.2, legacy=False)[3] == 100


def test_bootstrap_resamples_keep_every_defined_query(tmp_path):
    frames = _synthetic_frames(n_videos=8, frames=20)
    labels = frames.label.copy()
    rare, never = 13, 12                                 # a tool seen in one video only, a tool never seen
    labels[frames.query == rare] = 0
    labels[(frames.query == rare) & (frames.video == 3)] = 1
    labels[frames.query == never] = 0
    frames.label = labels
    counts, redrawn, defined = pa.bootstrap_counts(frames, 300, np.random.default_rng(0))
    assert counts.shape == (300, 8) and (counts.sum(axis=1) == 8).all()
    assert (counts[:, 3] > 0).all() and redrawn > 0                 # every resample contains the rare tool's video
    assert defined.sum() == 13 and not defined[never]               # the absent one is undefined, not redrawn for
    scores = np.random.default_rng(1).random(len(labels))
    reps = pa.frame_replicates(frames, scores, 0.5, counts)
    qap = reps['per_query_ap']
    assert np.isfinite(qap[:, defined]).all() and np.isnan(qap[:, never]).all()   # the same 13 queries every time
    with pytest.raises(ValueError, match='too few test videos'):
        pa.bootstrap_counts(frames, 300, np.random.default_rng(0), max_redraws=5)


def test_model_sources_records_revisions_and_never_raises(tmp_path, monkeypatch):
    from types import SimpleNamespace
    root = tmp_path / 'root'
    repo = root / 'hf_cache' / 'hub' / 'models--openai--clip-vit-base-patch32'
    (repo / 'snapshots' / ('a' * 40)).mkdir(parents=True)
    (repo / 'refs').mkdir()
    (repo / 'refs' / 'main').write_text('a' * 40 + '\n')
    (root / 'hf_cache' / 'hub' / 'models--broken--repo' / 'refs' / 'main').mkdir(parents=True)   # a folder, unreadable
    ck = root / 'torch_cache' / 'hub' / 'checkpoints'
    ck.mkdir(parents=True)
    (ck / 'raft_small_C_T_V2-01064c6d.pth').write_bytes(b'x' * 10)
    weights = tmp_path / 'm2crl.pth'
    weights.write_bytes(b'y' * 7)
    (root / 'state').mkdir()
    (root / 'state' / 'weights_ref.json').write_text(json.dumps(dict(path=str(weights), random_init=False)))
    good, odd, none = (SimpleNamespace(name=n, results=tmp_path / n) for n in ('good', 'odd', 'none'))
    for m in (good, odd):
        m.results.mkdir()
    (good.results / 'proposed_test.csv.manifest.json').write_text(json.dumps(dict(checkpoint_sha256='c' * 64, x=1)))
    (odd.results / 'proposed_test.csv.manifest.json').write_text('[1, 2]')                       # malformed
    s = pa.model_sources(root, [good, odd, none])
    assert s['huggingface']['openai/clip-vit-base-patch32'] == dict(revision='a' * 40, snapshots=['a' * 40])
    assert s['torchvision'] == [dict(file='raft_small_C_T_V2-01064c6d.pth', bytes=10)]
    assert s['backbone']['bytes'] == 7 and s['weights'] == {'good': {'checkpoint_sha256': 'c' * 64}}
    assert len(s['unreadable']) == 2 and any('broken' in u for u in s['unreadable']) and \
        any('odd' in u for u in s['unreadable'])
    assert pa.model_sources(tmp_path / 'nothing', [])['unreadable'] == []          # an empty root is fine too
    json.dumps(s)
    assert s['surgvlp_code'] == dict(repository=protocol.SURGVLP['repository'], commit=protocol.SURGVLP['commit'])
    monkeypatch.setattr(pa.protocol, 'SURGVLP', {})                                 # even a broken constant
    s = pa.model_sources(root, [good])
    assert s['surgvlp_code'] is None and any('SurgVLP' in u for u in s['unreadable']) and s['weights']


def test_a_missing_last_journal_row_is_reported(tmp_path):
    plan = [p for p in protocol.plans('paper') if p['name'] == 'paper_baseline_s42'][0]
    root = tmp_path
    m = pa.Method(plan, root)
    m.run.mkdir(parents=True)
    (m.run / 'run_config.json').write_text(json.dumps(dict(TRAIN=dict(NUM_EPOCHS=3))))
    (m.run / 'training_metrics.jsonl').write_text(''.join(json.dumps(dict(epoch=e, train_seconds=10)) + '\n'
                                                          for e in (1, 2)))
    missing = {}
    pa.efficiency(m, root, missing)
    assert missing == {}                                            # training not finished: nothing to report yet
    (root / 'state').mkdir()
    (root / 'state' / f'train.{m.name}.done.json').write_text('{}')
    pa.efficiency(m, root, missing)
    assert missing == {'runs/paper_baseline_s42/training_metrics.jsonl': [3]}     # the trailing epoch
    cmp_plan = [p for p in protocol.plans('paper') if p['kind'] == 'closedset'][0]
    c = pa.Method(cmp_plan, root)
    (c.run / 'closedset').mkdir(parents=True)
    (c.run / 'closedset' / 'training_metrics.jsonl').write_text(json.dumps(dict(epoch=1)) + '\n')
    (c.run / 'closedset' / 'FINISHED.json').write_text(json.dumps(dict(epochs=2)))
    missing = {}
    pa.efficiency(c, root, missing)
    assert missing == {'runs/paper_closedset_s42/closedset/training_metrics.jsonl': [2]}
    # A finished training whose whole journal is gone is reported too; one that never writes a journal is not.
    (m.run / 'training_metrics.jsonl').unlink()
    (c.run / 'closedset' / 'training_metrics.jsonl').unlink()
    missing = {}
    pa.efficiency(m, root, missing)
    pa.efficiency(c, root, missing)
    assert missing == {'runs/paper_baseline_s42/training_metrics.jsonl': [1, 2, 3],
                       'runs/paper_closedset_s42/closedset/training_metrics.jsonl': [1, 2]}
    t = pa.Method([p for p in protocol.plans('paper') if p['kind'] == 'tecno'][0], root)
    (t.run / 'tecno_tcn').mkdir(parents=True)
    (t.run / 'tecno_tcn' / 'FINISHED.json').write_text(json.dumps(dict(finished='x')))   # early stopping: no count
    x = pa.Method([p for p in protocol.plans('paper') if p['kind'] == 'xclip_finetune'][0], root)
    (x.run / 'xclip').mkdir(parents=True)
    (x.run / 'xclip' / 'FINISHED.json').write_text(json.dumps(dict(epochs=5)))          # X-CLIP writes no journal
    z = pa.Method([p for p in protocol.plans('paper') if p['kind'] == 'clip_zeroshot'][0], root)  # no training
    missing = {}
    for method in (t, x, z):
        pa.efficiency(method, root, missing)
    assert missing == {'runs/paper_tecno_s42/tecno_tcn/training_metrics.jsonl': 'all (no journal)'}
    (t.run / 'tecno_tcn' / 'training_metrics.jsonl').write_text('')                    # present but empty
    missing = {}
    pa.efficiency(t, root, missing)
    assert missing == {'runs/paper_tecno_s42/tecno_tcn/training_metrics.jsonl': 'all (empty journal)'}
    # damaged rows are skipped, never fatal; the epochs they would have recorded show as gaps
    (c.run / 'closedset' / 'training_metrics.jsonl').write_text(
        '5\n' + json.dumps(dict(epoch=1, train_seconds='broken')) + '\n' + json.dumps(dict(epoch=2, train_seconds=7)) + '\n')
    missing = {}
    row = pa.efficiency(c, root, missing)
    assert missing == {'runs/paper_closedset_s42/closedset/training_metrics.jsonl': [1]} and row['Training (h)'] == '0.0'


def test_holm_and_bootstrap_p_values():
    np.testing.assert_allclose(pa.holm([0.01, 0.04, 0.03, math.nan]), [0.03, 0.06, 0.06, math.nan])
    # null-centred: the bootstrap distribution shifted to 0 is compared with the observed difference
    rng = np.random.default_rng(0)
    assert pa.bootstrap_p(rng.normal(0.5, 0.1, 999), 0.5) == pytest.approx(1 / 1000)     # 5 SE away: never reached
    assert pa.bootstrap_p(rng.normal(0.05, 0.1, 999), 0.05) > 0.5                       # half an SE: not significant
    assert pa.bootstrap_p(np.r_[np.ones(500), -np.ones(500)], 0.0) == 1.0
    assert math.isnan(pa.bootstrap_p([], 0.1))


def _synthetic_frames(n_videos=8, frames=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for v in range(n_videos):
        for q in PHRASES:
            y = (rng.random(frames) < 0.3).astype(int)
            for i in range(frames):
                rows.append(dict(video_id=f'CHOLEC80__video{v + 1:02d}', text_query=q, frame_idx=25 * i, label=y[i]))
    df = pd.DataFrame(rows).sort_values(['video_id', 'text_query', 'frame_idx'])
    df['_q'] = df.text_query.map({q: i for i, q in enumerate(PHRASES)})
    df = df.sort_values(['video_id', '_q', 'frame_idx'], kind='mergesort').reset_index(drop=True)
    return pa.Frames(df)


def test_paired_permutation_test_of_the_map_difference():
    frames = _synthetic_frames()
    rng = np.random.default_rng(1)
    y = frames.label.astype(float)
    good = np.clip(0.5 * y + 0.5 * rng.random(len(y)), 0, 1)              # informative scores
    noise = rng.random(len(y))                                             # chance level
    twin_a = np.clip(good + rng.normal(0, 0.2, len(y)), 0, 1)            # two equally good methods
    twin_b = np.clip(good + rng.normal(0, 0.2, len(y)), 0, 1)
    scores = dict(good=good, noise=noise, twin_a=twin_a, twin_b=twin_b, same=good.copy())
    pairs = [('good', 'noise'), ('good', 'same'), ('twin_a', 'twin_b')]
    res = pa.permutation_map_tests(frames, scores, pairs, 200, seed=3)
    stats = {k: pa.frame_summary(frames, v, 0.5) for k, v in scores.items()}
    for x, z in pairs:                                                    # the observed difference is the mAP difference
        assert res[(x, z)][0] == pytest.approx(stats[x]['macro_map'] - stats[z]['macro_map'])
    # only swapping none or all of the 8 videos reaches |d_obs| (2 of 256 patterns): p near 1/201 + 2/256
    assert res[('good', 'noise')][1] < 0.05
    assert res[('good', 'same')] == (0.0, 1.0)                            # identical methods
    assert res[('twin_a', 'twin_b')][1] > 0.05
    # the same numbers with several processes (each pair has its own random stream)
    assert pa.permutation_map_tests(frames, scores, pairs, 200, seed=3, workers=2) == res
    # A monotone change of one method's score scale (e.g. a probability vs a similarity) changes no AP, and
    # now changes no p-value either: the test works on within-query ranks.
    rescaled = dict(scores, good=good ** 5, twin_a=0.2 + 0.1 * twin_a)
    again = pa.permutation_map_tests(frames, rescaled, pairs, 200, seed=3)
    for pair in pairs:
        assert again[pair][0] == pytest.approx(res[pair][0], abs=1e-12) and again[pair][1] == res[pair][1]


def _write_results(root, d, plan, rng, quality, sets=('val', 'test'), para=None):
    from data_contract import AnnotationIndex, query_spec
    meta = json.loads(d['metadata'].read_text())
    index = AnnotationIndex(d['annotations'])
    results = root / 'results' / plan['name']
    results.mkdir(parents=True, exist_ok=True)
    for name in sets + tuple(f'test_para{k}' for k in (para or {})):
        split = 'val' if name == 'val' else 'test'
        rows = []
        for v in d['split'][split]:
            m = meta[v]
            for q in PHRASES:
                text = q if not name.startswith('test_para') else para[name[-1]][query_spec(q)]
                for i in range(0, m['frame_count'], 25):
                    y = index.label(v, i, query_spec(q))
                    s = 1 / (1 + math.exp(-(quality * (2 * (y == 1) - 1) + rng.normal())))
                    rows.append(dict(video_id=v, text_query=text, frame_idx=i, time_sec=i / 25, score=s, label=y))
        df = pd.DataFrame(rows)
        if plan['short'] == 'advanced':
            df['evidence_uncertainty'] = rng.random(len(df))
        df.to_csv(results / f'proposed_{name}.csv', index=False)
        df[['video_id', 'text_query', 'frame_idx', 'label']].to_csv(results / f'{name}_reference.csv', index=False)
        if name == 'test' and plan['kind'] in protocol.SURGVLP_ZERO_SHOT:
            df.to_csv(results / 'authorprompts_test.csv', index=False)
        if name == 'test' and plan['kind'] == 'univtg':
            with open(results / 'native_segments_test.jsonl', 'w') as f:
                for (v, q), g in df.groupby(['video_id', 'text_query']):
                    f.write(json.dumps(dict(video_id=v, text_query=q, duration=meta[v]['frame_count'] / 25,
                                            predictions=[[1.0, 4.0, 0.9], [6.0, 9.0, 0.4]])) + '\n')
    (results / 'proposed_calibration.json').write_text(json.dumps(dict(split='validation', threshold=0.5)))
    (results / 'proposed_test_metrics.json').write_text('{}')
    (root / 'state').mkdir(exist_ok=True)
    (root / 'state' / f'evaluate.{plan["name"]}.done.json').write_text('{}')
    run = root / 'runs' / plan['name']
    run.mkdir(parents=True, exist_ok=True)
    if plan['kind'] == 'proposed':
        (run / 'training_metrics.jsonl').write_text(''.join(json.dumps(dict(
            epoch=e, train_loss=1 / e, val_frame_nll=0.5 + abs(e - 2) / 10, average_precision=0.8, train_seconds=60,
            val_seconds=10, peak_gpu_memory_gb=12.5, train_clips=10000, optimizer_steps=100)) + '\n' for e in (1, 2, 3)))
        (run / 'parameter_counts.json').write_text(json.dumps(dict(vision_backbone=dict(total=86e6, trainable=1e6))))
        man = dict(frame_scores=1000, seconds=10, parameters=150e6)
    else:
        man = dict(frames=100, seconds=2, parameters=50e6)
    (results / 'proposed_test.csv.manifest.json').write_text(json.dumps(man))


def test_the_paper_analysis_writes_every_table_and_figure(tmp_path):
    root = tmp_path / 'root'
    d = make_prepared(root, n_videos=8, seconds=40, split=dict(
        train=[f'CHOLEC80__video0{i}' for i in (1, 2, 3, 4)], val=['CHOLEC80__video05'],
        test=['CHOLEC80__video06', 'CHOLEC80__video07', 'CHOLEC80__video08']))
    para = protocol.load_paraphrases()
    rng = np.random.default_rng(0)
    plans = protocol.plans('paper')
    for p in plans:
        p['short'] = 'advanced' if p['flags'] == protocol.ADVANCED else p['kind']
        if p['name'] == 'paper_peskavlp_finetune_s42':
            continue                                                   # a run that has not finished yet
        _write_results(root, d, p, rng, quality=3.0 if p['kind'] == 'proposed' else 1.0,
                       para=para if p['paraphrases'] else None)
    assert pa.main(['--root', str(root), '--bootstrap', '30', '--permutations', '20']) == 0
    out = root / 'paper'
    text = (out / 'PAPER_RESULTS.md').read_text()
    assert 'Runs finished: **19 of 20**' in text and 'paper_peskavlp_finetune_s42' in text
    for section in ('Main results', 'Paired significance', 'Temporal segments', 'phase recognition', 'Tool presence',
                    'Per-query AP', 'Ablations', 'seeds', 'unseen query wordings', 'Efficiency', 'Published results'):
        assert section.lower() in text.lower(), section
    for table in ('main_results', 'significance', 'segment_results', 'phase_recognition', 'phase_recognition_relaxed',
                  'phase_recognition_relaxed_symmetric', 'tool_presence', 'per_query_ap', 'breakdown', 'ablations',
                  'seeds', 'generalisation', 'efficiency', 'literature_template'):
        assert (out / 'tables' / f'{table}.csv').exists() and (out / 'tables' / f'{table}.tex').exists(), table
        assert '\\begin{tabular}' in (out / 'tables' / f'{table}.tex').read_text()
    for fig in ('reliability', 'risk_coverage', 'training_dynamics', 'per_query_ap', 'generalisation',
                'timeline_CHOLEC80__video06'):
        assert (out / 'figures' / f'{fig}.png').exists() and (out / 'figures' / f'{fig}.pdf').exists(), fig
    report = json.loads((out / 'analysis.json').read_text())
    assert 'figure_error' not in report
    main = pd.read_csv(out / 'tables' / 'main_results.csv')
    assert list(main.Method)[-2:] == ['Baseline (ours)', 'Advanced (ours)'] and len(main) == 11
    assert list(main.Method)[:4] == ['CLIP zero-shot', 'SurgVLP zero-shot', 'HecVL zero-shot', 'PeskaVLP zero-shot']
    assert {'TeCNO (re-implementation)', 'Closed-set, our backbone (compute-efficient)',
            'Closed-set, our backbone (exposure-matched)'} <= set(main.Method)
    seg = pd.read_csv(out / 'tables' / 'segment_results.csv')
    assert any('own windows' in m for m in seg.Method)
    phase = pd.read_csv(out / 'tables' / 'phase_recognition.csv')
    assert sum("authors' phase prompts" in m for m in phase.Method) == 3          # SurgVLP, HecVL, PeskaVLP
    relaxed = pd.read_csv(out / 'tables' / 'phase_recognition_relaxed.csv')
    assert list(relaxed.Method) == list(phase.Method)
    for key, r in report['phase_recognition'].items():                 # relaxing the boundaries never hurts
        assert r['relaxed']['accuracy'] >= r['accuracy'] - 1e-12, key
        assert r['relaxed_symmetric']['accuracy'] >= r['accuracy'] - 1e-12, key
    assert report['phase_annotation_gaps']['videos_with_gaps'] == []
    assert 'No test video has a frame without a phase annotation' in text
    lit = pd.read_csv(out / 'tables' / 'literature_template.csv')
    tecno = lit[lit.Method.str.startswith('TeCNO')].iloc[0]
    assert '40 / 8 / 32' in tecno['What matches'] and 'legacy relaxed' in tecno['Ours: accuracy on our frames']
    assert lit[lit.Method.str.startswith('HecVL')].iloc[0]['Ours: accuracy on our frames'] != 'n/a'
    eff = pd.read_csv(out / 'tables' / 'efficiency.csv').set_index('Method')
    assert eff.loc['HecVL zero-shot', 'Training (h)'] == '0 (zero-shot)'
    assert eff.loc['Baseline (ours)', 'Training clips'] == '30,000'           # 3 epochs x 10,000 clips
    assert int(eff.loc['Baseline (ours)', 'Optimiser steps']) == 300
    gen = pd.read_csv(out / 'tables' / 'generalisation.csv')
    # every method with a text input, except the unfinished run; never the closed-set models
    assert {'AP set A', 'AP set B', 'mAP set A'} <= set(gen.columns) and len(gen) == 2 + 4 + 6
    assert not any(m.startswith(('TeCNO', 'Closed-set')) for m in gen.Method)
    seeds = pd.read_csv(out / 'tables' / 'seeds.csv')
    assert list(seeds.Seeds) == ['42, 43, 44', '42, 43, 44']
    m = report['methods']['paper_advanced_s42']['frame']
    assert m['average_precision'] > 0.8 and 'evidence_uncertainty_aurc' in m
    assert report['queries_without_test_positives'] == [] and report['bootstrap_resamples_redrawn'] >= 0
    assert report['model_sources']['surgvlp_code']['commit'] == protocol.SURGVLP['commit']
    sig = report['significance']
    assert {r['metric'] for r in sig} == {'AP', 'mAP', 'F1', 'Segment mAP@0.5'}
    assert {r['test'] for r in sig if r['metric'] == 'mAP'} == {'permutation (primary)'}
    assert {r['test'] for r in sig if r['metric'] != 'mAP'} == {'bootstrap (secondary)'}
    primary = [r for r in sig if r['metric'] == 'mAP']
    assert len(primary) == 1 + 2 * 9 + 4             # adv vs base, ours vs the 9 finished comparisons, ablations
    assert all(1 / 21 - 1e-9 <= r['p'] <= 1 for r in primary) and report['permutations'] == 20
    assert 'mAP (14 queries)' in main.columns and report['map_queries'] == 14
    assert 'Primary endpoint: 14-query mAP' in text
    # A query without any positive test frame: the mAP is named after the queries it really averages, everywhere.
    gone = protocol.canonical_queries()[-1]                         # a tool (phases must cover every frame)
    wordings = {gone} | {para[k][query_spec(gone)] for k in para}
    for f in (root / 'results').glob('*/*test*.csv'):              # every test file of every run, all wordings
        df = pd.read_csv(f)
        if {'text_query', 'label'} <= set(df.columns):
            df.loc[df.text_query.isin(wordings) & (df.label == 1), 'label'] = 0
            df.to_csv(f, index=False)
    assert pa.main(['--root', str(root), '--bootstrap', '10', '--permutations', '5']) == 0
    report = json.loads((out / 'analysis.json').read_text())
    assert report['queries_without_test_positives'] == [gone] and report['map_queries'] == 13
    main = pd.read_csv(out / 'tables' / 'main_results.csv')
    assert 'mAP (13 of 14 queries)' in main.columns and 'mAP (14 queries)' not in main.columns
    text = (out / 'PAPER_RESULTS.md').read_text()
    assert '14-query mAP' not in text and 'mAP over the 13 of 14 queries with positive test frames' in text
    assert f'no positive test frame: {gone}' in text
    # a missing input of a finished run is an error, not a silently smaller table
    (root / 'results' / 'paper_clip_zeroshot' / 'test_reference.csv').write_text('video_id,text_query,frame_idx,label\n')
    with pytest.raises(ValueError):
        pa.main(['--root', str(root), '--bootstrap', '5', '--permutations', '5'])


def test_the_analysis_of_a_root_without_finished_runs_is_a_short_note(tmp_path):
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data' / 'video_metadata.json').write_text(json.dumps({'v': dict(source_fps=25.0, frame_count=10)}))
    assert pa.main(['--root', str(tmp_path), '--bootstrap', '5']) == 0
    assert 'No run has finished yet' in (tmp_path / 'paper' / 'PAPER_RESULTS.md').read_text()


# --------------------------------------------------------------------------- report bundle
def test_report_bundle_carries_the_paper_tables_figures_and_per_run_logs(tmp_path):
    import tarfile
    p = pipeline.Paths(tmp_path, 'paper_baseline_s42')
    (p.paper / 'tables').mkdir(parents=True)
    (p.paper / 'figures').mkdir(parents=True)
    (p.paper / 'PAPER_RESULTS.md').write_text('# results')
    (p.paper / 'tables' / 'main_results.tex').write_text('\\begin{tabular}')
    (p.paper / 'figures' / 'reliability.png').write_bytes(b'\x89PNG fake')
    (p.paper / 'figures' / 'reliability.pdf').write_bytes(b'%PDF fake')
    (p.logs / 'paper_baseline_s42').mkdir(parents=True)
    (p.logs / 'paper_baseline_s42' / 'train.log').write_text('epoch 1')
    (p.logs / 'main.log').write_text('main')
    a = argparse.Namespace(run_name='paper', preset='paper')
    pipeline.write_report_bundle(a, [p], 0)
    name = (p.outbox / 'LATEST.txt').read_text().strip()
    with tarfile.open(p.outbox / name) as tar:
        names = set(tar.getnames())
    assert {'paper/PAPER_RESULTS.md', 'paper/tables/main_results.tex', 'paper/figures/reliability.png',
            'logs/paper_baseline_s42/train.log', 'logs/main.log'} <= names
    assert 'paper/figures/reliability.pdf' not in names


def test_each_run_records_the_unseen_wording_manifests_it_was_evaluated_on(tmp_path):
    make_prepared(tmp_path, n_videos=4, seconds=10)
    p = pipeline.Paths(tmp_path, 'r')
    pipeline.check_run_inputs(p, 'r')                                  # recorded before the manifests existed
    protocol.write_paraphrase_triplets(p.triplet_csv('test'), p.triplets)
    p.results.mkdir(parents=True)
    (p.results / 'proposed_test.csv').write_text('x')                 # the run has produced something
    pipeline.check_run_inputs(p, 'r')                                  # new files are added to the record
    assert 'cholec80_test_paraA_triplets.csv' in json.loads(p.run_inputs.read_text())
    manifest = p.triplets / 'cholec80_test_paraA_triplets.csv'
    manifest.write_text(manifest.read_text() + '\n')
    with pytest.raises(pipeline.StageError, match='changed'):
        pipeline.check_run_inputs(p, 'r')


def test_a_damaged_journal_row_never_costs_the_training_figure(tmp_path):
    from types import SimpleNamespace
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run = tmp_path / 'run'
    run.mkdir()
    good = [dict(epoch=e, train_loss=1 / e, val_frame_nll=0.5, average_precision=0.7) for e in (2, 1)]
    (run / 'training_metrics.jsonl').write_text('\n'.join(
        [json.dumps(good[0]), '{"epoch": 3, "train_lo', '7', json.dumps(dict(epoch=4, train_loss='x', val_frame_nll=0.1,
                                                                         average_precision=0.1)), json.dumps(good[1])]) + '\n')
    m = SimpleNamespace(run=run, label='Baseline (ours)')
    assert pa.training_figure(plt, tmp_path, [m]) == ['training_dynamics']
    assert list(tmp_path.rglob('training_dynamics.png'))
