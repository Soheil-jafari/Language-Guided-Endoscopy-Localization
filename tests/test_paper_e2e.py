"""The whole paper protocol, end to end, on synthetic videos and on a CPU (about 45 minutes):

    LGEL_PAPER_E2E=1 LGEL_PESKAVLP_PYTHON=/path/to/envs/lgel-peskavlp/bin/python \
        python -m pytest tests/test_paper_e2e.py -s

Real code everywhere (main.py and every stage, train.py, predict.py, evaluate.py, the comparison runners -
CLIP, X-CLIP, the official SurgVLP code (SurgVLP, HecVL, PeskaVLP) in its own environment, the official UniVTG
model, TeCNO, the closed-set model -, the analysis and the report), with three test-only substitutions made by
tests/e2e_hook: a smaller backbone and crop (so a CPU can train it), zero optical flow and a Mamba stand-in
(both need a GPU or the network); and stand-ins for the downloaded models (random weights of the real
architectures; tiny Hugging Face models under their real names in the job's own model cache), because this
machine has no internet (the job runs with --offline).

The test stops a job in the middle (as the time limit would), injects a failure into one run, and checks
that the protocol carries on, that the next job finishes exactly what is missing, and that a job after
everything is finished retrains nothing."""
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pandas as pd
import pytest

from paper_fixtures import install_in_hf_cache, tiny_clip, tiny_xclip

REPO = Path(__file__).resolve().parents[1]
PESKA = os.environ.get('LGEL_PESKAVLP_PYTHON')
pytestmark = pytest.mark.skipif(not (os.environ.get('LGEL_PAPER_E2E') == '1' and PESKA),
                                reason='set LGEL_PAPER_E2E=1 and LGEL_PESKAVLP_PYTHON')

BERT_AND_WEIGHTS = r'''
import copy, os, shutil, sys, zipfile
from pathlib import Path
hf, out = Path(sys.argv[1]), Path(sys.argv[2])
from transformers import BertConfig, BertModel, BertTokenizer
tmp = out.parent / 'bert'
tmp.mkdir(parents=True, exist_ok=True)
letters = 'abcdefghijklmnopqrstuvwxyz0123456789'
vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + list(letters) + ['##' + c for c in letters] + list('.,-()/')
(tmp / 'vocab.txt').write_text('\n'.join(vocab) + '\n')
BertTokenizer(str(tmp / 'vocab.txt')).save_pretrained(tmp)
BertModel(BertConfig(vocab_size=len(vocab), hidden_size=768, intermediate_size=32, num_hidden_layers=4,
                     num_attention_heads=4, max_position_embeddings=128)).save_pretrained(tmp)
repo = hf / 'hub' / 'models--emilyalsentzer--Bio_ClinicalBERT'
shutil.copytree(tmp, repo / 'snapshots' / ('0' * 40))
(repo / 'refs').mkdir(parents=True)
(repo / 'refs' / 'main').write_text('0' * 40)
os.environ['HF_HOME'] = str(hf)
sys.path.insert(0, sys.argv[3])
import peska_runner, torch
from surgvlp.codes.models import build_algorithm
torch.manual_seed(0)
torch.save(build_algorithm(copy.deepcopy(peska_runner.MODEL_CONFIG)).state_dict(), out.parent / 'PeskaVLP.pth')
with zipfile.ZipFile(out, 'w') as z:                 # the shape of the official download: a zip with one .pth
    z.write(out.parent / 'PeskaVLP.pth', 'PeskaVLP.pth')
    z.writestr('__MACOSX/._PeskaVLP.pth', b'resource fork')
'''


def job(root, env, timeout=None, log=None):
    args = ['--preset', 'paper', '--root', str(root), '--synthetic', '8', '--random-init', '--allow-cpu', '--offline',
            '--single-job', '--splits-json', str(root.parent / 'split.json'), '--epochs', '2', '--warmup', '1',
            '--subset-ratio', '0.3', '--batch-size', '2', '--accum', '1', '--audit', 'sample', '--min-free-gb', '1',
            '--train-workers', '0', '--pred-batch-size', '8']
    with open(log, 'a') as out:
        out.write(f'\n\n######## job at {time.ctime()}\n')
        out.flush()
        proc = subprocess.Popen([sys.executable, str(REPO / 'main.py'), *args], env=env, cwd=REPO, stdout=out,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)             # what the scheduler does at the time limit
            return proc.wait(timeout=300)


def latest_report(root):
    name = (root / 'outbox' / 'LATEST.txt').read_text().strip()
    with tarfile.open(root / 'outbox' / name) as tar:
        return name, set(tar.getnames())


def test_the_paper_protocol_end_to_end(tmp_path):
    root = tmp_path / 'root'
    log = tmp_path / 'jobs.log'
    hf = root / 'hf_cache'
    clip = tiny_clip(tmp_path / 'clip')
    install_in_hf_cache(hf, 'openai/clip-vit-base-patch32', clip)
    install_in_hf_cache(hf, 'microsoft/xclip-base-patch32', tiny_xclip(tmp_path / 'xclip', frames=4))
    download = root / 'comparisons' / 'peskavlp' / 'PeskaVLP.download'
    download.parent.mkdir(parents=True)
    clean = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}
    subprocess.run([PESKA, '-c', BERT_AND_WEIGHTS, hf, download, REPO / 'comparison_models' / 'peskavlp'],
                   env=dict(clean, HF_HOME=str(hf), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'), check=True)
    (download.parent / f'{download.name}.complete').write_text(json.dumps(dict(size=download.stat().st_size)))
    import zipfile
    for model in ('SurgVLP', 'HecVL'):                  # same architecture: the same random state dict, own download
        other = root / 'comparisons' / model.lower() / f'{model}.download'
        other.parent.mkdir(parents=True)
        with zipfile.ZipFile(other, 'w') as z:
            z.write(download.parent / 'PeskaVLP.pth', f'{model}.pth')
        (other.parent / f'{other.name}.complete').write_text(json.dumps(dict(size=other.stat().st_size)))
    (tmp_path / 'split.json').write_text(json.dumps(dict(
        train=[f'CHOLEC80__video0{i}' for i in (1, 2, 3, 4)], val=['CHOLEC80__video05', 'CHOLEC80__video06'],
        test=['CHOLEC80__video07', 'CHOLEC80__video08'])))
    import torchvision.models.optical_flow as of
    flow = root / 'torch_cache' / 'hub' / 'checkpoints' / Path(of.Raft_Small_Weights.DEFAULT.url).name
    flow.parent.mkdir(parents=True)
    flow.write_bytes(b'not used: the test hook replaces the flow network')
    import torch
    import torchvision
    torch.save(torchvision.models.resnet50().state_dict(),               # TeCNO's ImageNet start (random here)
               flow.parent / Path(torchvision.models.ResNet50_Weights.IMAGENET1K_V1.url).name)
    sentinel = tmp_path / 'fail_univtg'
    sentinel.write_text('x')
    env = dict(clean, PYTHONPATH=str(REPO / 'tests' / 'e2e_hook'), LGEL_E2E_TINY='1', LGEL_E2E_FAIL_UNIVTG=str(sentinel),
               LGEL_PESKAVLP_PREFIX=str(Path(PESKA).parents[1]))
    names = [p['name'] for p in __import__('paper_protocol').plans('paper')]

    # 1. stopped in the middle, as by the time limit: a clean stop, with a report
    rc = job(root, env, timeout=420, log=log)
    assert rc == 143, log.read_text()[-3000:]
    name, files = latest_report(root)
    assert name.endswith('-STOPPED.tar.gz') and 'logs/main.log' in files

    # 2. the next job continues; the injected UniVTG failure does not stop the other runs
    rc = job(root, env, log=log)
    text = log.read_text()
    assert rc == 1, text[-3000:]
    assert 'run paper_univtg_s42 failed (stage train); continuing with the next run' in text
    finished = [n for n in names if (root / 'results' / n / 'proposed_test_metrics.json').exists()]
    assert finished == [n for n in names if n != 'paper_univtg_s42'], finished
    paper = root / 'paper'
    assert 'Not finished yet (left out below): paper_univtg_s42' in (paper / 'PAPER_RESULTS.md').read_text()
    name, files = latest_report(root)
    assert name.endswith('-FAILED.tar.gz') and 'paper/PAPER_RESULTS.md' in files

    # 3. the failure is gone: only UniVTG is trained now; everything else is already finished
    sentinel.unlink()
    before = {n: (root / 'runs' / n / 'training_metrics.jsonl').read_text()
              for n in names if (root / 'runs' / n / 'training_metrics.jsonl').exists()}
    rc = job(root, env, log=log)
    assert rc == 0, log.read_text()[-3000:]
    for n, content in before.items():
        assert (root / 'runs' / n / 'training_metrics.jsonl').read_text() == content, n
    assert all((root / 'results' / n / 'proposed_test_metrics.json').exists() for n in names)
    text = (paper / 'PAPER_RESULTS.md').read_text()
    assert 'Runs finished: **19 of 19**' in text
    report = json.loads((paper / 'analysis.json').read_text())
    assert 'figure_error' not in report and len(report['methods']) == 19
    for table in ('main_results', 'significance', 'segment_results', 'phase_recognition', 'phase_recognition_relaxed',
                  'tool_presence', 'ablations', 'seeds', 'generalisation', 'efficiency'):
        assert (paper / 'tables' / f'{table}.tex').exists(), table
    main = pd.read_csv(paper / 'tables' / 'main_results.csv')
    assert len(main) == 11
    phase = pd.read_csv(paper / 'tables' / 'phase_recognition.csv')
    assert sum("authors' phase prompts" in m for m in phase.Method) == 3
    seg = pd.read_csv(paper / 'tables' / 'segment_results.csv')
    assert any('own windows' in m for m in seg.Method)
    assert len(pd.read_csv(paper / 'tables' / 'generalisation.csv')) == 13       # every method with a text input
    eff = pd.read_csv(paper / 'tables' / 'efficiency.csv').set_index('Method')
    for label in ('TeCNO (re-implementation)', 'Our backbone without language (closed-set)'):
        assert eff.loc[label, 'Training (h)'] != 'n/a' and eff.loc[label, 'Trainable (M)'] != 'n/a', label
    for key in ('baseline_s42', 'advanced_s42'):
        done = json.loads((paper / 'explanations' / key / 'done.json').read_text())
        assert done['figures'] and (paper / 'explanations' / key / f'explanation_{done["figures"][0]}.png').exists()
    # every run's summary says what it is
    assert 'Method: **UniVTG (M2CRL features)**' in (root / 'results' / 'paper_univtg_s42' / 'SUMMARY.md').read_text()
    assert 'Method: **TeCNO (re-implementation)**' in (root / 'results' / 'paper_tecno_s42' / 'SUMMARY.md').read_text()
    assert 'Training (tecno_tcn)' in (root / 'results' / 'paper_tecno_s42' / 'SUMMARY.md').read_text()
    assert 'Training (closedset)' in (root / 'results' / 'paper_closedset_s42' / 'SUMMARY.md').read_text()
    assert 'Model: **baseline + Mamba temporal head**' in \
        (root / 'results' / 'paper_ablation_mamba_s42' / 'SUMMARY.md').read_text()
    assert 'Unseen query wordings' in (root / 'results' / 'paper_advanced_s43' / 'SUMMARY.md').read_text()
    # every method was evaluated on exactly the same frames with its validation threshold
    ref = pd.read_csv(root / 'results' / names[0] / 'test_reference.csv')
    for n in names:
        other = pd.read_csv(root / 'results' / n / 'test_reference.csv')
        assert other.equals(ref), n
        cal = json.loads((root / 'results' / n / 'proposed_calibration.json').read_text())
        assert cal['split'] == 'validation'
    name, files = latest_report(root)
    assert name.endswith('-OK.tar.gz')
    assert {'paper/PAPER_RESULTS.md', 'paper/tables/main_results.tex', 'paper/figures/reliability.png'} <= files
    assert any(f.startswith('paper/explanations/') and f.endswith('.png') for f in files)
    assert not [f for f in files if f.endswith(('.pth', '.pt', '.npz', '.npy'))]

    # 4. a job after everything has finished retrains and repredicts nothing
    stamp = {p: p.stat().st_mtime for p in (root / 'results').rglob('proposed_*.csv')}
    t0 = time.time()
    rc = job(root, env, log=log)
    assert rc == 0
    assert {p: p.stat().st_mtime for p in (root / 'results').rglob('proposed_*.csv')} == stamp
    print(f'job after completion took {time.time() - t0:.0f} s')
