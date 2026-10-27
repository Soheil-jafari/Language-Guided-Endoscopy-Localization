"""SurgVLP, HecVL and PeskaVLP in their own environment, with the official SurgVLP code (installed by
comparison_models/peskavlp/setup_env.sh). Runs only when that environment exists:

    LGEL_PESKAVLP_PYTHON=/path/to/envs/lgel-peskavlp/bin/python python -m pytest tests/test_peskavlp_env.py

No network: the PeskaVLP weights are a random state dict of the official architecture (built by the official
code), and Bio_ClinicalBERT is replaced by a tiny 768-d BERT placed in a private Hugging Face cache under its
real name. Everything else - model construction, strict loading, the authors' preprocessing and tokenizer
calls, scoring, fine-tuning, resuming - is the real code path of the paper run."""
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from data_contract import PHRASES
from paper_fixtures import make_prepared

REPO = Path(__file__).resolve().parents[1]
PY = os.environ.get('LGEL_PESKAVLP_PYTHON')
pytestmark = pytest.mark.skipif(not PY, reason='set LGEL_PESKAVLP_PYTHON to the PeskaVLP environment')

MAKE_WEIGHTS = r'''
import json, sys, shutil
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
snap = repo / 'snapshots' / ('0' * 40)
shutil.copytree(tmp, snap)
(repo / 'refs').mkdir(parents=True)
(repo / 'refs' / 'main').write_text('0' * 40)
import os
os.environ['HF_HOME'] = str(hf)
sys.path.insert(0, sys.argv[3])
import peska_runner, copy, torch
from surgvlp.codes.models import build_algorithm
torch.manual_seed(0)
model = build_algorithm(copy.deepcopy(peska_runner.MODEL_CONFIG))
torch.save(model.state_dict(), out)
print('weights', sum(v.numel() for v in model.state_dict().values()))
'''


def peska(args, hf, check=True):
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}
    env.update(HF_HOME=str(hf), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONNOUSERSITE='1')
    return subprocess.run([PY, str(REPO / 'comparison_models' / 'peskavlp' / 'peska_runner.py'), *map(str, args)],
                          env=env, cwd=REPO, capture_output=True, text=True, check=check)


def test_peskavlp_checks_scores_finetunes_and_resumes(tmp_path):
    hf = tmp_path / 'hf'
    weights = tmp_path / 'w' / 'PeskaVLP.pth'
    weights.parent.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}
    env.update(HF_HOME=str(hf), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONNOUSERSITE='1')
    subprocess.run([PY, '-c', MAKE_WEIGHTS, hf, weights, REPO / 'comparison_models' / 'peskavlp'], env=env,
                   check=True, capture_output=True, text=True)
    # 1. check: strict loading of the official architecture, text and image towers
    out = peska(['check', '--weights', weights, '--device', 'cpu'], hf)
    assert 'PeskaVLP OK' in out.stdout
    import torch
    state = torch.load(weights, weights_only=False)
    state.pop(next(iter(state)))
    torch.save(state, tmp_path / 'broken.pth')
    bad = peska(['check', '--weights', tmp_path / 'broken.pth', '--device', 'cpu'], hf, check=False)
    assert bad.returncode != 0 and 'Missing key' in bad.stderr               # strict: never a partial model

    d = make_prepared(tmp_path / 'data', n_videos=4, seconds=10)
    from comparison_models import frame_scores as fs
    orders = {}
    for split in ('train', 'val', 'test'):
        orders[split] = fs.write_work_order(d['triplets'](split), d['metadata'], d['frames'], d['annotations'],
                                            tmp_path / f'{split}.json', labels=tmp_path / f'{split}.npz' if split != 'test' else None)
    # 2. zero-shot scores -> the standard prediction file
    peska(['score', '--work-order', tmp_path / 'test.json', '--weights', weights, '--output', tmp_path / 'zs.npz',
           '--device', 'cpu', '--workers', '0', '--batch-size', '4'], hf)
    df = fs.assemble(tmp_path / 'zs.npz', d['triplets']('test'), d['metadata'], d['annotations'], tmp_path / 'zs.csv')
    assert set(df.text_query) == set(PHRASES) and df.score.between(0, 1).all()
    man = json.loads((tmp_path / 'zs.npz.manifest.json').read_text())
    assert man['model'] == 'PeskaVLP zero-shot' and man['frames'] == len(df) // 14
    # 3. fine-tuning: two epochs, then a third continues from latest.pt with identical settings
    ft = tmp_path / 'ft'
    base = ['finetune', '--train-order', tmp_path / 'train.json', '--train-labels', tmp_path / 'train.npz',
            '--val-order', tmp_path / 'val.json', '--val-labels', tmp_path / 'val.npz', '--weights', weights,
            '--output', ft, '--lr', '1e-4', '--batch-size', '4', '--seed', '3', '--workers', '0', '--device', 'cpu']
    peska(base + ['--epochs', '2'], hf)
    rows = [json.loads(l) for l in (ft / 'training_metrics.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in rows] == [1, 2] and all(np.isfinite(r['val_loss']) for r in rows)
    refused = peska(base + ['--epochs', '3'], hf, check=False)               # epochs are part of the signature
    assert refused.returncode != 0 and 'different settings' in refused.stderr
    peska(base + ['--epochs', '2'], hf)                                       # finished: a no-op
    assert len((ft / 'training_metrics.jsonl').read_text().splitlines()) == 2
    # 4. scores of the fine-tuned model, and the checkpoint must match the weights
    peska(['score', '--work-order', tmp_path / 'test.json', '--weights', weights, '--checkpoint', ft / 'best.pt',
           '--output', tmp_path / 'ft.npz', '--device', 'cpu', '--workers', '0'], hf)
    df = fs.assemble(tmp_path / 'ft.npz', d['triplets']('test'), d['metadata'], d['annotations'], tmp_path / 'ft.csv')
    assert df.score.between(0, 1).all() and df.score.std() > 0
    other = tmp_path / 'other.pth'
    torch.save({k: v + 1 if v.is_floating_point() else v for k, v in torch.load(weights, weights_only=False).items()}, other)
    mismatch = peska(['score', '--work-order', tmp_path / 'test.json', '--weights', other, '--checkpoint',
                      ft / 'best.pt', '--output', tmp_path / 'x.npz', '--device', 'cpu', '--workers', '0'], hf, check=False)
    assert mismatch.returncode != 0 and 'does not match' in mismatch.stderr
    # 5. the authors' phase prompts are scored in place of the phase sentences
    import paper_protocol as protocol
    prompts = dict(zip(PHRASES[:7], protocol.PESKAVLP_PHASE_PROMPTS))
    fs.write_work_order(d['triplets']('test'), d['metadata'], d['frames'], d['annotations'], tmp_path / 'ap.json',
                        prompts=prompts)
    peska(['score', '--work-order', tmp_path / 'ap.json', '--weights', weights, '--output', tmp_path / 'ap.npz',
           '--device', 'cpu', '--workers', '0'], hf)
    z = np.load(tmp_path / 'ap.npz')
    assert set(map(str, z['queries'])) == set(protocol.PESKAVLP_PHASE_PROMPTS) | set(PHRASES[7:])
    df = fs.assemble(tmp_path / 'ap.npz', d['triplets']('test'), d['metadata'], d['annotations'], tmp_path / 'ap.csv',
                     prompts=prompts)
    assert set(df.text_query) == set(PHRASES)
    # 6. SurgVLP and HecVL: the same package and architecture (their own model classes, the same state-dict
    #    layout), selected with --model; a fine-tuned checkpoint is only accepted for the model it was made from
    for model in ('SurgVLP', 'HecVL'):
        assert f'{model} OK' in peska(['check', '--weights', weights, '--device', 'cpu', '--model', model], hf).stdout
        peska(['score', '--work-order', tmp_path / 'test.json', '--weights', weights, '--model', model, '--output',
               tmp_path / f'{model}.npz', '--device', 'cpu', '--workers', '0', '--batch-size', '4'], hf)
        assert json.loads((tmp_path / f'{model}.npz.manifest.json').read_text())['model'] == f'{model} zero-shot'
        wrong = peska(['score', '--work-order', tmp_path / 'test.json', '--weights', weights, '--model', model,
                       '--checkpoint', ft / 'best.pt', '--output', tmp_path / 'y.npz', '--device', 'cpu', '--workers',
                       '0'], hf, check=False)
        assert wrong.returncode != 0 and f'does not match these {model} weights' in wrong.stderr
    same = np.load(tmp_path / 'SurgVLP.npz')['scores']                 # same weights here: the same numbers
    np.testing.assert_allclose(same, np.load(tmp_path / 'zs.npz')['scores'], rtol=1e-5, atol=1e-6)
    unknown = peska(['check', '--weights', weights, '--device', 'cpu', '--model', 'CLIP'], hf, check=False)
    assert unknown.returncode != 0
