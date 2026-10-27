"""The paper's experimental protocol: every run, in the order the `paper` preset trains them.

One command (`bash hpc/submit.sh paper ...`, or `python main.py --preset paper ...`) works through this
list. Each entry is a separate run with its own checkpoints, predictions and metrics, so an interrupted
job continues where it stopped and a finished run is never repeated. Everything shares one prepared
dataset and the standard Cholec80 split (40 training, 8 validation, 32 test videos).

Order = priority: the main comparison first (so the most important results exist early), then the
other methods, the ablations, and finally the extra seeds.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
SPLIT_FILE = REPO / 'protocol' / 'cholec80_split_40_8_32.json'
PARAPHRASE_FILE = REPO / 'protocol' / 'paraphrases.json'
SEEDS = (42, 43, 44)
MAIN_SEED = SEEDS[0]

# The four add-ons of the advanced model (names as in project_config.py). The advanced model is the
# baseline with all four switched on; each ablation switches on exactly one of them.
ADDONS = {
    'mamba': ('Mamba temporal head', dict(MODEL=dict(TEMPORAL_HEAD_TYPE='SSM', SSM_USE_OFFICIAL_MAMBA=True))),
    'bilevel': ('bi-level consistency loss', dict(TRAIN=dict(USE_BILEVEL_CONSISTENCY=True))),
    'uncertainty': ('evidential uncertainty', dict(MODEL=dict(USE_UNCERTAINTY=True))),
    'fusion': ('confidence fusion', dict(MODEL=dict(USE_CONFIDENCE_FUSION=True))),
}


def merge_flags(*parts):
    out = {}
    for part in parts:
        for section, values in part.items():
            out.setdefault(section, {}).update(copy.deepcopy(values))
    return out


ADVANCED = merge_flags(*(flags for _, flags in ADDONS.values()))

# Comparison methods (all evaluated exactly like the proposed model: same frames, same validation-selected
# threshold, same metrics). Settings are the methods' own published defaults where they exist.
COMPARISONS = {
    # General-domain image-text model, no training: lower bound.
    'clip_zeroshot': dict(label='CLIP zero-shot', model='openai/clip-vit-base-patch32'),
    # Surgical video-language model (CAMMA, NeurIPS 2024), no training. Weights: public direct link.
    'peskavlp_zeroshot': dict(label='PeskaVLP zero-shot'),
    # The same surgical model fine-tuned on our training videos (image encoder trained, text encoder
    # frozen, frame-wise binary objective, best epoch chosen on validation).
    'peskavlp_finetune': dict(label='PeskaVLP fine-tuned', epochs=5, lr=1e-5, batch_size=64),
    # General video-language model fine-tuned on our training videos (multi-positive contrastive).
    'xclip_finetune': dict(label='X-CLIP fine-tuned', model='microsoft/xclip-base-patch32', epochs=5,
                           batch_size=8, lr=1e-5),
    # Official UniVTG model (ICCV 2023, MIT licence) for temporal grounding, trained on the same
    # pretrained M2CRL video features our model starts from (feature-matched comparison).
    'univtg': dict(label='UniVTG (M2CRL features)', clip_length=4, epochs=200, eval_every=5, batch_size=8,
                   lr=1e-4),
}

PESKAVLP = dict(
    repository='https://github.com/CAMMA-public/SurgVLP.git',
    commit='85e858998eec47614614b89f6af1363e0ad3f47b',     # pinned (also in comparison_models/peskavlp/setup_env.sh)
    weights_url='https://seafile.unistra.fr/f/65a2b1bf113e428280d0/?dl=1',   # public link of the official weights
    licence='CC BY-NC-SA 4.0 (research use)',
)

# The authors' own Cholec80 phase prompts (SurgVLP repository, tests/class_prompt.txt at the pinned commit),
# in the order of data_contract.PHASES. Used once more for PeskaVLP zero-shot, so that its phase recognition
# can be checked against the numbers the authors report with exactly these prompts.
PESKAVLP_PHASE_PROMPTS = (
    'In preparation phase I insert trocars to patient abdomen cavity',
    'In calot triangle dissection phase I use grasper to hold gallbladder and use hook to expose the hepatic '
    'triangle area and cystic duct and cystic artery',
    'In clip and cut phase I use clipper to clip the cystic duct and artery then use scissor to cut them',
    'In dissection phase I use the hook to dissect the connective tissue between gallbladder and liver',
    'In packaging phase I put the gallbladder into the specimen bag',
    'In clean and coagulation phase I use suction and irrigation to clear the surgical field and coagulate '
    'bleeding vessels',
    'In retraction phase I grasp the specimen bag and remove it from trocar',
)


def canonical_queries():
    """The 14 training sentences (7 phases, then 7 tools), as in data_contract.PHRASES."""
    from data_contract import PHRASES
    return tuple(PHRASES)


def phase_queries():
    return canonical_queries()[:7]


def tool_queries():
    return canonical_queries()[7:]


def plans(prefix='paper'):
    """Every run of the paper protocol, in order. Keys: name, kind, flags, seed, paraphrases, group, label."""
    out = []

    def add(name, kind, group, label, seed=MAIN_SEED, flags=None, paraphrases=True):
        out.append(dict(name=f'{prefix}_{name}', kind=kind, group=group, label=label, seed=seed,
                        flags=copy.deepcopy(flags or {}), paraphrases=paraphrases))

    add(f'baseline_s{MAIN_SEED}', 'proposed', 'main', 'Baseline (ours)')
    add(f'advanced_s{MAIN_SEED}', 'proposed', 'main', 'Advanced (ours)', flags=ADVANCED)
    for key in ('clip_zeroshot', 'peskavlp_zeroshot'):
        add(key, key, 'comparison', COMPARISONS[key]['label'])
    for key in ('xclip_finetune', 'peskavlp_finetune', 'univtg'):
        add(f'{key}_s{MAIN_SEED}', key, 'comparison', COMPARISONS[key]['label'])
    for key, (label, flags) in ADDONS.items():
        add(f'ablation_{key}_s{MAIN_SEED}', 'proposed', 'ablation', f'Baseline + {label}', flags=flags,
            paraphrases=False)
    for seed in SEEDS[1:]:
        add(f'baseline_s{seed}', 'proposed', 'seed', 'Baseline (ours)', seed=seed)
        add(f'advanced_s{seed}', 'proposed', 'seed', 'Advanced (ours)', seed=seed, flags=ADVANCED)
    names = [p['name'] for p in out]
    assert len(names) == len(set(names)), names
    return out


def describe_flags(flags):
    """Human-readable model description from the add-on flags."""
    on = [label for key, (label, f) in ADDONS.items()
          if all(flags.get(s, {}).get(k) == v for s, values in f.items() for k, v in values.items())]
    if len(on) == len(ADDONS):
        return 'advanced (' + ', '.join(on) + ')'
    if not on:
        return 'baseline (Transformer temporal head)'
    return 'baseline + ' + ', '.join(on)


# --------------------------------------------------------------------------- unseen query wordings
def load_paraphrases(path=PARAPHRASE_FILE):
    """{set name: {(kind, concept_id): text}}, validated: every set covers all 14 concepts once, no
    sentence repeats a canonical training sentence, and none contradicts a known query's concept."""
    from data_contract import PHRASES, normalized, query_spec
    data = json.loads(Path(path).read_text())
    canonical = {normalized(q) for q in PHRASES}
    sets = {}
    seen_texts = set()
    for name, body in data['sets'].items():
        mapping = {}
        for q in body['queries']:
            key = (q['concept'], int(q['concept_id']))
            text = str(q['text']).strip()
            if key in mapping:
                raise ValueError(f'paraphrase set {name}: concept {key} appears twice')
            if not text or normalized(text) in canonical:
                raise ValueError(f'paraphrase set {name}: {text!r} is empty or a training sentence')
            if normalized(text) in seen_texts:
                raise ValueError(f'paraphrase {text!r} is used twice')
            if query_spec(text, key[0], key[1]) != key:        # also refuses a contradiction with a known query
                raise ValueError(f'paraphrase {text!r} does not map to {key}')
            seen_texts.add(normalized(text))
            mapping[key] = text
        expected = {(k, i) for k in ('phase', 'tool') for i in range(7)}
        if set(mapping) != expected:
            raise ValueError(f'paraphrase set {name} must cover all 14 concepts (missing {sorted(expected - set(mapping))})')
        sets[name] = mapping
    if not sets:
        raise ValueError('no paraphrase sets')
    return sets


def paraphrase_triplet_name(set_name):
    return f'cholec80_test_para{set_name}_triplets.csv'


def write_paraphrase_triplets(test_csv, out_dir, path=PARAPHRASE_FILE):
    """Copies of the test manifest in which every canonical query is replaced by its unseen wording
    (same frames, same labels: the concept metadata travels with the row)."""
    import pandas as pd
    from data_contract import query_spec
    test = pd.read_csv(test_csv)
    written = {}
    for name, mapping in load_paraphrases(path).items():
        df = test.copy()
        specs = [query_spec(r['text_query'], r.get('query_kind'), r.get('concept_id')) for r in df.to_dict('records')]
        df['text_query'] = [mapping[s] for s in specs]
        df['query_kind'] = [s[0] for s in specs]
        df['concept_id'] = [s[1] for s in specs]
        out = Path(out_dir) / paraphrase_triplet_name(name)
        df.to_csv(out, index=False)
        written[name] = out
    return written


def paraphrase_signature(path=PARAPHRASE_FILE):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
