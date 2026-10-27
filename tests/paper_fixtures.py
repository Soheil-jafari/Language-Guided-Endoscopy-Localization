"""Small, fully synthetic inputs for the paper-protocol tests: prepared data in the pipeline's layout
(frames, metadata, parsed annotations, splits, manifests made by the real build_dataset.build) and tiny
Hugging Face models saved under the names the pipeline asks for (no network)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

TOOL_COLUMNS = ('Grasper', 'Bipolar', 'Hook', 'Scissors', 'Clipper', 'Irrigator', 'SpecimenBag')
PHASE_NAMES = ('Preparation', 'CalotTriangleDissection', 'ClippingCutting', 'GallbladderDissection',
               'GallbladderPackaging', 'CleaningCoagulation', 'GallbladderRetraction')


def videos(n):
    return [f'CHOLEC80__video{i:02d}' for i in range(1, n + 1)]


def make_prepared(root, n_videos=6, seconds=30, fps=25, seed=0, split=None, size=(24, 16)):
    """<root>/data/... as the pipeline's data stages leave it. Returns a dict of paths."""
    from data_contract import sample_indices
    from dataset_preprocessing.build_dataset import build
    root = Path(root)
    data = root / 'data'
    frames = data / 'frames'
    rng = np.random.default_rng(seed)
    meta, rows = {}, []
    names = videos(n_videos)
    for v in names:
        n = seconds * fps
        meta[v] = dict(source_fps=float(fps), frame_count=n)
        cuts = np.sort(rng.choice(np.arange(2, seconds - 2), 6, replace=False)) * fps
        bounds = [0, *cuts.tolist(), n]
        phase_of = np.zeros(n, dtype=int)
        for i in range(7):
            phase_of[bounds[i]:bounds[i + 1]] = i
        tools = rng.random((n, 7)) < 0.3
        for idx in range(0, n, fps):                       # 1-fps annotation rows, like the real release
            rows.append(dict(standardized_video_id=v, frame_idx=idx, original_label=PHASE_NAMES[phase_of[idx]],
                             **{t: int(tools[idx, k]) for k, t in enumerate(TOOL_COLUMNS)}))
        folder = frames / v
        folder.mkdir(parents=True, exist_ok=True)
        for idx in sample_indices(n, fps, 1.0):
            colour = (30 * phase_of[idx], 120 + 15 * int(tools[idx, 0]), 80 + 20 * int(tools[idx, 4]))
            Image.new('RGB', size, colour).save(folder / f'frame_{idx:07d}.jpg')
    data.mkdir(parents=True, exist_ok=True)
    (data / 'video_metadata.json').write_text(json.dumps(meta))
    pd.DataFrame(rows).to_csv(data / 'parsed_annotations.csv', index=False)
    if split is None:
        k = max(1, n_videos // 3)
        split = dict(train=names[:n_videos - 2 * k], val=names[n_videos - 2 * k:n_videos - k], test=names[n_videos - k:])
    (data / 'splits.json').write_text(json.dumps(split))
    build(data / 'parsed_annotations.csv', data / 'video_metadata.json', frames, data / 'splits.json',
          data / 'triplets', 1.0)
    return dict(root=root, data=data, frames=frames, metadata=data / 'video_metadata.json',
                annotations=data / 'parsed_annotations.csv', splits=data / 'splits.json', split=split,
                triplets=lambda s: data / 'triplets' / f'cholec80_{s}_triplets.csv')


# --------------------------------------------------------------------------- tiny Hugging Face models
def byte_level_vocab():
    from transformers.models.clip.tokenization_clip import bytes_to_unicode
    chars = list(bytes_to_unicode().values())
    vocab = chars + [c + '</w>' for c in chars] + ['<|startoftext|>', '<|endoftext|>']
    return {t: i for i, t in enumerate(vocab)}


def tiny_clip_tokenizer(folder):
    """A valid CLIP tokenizer whose BPE has no merges (every character is a token)."""
    from transformers import CLIPTokenizer
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'vocab.json').write_text(json.dumps(byte_level_vocab()))
    (folder / 'merges.txt').write_text('#version: 0.2\n')
    return CLIPTokenizer(str(folder / 'vocab.json'), str(folder / 'merges.txt'))


def tiny_clip(folder, text_hidden=512, positions=128):
    """openai/clip-vit-base-patch32 stand-in: CLIP with a 512-d text tower (as the real one), tiny otherwise."""
    from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel, CLIPProcessor
    folder = Path(folder)
    tok = tiny_clip_tokenizer(folder / '_tok')
    vocab = len(tok)
    cfg = CLIPConfig(text_config=dict(vocab_size=vocab, hidden_size=text_hidden, intermediate_size=64,
                                      num_hidden_layers=1, num_attention_heads=8, max_position_embeddings=positions,
                                      bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                                      pad_token_id=tok.pad_token_id),
                     vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
                                        image_size=32, patch_size=16),
                     projection_dim=32)
    CLIPModel(cfg).save_pretrained(folder)
    proc = CLIPProcessor(image_processor=CLIPImageProcessor(size={'shortest_edge': 32}, crop_size={'height': 32, 'width': 32}),
                         tokenizer=tok)
    proc.save_pretrained(folder)
    return folder


def tiny_xclip(folder, frames=4, positions=128):
    from transformers import VideoMAEImageProcessor, XCLIPConfig, XCLIPModel, XCLIPProcessor
    folder = Path(folder)
    tok = tiny_clip_tokenizer(folder / '_tok')
    cfg = XCLIPConfig(text_config=dict(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                       num_attention_heads=4, max_position_embeddings=positions,
                                       bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                                       pad_token_id=tok.pad_token_id),
                      vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
                                         image_size=32, patch_size=16, num_frames=frames, mit_hidden_size=32,
                                         mit_intermediate_size=64, mit_num_hidden_layers=1, mit_num_attention_heads=4),
                      projection_dim=32, prompt_layers=1, prompt_num_attention_heads=4)
    XCLIPModel(cfg).save_pretrained(folder)
    proc = XCLIPProcessor(image_processor=VideoMAEImageProcessor(size={'shortest_edge': 32},
                                                                 crop_size={'height': 32, 'width': 32}), tokenizer=tok)
    proc.save_pretrained(folder)
    return folder


def tiny_bert(folder, layers=4):
    """emilyalsentzer/Bio_ClinicalBERT stand-in: 768-d (PeskaVLP's text embedding size), otherwise tiny."""
    from transformers import BertConfig, BertModel, BertTokenizer
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    letters = 'abcdefghijklmnopqrstuvwxyz0123456789'
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + list(letters) + ['##' + c for c in letters] + list('.,-()/')
    (folder / 'vocab.txt').write_text('\n'.join(vocab) + '\n')
    BertTokenizer(str(folder / 'vocab.txt')).save_pretrained(folder)
    BertModel(BertConfig(vocab_size=len(vocab), hidden_size=768, intermediate_size=32, num_hidden_layers=layers,
                         num_attention_heads=4, max_position_embeddings=128)).save_pretrained(folder)
    return folder


def install_in_hf_cache(hf_home, repo_id, folder, revision='0' * 40):
    """Make `folder` what from_pretrained(repo_id) finds offline under HF_HOME (the hub cache layout)."""
    import shutil
    repo = Path(hf_home) / 'hub' / ('models--' + repo_id.replace('/', '--'))
    snap = repo / 'snapshots' / revision
    if snap.exists():
        shutil.rmtree(snap)
    snap.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(folder, snap, ignore=shutil.ignore_patterns('_tok'))
    (repo / 'refs').mkdir(parents=True, exist_ok=True)
    (repo / 'refs' / 'main').write_text(revision)
    return snap
