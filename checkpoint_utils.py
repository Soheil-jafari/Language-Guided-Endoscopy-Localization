"""Checkpoint container handling shared by inference and exact resume."""
from collections.abc import Mapping
import hashlib
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

SCHEMA_VERSION = 2


def config_dict(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [config_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: config_dict(v) for k,v in obj.items()}
    return {k: config_dict(v) for k,v in vars(obj).items() if not k.startswith('_')}


def restore_config(value):
    return SimpleNamespace(**{k: restore_config(v) if isinstance(v, dict) else v for k,v in value.items()})


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    try:
        torch.save(value, temp)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Exact resume requires the same CUDA device count')
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def read_checkpoint(path, require_metadata=True):
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    # Full resumes include Python/NumPy RNG state. Only load trusted local checkpoints.
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if require_metadata and (not isinstance(ckpt, Mapping) or ckpt.get('schema_version') != SCHEMA_VERSION or 'config' not in ckpt):
        raise ValueError('Legacy checkpoint lacks the repaired data/model contract. Use an explicit weights-only fine-tune; do not resume or report it as a repaired run.')
    return ckpt


def extract_model_state(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise ValueError('Expected a state dictionary or a checkpoint mapping.')
    state = checkpoint
    for key in ('model_state_dict', 'state_dict', 'model'):
        if key in checkpoint and isinstance(checkpoint[key], Mapping):
            state = checkpoint[key]
            break
    if not state or not all(isinstance(k, str) and hasattr(v, 'shape') for k, v in state.items()):
        raise ValueError('Checkpoint does not contain a nonempty tensor state dictionary.')
    result = {}
    for key, value in state.items():
        name = key[7:] if key.startswith('module.') else key
        if name in result:
            raise ValueError(f'Duplicate checkpoint key after prefix removal: {name}')
        result[name] = value
    return result


def load_model_state(model, checkpoint):
    """Require a complete architecture match; partial loads are not evaluation."""
    core = getattr(model, 'module', model)
    return core.load_state_dict(extract_model_state(checkpoint), strict=True)
