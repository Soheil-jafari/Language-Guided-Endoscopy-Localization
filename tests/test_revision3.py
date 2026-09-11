"""Tests for the third repair round: backbone fidelity, decode tolerance, FPS snap,
bounded phase intervals, stride training windows, consistent augmentation,
order-aware temporal head, and the strict loader."""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from data_contract import AnnotationIndex, check_decoded_count, snap_fps, sample_indices


def test_snap_fps_only_near_integers():
    assert snap_fps(24.9999) == 25.0 and snap_fps(25.004) == 25.0 and snap_fps(25) == 25.0
    assert snap_fps(29.97) == 29.97 and snap_fps(23.976) == 23.976
    with pytest.raises(ValueError): snap_fps(0)


def test_snapped_fps_keeps_grid_on_tool_frames():
    # 24.9999 reported for a nominal 25 FPS video would drift off multiples of 25.
    drift = sample_indices(150_000, 24.9999, 1.)
    assert (drift % 25 != 0).any()
    assert (sample_indices(150_000, snap_fps(24.9999), 1.) % 25 == 0).all()


def test_decoded_count_tolerance():
    assert check_decoded_count(1499, 1500, 'v') == 1499       # small overestimate tolerated
    assert check_decoded_count(1510, 1500, 'v') == 1510       # underestimates are harmless
    with pytest.raises(OSError): check_decoded_count(1400, 1500, 'v')
    with pytest.raises(OSError): check_decoded_count(0, 10, 'v')


def test_phase_interval_ends_at_last_annotated_frame():
    df = pd.DataFrame([dict(standardized_video_id='v', frame_idx=0, original_label='Preparation', grasper=1),
                       dict(standardized_video_id='v', frame_idx=25, original_label='CalotTriangleDissection', grasper=0),
                       dict(standardized_video_id='v', frame_idx=75, grasper=1)])
    ann = AnnotationIndex(df)
    assert ann.label('v', 24, ('phase', 0)) == 1 and ann.label('v', 25, ('phase', 1)) == 1
    assert ann.label('v', 75, ('phase', 1)) == 1      # tool row at 75 extends the annotated range
    assert ann.label('v', 76, ('phase', 1)) == -100   # nothing is extrapolated past it
    assert ann.label('v', 76, ('phase', 0)) == -100


def _capture_with_reported_count(cv2, transform):
    """A VideoCapture wrapper (composition, never a subclass of the pybind type)
    whose CAP_PROP_FRAME_COUNT is deliberately wrong, as containers often are."""
    real = cv2.VideoCapture
    class Wrapped:
        def __init__(self, path): self._cap = real(path)
        def isOpened(self): return self._cap.isOpened()
        def read(self): return self._cap.read()
        def release(self): return self._cap.release()
        def get(self, prop):
            value = self._cap.get(prop)
            return float(transform(value)) if prop == cv2.CAP_PROP_FRAME_COUNT else value
    return Wrapped


def test_extractor_tolerates_container_count_overestimate(tmp_path, monkeypatch):
    import cv2
    from dataset_preprocessing import build_dataset
    source = tmp_path / 'videos'; source.mkdir(); video = source / 'video01.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 25., (32, 32)); assert writer.isOpened()
    for i in range(61): writer.write(np.full((32, 32, 3), i, dtype=np.uint8))
    writer.release()
    monkeypatch.setattr(cv2, 'VideoCapture', _capture_with_reported_count(cv2, lambda n: n + 2))
    build_dataset.extract(source, tmp_path / 'frames', tmp_path / 'meta.json', 1.)
    meta = json.loads((tmp_path / 'meta.json').read_text())['CHOLEC80__video01']
    assert meta['frame_count'] == 61 and meta['reported_frame_count'] == 63 and meta['source_fps'] == 25.
    assert sorted(p.name for p in (tmp_path / 'frames' / 'CHOLEC80__video01').glob('*.jpg')) == ['frame_0000000.jpg', 'frame_0000025.jpg', 'frame_0000050.jpg']
    from inference import read_sampled_video
    frames, ids, fps, duration, info = read_sampled_video(video, 1., 16)
    assert ids.tolist() == [0, 25, 50] and info['decoded_frame_count'] == 61 and duration == pytest.approx(61 / 25)


def test_extractor_removes_partial_output_on_truncated_decode(tmp_path, monkeypatch):
    import cv2
    from dataset_preprocessing import build_dataset
    source = tmp_path / 'videos'; source.mkdir(); video = source / 'video01.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 25., (32, 32)); assert writer.isOpened()
    for i in range(61): writer.write(np.full((32, 32, 3), i, dtype=np.uint8))
    writer.release()
    monkeypatch.setattr(cv2, 'VideoCapture', _capture_with_reported_count(cv2, lambda n: 1000))
    with pytest.raises(OSError, match='truncated'):
        build_dataset.extract(source, tmp_path / 'frames', tmp_path / 'meta.json', 1.)
    assert not (tmp_path / 'frames' / 'CHOLEC80__video01').exists() and not (tmp_path / 'meta.json').exists()


def _tiny_backbone(**kw):
    from backbone.vision_transformer import VisionTransformer
    return VisionTransformer(img_size=16, patch_size=8, embed_dim=16, depth=2, num_heads=4, num_frames=3, **kw)


def test_block_has_temporal_fc_and_timesformer_zero_init():
    model = _tiny_backbone()
    assert all(hasattr(b, 'temporal_fc') for b in model.blocks)
    assert model.blocks[1].temporal_fc.weight.abs().sum() == 0 and model.blocks[1].temporal_fc.bias.abs().sum() == 0
    assert model.blocks[0].temporal_fc.weight.abs().sum() > 0
    out = model.forward_features(torch.randn(1, 3, 3, 16, 16), get_all=True)
    assert out.shape == (1, 1 + 3 * 4, 16)
    out.sum().backward(); assert model.blocks[0].temporal_fc.weight.grad is not None


def test_pretrained_loader_remaps_time_embed_and_requires_exact_match(tmp_path):
    from backbone.vision_transformer import load_pretrained, _cfg
    torch.manual_seed(0); source = _tiny_backbone(); target = _tiny_backbone()
    sd = {k: v.clone() for k, v in source.state_dict().items()}
    sd['time_embed'] = sd.pop('temporal_embed')                          # TimeSformer naming
    sd['head.weight'] = torch.zeros(5, 16); sd['head.bias'] = torch.zeros(5)  # classification head is dropped
    sd['patch_embed.proj.weight'] = sd['patch_embed.proj.weight'][:, :, 0]  # 2D Conv weights are inflated
    path = tmp_path / 'ckpt.pth'; torch.save({'model': sd}, path)
    load_pretrained(target, cfg=_cfg(), num_classes=0, pretrained_model=str(path))
    for k, v in source.state_dict().items(): torch.testing.assert_close(target.state_dict()[k], v)
    # A checkpoint-only tensor is an error unless its prefix is declared.
    sd['projector.weight'] = torch.zeros(4, 16); torch.save({'model': sd}, path)
    with pytest.raises(RuntimeError, match='unexpected'):
        load_pretrained(_tiny_backbone(), cfg=_cfg(), num_classes=0, pretrained_model=str(path))
    load_pretrained(_tiny_backbone(), cfg=_cfg(), num_classes=0, pretrained_model=str(path),
                    ignore_unexpected_prefixes=('head.', 'projector.'))
    # A missing backbone tensor is an error unless explicitly allowed.
    del sd['projector.weight']; del sd['blocks.1.temporal_fc.weight']; del sd['blocks.1.temporal_fc.bias']
    torch.save({'model': sd}, path)
    with pytest.raises(RuntimeError, match='missing'):
        load_pretrained(_tiny_backbone(), cfg=_cfg(), num_classes=0, pretrained_model=str(path))
    load_pretrained(_tiny_backbone(), cfg=_cfg(), num_classes=0, pretrained_model=str(path),
                    allow_missing=('blocks.1.temporal_fc.weight', 'blocks.1.temporal_fc.bias'))
    with pytest.raises(ValueError):
        load_pretrained(_tiny_backbone(), cfg=_cfg(), num_classes=0, pretrained_model=str(path), strict=False)


def test_pretrained_loader_resizes_embeddings_from_other_clip_length_and_resolution(tmp_path):
    from backbone.vision_transformer import load_pretrained, _cfg, VisionTransformer
    source = VisionTransformer(img_size=24, patch_size=8, embed_dim=16, depth=2, num_heads=4, num_frames=8)
    sd = source.state_dict(); path = tmp_path / 'ckpt.pth'; torch.save(sd, path)
    target = _tiny_backbone()  # 16x16 grid (2x2 patches) and 3 frames
    load_pretrained(target, cfg=_cfg(), num_classes=0, pretrained_model=str(path))
    assert target.temporal_embed.shape == (1, 3, 16) and target.pos_embed.shape == (1, 1 + 4, 16)
    torch.testing.assert_close(target.pos_embed[:, :1], sd['pos_embed'][:, :1])


def test_pos_embed_tiling_skips_interpolation_when_grid_matches():
    model = _tiny_backbone()
    tiled = model._resize_pos_embed(model.pos_embed, 2, 2, 3)
    assert tiled.shape == (1, 1 + 3 * 4, 16)
    torch.testing.assert_close(tiled[:, 1:5], model.pos_embed[:, 1:])
    torch.testing.assert_close(tiled[:, 5:9], model.pos_embed[:, 1:])


def test_temporal_head_is_order_aware():
    from models import TemporalHead
    torch.manual_seed(0); head = TemporalHead(8, 1, num_attention_heads=2, num_layers=1, max_length=4).eval()
    x = torch.randn(1, 4, 8)
    assert not torch.allclose(head(x)[0, [1, 0, 2, 3]], head(x[:, [1, 0, 2, 3]])[0])
    with pytest.raises(ValueError): head(torch.randn(1, 5, 8))
    assert head(torch.randn(1, 2, 8)).shape == (1, 2, 1)


def test_train_and_eval_transforms_share_field_of_view():
    from dataset import image_transform
    torch.manual_seed(0)
    clip = torch.zeros(2, 3, 90, 160, dtype=torch.uint8); clip[:, :, :, :80] = 255  # left half white
    train = image_transform(32, True, (0.99, 1.0), (1.0, 1.0))(clip)
    val = image_transform(32, False)(clip)
    assert train.shape == val.shape == (2, 3, 32, 32)
    torch.testing.assert_close(train[0], train[1])  # one geometric draw per clip
    # With a full-frame crop the training view is the (possibly flipped) evaluation view,
    # up to resampling at the hard edge: the white half must occupy the same columns.
    white_train = (train[0, 0] > 0).float().mean(0); white_val = (val[0, 0] > 0).float().mean(0)
    assert white_val[:14].mean() > 0.9 and white_val[18:].mean() < 0.1
    assert (white_train - white_val).abs().mean() < 0.1 or (white_train - white_val.flip(0)).abs().mean() < 0.1
    with pytest.raises(ValueError): image_transform(32, True, (1.5, 2.0), (1, 1))


def test_tiny_framework_default_config_round_trip(monkeypatch):
    import models
    from project_config import Config
    from backbone.vision_transformer import VisionTransformer
    from transformers import CLIPTextModel, CLIPTextConfig
    from checkpoint_utils import config_dict, restore_config, load_model_state

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return dict(input_ids=torch.tensor([[1, 2, 3, 0]]), attention_mask=torch.tensor([[1, 1, 1, 0]]))
    cfg = Config(); cfg.DATA.TRAIN_CROP_SIZE = 16; cfg.DATA.NUM_FRAMES = 3; cfg.DATA.CLIP_LENGTH = 3
    cfg.MODEL.HEAD_NUM_ATTENTION_HEADS = 4; cfg.MODEL.HEAD_NUM_LAYERS = 1; cfg.TRAIN.DEVICE = 'cpu'; cfg.TRAIN.LORA_DROPOUT = 0
    monkeypatch.setattr(models, 'VisionTransformer', lambda **kw: VisionTransformer(img_size=16, patch_size=8, embed_dim=16, depth=1, num_heads=4, num_frames=3))
    monkeypatch.setattr(models.AutoTokenizer, 'from_pretrained', lambda *a, **k: Tokenizer())
    monkeypatch.setattr(models.CLIPTextModel, 'from_pretrained', lambda *a, **k: CLIPTextModel(CLIPTextConfig(vocab_size=10, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=8)))
    model = models.LocalizationFramework(cfg, initialize_backbone=False).eval()
    assert model.temporal_head.pos_embed.shape == (1, 3, 16)
    # The serialised configuration round-trips through the JSON form used by checkpoints.
    restored = restore_config(json.loads(json.dumps(config_dict(cfg))))
    second = models.LocalizationFramework(restored, initialize_backbone=False).eval()
    load_model_state(second, {'model_state_dict': model.state_dict()})
    video = torch.randn(1, 3, 3, 16, 16); ids = torch.tensor([[1, 2, 3, 0]]); mask = (ids != 0).long()
    torch.testing.assert_close(model(video, ids, mask)[0], second(video, ids, mask)[0])
    assert not hasattr(cfg.MODEL, 'EMBED_DIM') and not hasattr(cfg.DATA, 'NUM_INFERENCE_FRAMES')
