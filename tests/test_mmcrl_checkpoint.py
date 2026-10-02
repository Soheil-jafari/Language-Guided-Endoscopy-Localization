"""MMCRL train_ssl.py checkpoint layout: {'student': module.backbone.*/module.head.*,
'teacher': backbone.*/head.*, 'optimizer', 'epoch', 'args', 'dino_loss', ...}."""
import pytest
import torch


def _tiny(**kw):
    from backbone.vision_transformer import VisionTransformer
    return VisionTransformer(img_size=16, patch_size=8, embed_dim=16, depth=2, num_heads=4, num_frames=3, **kw)


def _mmcrl_ckpt(enc):
    bb = {k: v.clone() for k, v in enc.state_dict().items()}
    bb['time_embed'] = bb.pop('temporal_embed')
    bb['patch_embed.proj.weight'] = bb['patch_embed.proj.weight'][:, :, 0]      # Conv2d in MMCRL
    bb['masked_embed'] = torch.zeros(1, 16)                                     # pretraining-only
    bb['decoder.0.weight'] = torch.zeros(8 * 8 * 3, 16, 1, 1); bb['decoder.0.bias'] = torch.zeros(8 * 8 * 3)
    head = {'mlp.0.weight': torch.zeros(32, 16), 'mlp.0.bias': torch.zeros(32),
            'last_layer.weight_g': torch.ones(64, 1), 'last_layer.weight_v': torch.zeros(64, 8)}
    teacher = {**{f'backbone.{k}': v for k, v in bb.items()}, **{f'head.{k}': v for k, v in head.items()}}
    student = {f'module.{k}': v + 1 for k, v in teacher.items()}
    return {'student': student, 'teacher': teacher, 'motion_student': 0, 'motion_teacher': 0,
            'optimizer': {'state': {}, 'param_groups': []}, 'epoch': 30, 'args': None,
            'dino_loss': {'center': torch.zeros(1, 64)}}


def test_mmcrl_training_checkpoint_loads_teacher_exactly(tmp_path):
    from backbone.vision_transformer import load_pretrained, _cfg
    torch.manual_seed(0); src = _tiny(); dst = _tiny()
    ck = _mmcrl_ckpt(src); path = tmp_path / 'checkpoint.pth'; torch.save(ck, path)
    load_pretrained(dst, cfg=_cfg(), num_classes=0, pretrained_model=str(path))
    for k, v in src.state_dict().items():
        torch.testing.assert_close(dst.state_dict()[k], v)                      # teacher, not student (+1)


def test_mmcrl_checkpoint_stays_strict(tmp_path):
    from backbone.vision_transformer import load_pretrained, _cfg
    torch.manual_seed(0); ck = _mmcrl_ckpt(_tiny()); path = tmp_path / 'c.pth'
    ck['teacher']['backbone.decoder.1.weight'] = torch.zeros(3); torch.save(ck, path)   # unknown extra tensor
    with pytest.raises(RuntimeError, match='unexpected'):
        load_pretrained(_tiny(), cfg=_cfg(), num_classes=0, pretrained_model=str(path))
    del ck['teacher']['backbone.decoder.1.weight']; del ck['teacher']['backbone.blocks.1.temporal_fc.bias']
    torch.save(ck, path)
    with pytest.raises(RuntimeError, match='missing'):
        load_pretrained(_tiny(), cfg=_cfg(), num_classes=0, pretrained_model=str(path))
    ck = _mmcrl_ckpt(_tiny()); ck['teacher']['backbone.norm.weight'] = torch.zeros(17); torch.save(ck, path)
    with pytest.raises(RuntimeError, match='shape mismatch'):
        load_pretrained(_tiny(), cfg=_cfg(), num_classes=0, pretrained_model=str(path))
