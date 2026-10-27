"""TEST-ONLY start-up hook for the synthetic end-to-end run of the paper protocol (tests/test_paper_e2e.py).

It is put on PYTHONPATH by that test and does nothing unless LGEL_E2E_TINY=1. It shrinks the model so that
the whole protocol runs on a CPU in minutes - 32-pixel crops, 4-frame clips, a 1-block backbone (still
768-d), a smaller PeskaVLP fine-tuning batch (RAM) - and replaces the two GPU-only/network-only parts: the optical-flow network (zero flow) and the
compiled Mamba kernels (mamba_ssm stub in this folder). Everything else is the real code. With the file
named by LGEL_E2E_FAIL_UNIVTG present, the official UniVTG model refuses to import (an injected failure of
one run, to test that the protocol continues and retries it).
"""
import importlib.abc
import os
import sys

if os.environ.get('LGEL_E2E_TINY') == '1':

    def _config(module):
        c = module.config
        c.DATA.TRAIN_CROP_SIZE, c.DATA.CLIP_LENGTH, c.DATA.NUM_FRAMES = 32, 4, 4
        c.MODEL.HEAD_NUM_LAYERS = 1

    def _models(module):
        original = module.VisionTransformer

        def tiny(**kw):
            kw['depth'] = 1
            return original(**kw)
        module.VisionTransformer = tiny

    def _flow(module):
        import torch

        class ZeroFlow(torch.nn.Module):
            def forward(self, a, b):
                return [torch.zeros(a.shape[0], 2, a.shape[-2], a.shape[-1], device=a.device)]
        module.raft_small = lambda *a, **k: ZeroFlow()

    def _univtg(module):
        sentinel = os.environ.get('LGEL_E2E_FAIL_UNIVTG')
        if sentinel and os.path.exists(sentinel):
            raise RuntimeError('injected failure (test): UniVTG unavailable')

    def _protocol(module):
        module.COMPARISONS['peskavlp_finetune']['batch_size'] = 8      # ResNet-50 training on a small CPU machine

    PATCHES = {'project_config': _config, 'models': _models, 'torchvision.models.optical_flow': _flow,
               'model.univtg': _univtg, 'paper_protocol': _protocol}

    class _Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in PATCHES:
                return None
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, 'find_spec'):
                    continue
                spec = finder.find_spec(name, path, target)
                if spec is not None:
                    break
            else:
                return None
            run = spec.loader.exec_module

            def exec_module(module, _run=run, _patch=PATCHES[name]):
                _run(module)
                _patch(module)
            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Hook())
