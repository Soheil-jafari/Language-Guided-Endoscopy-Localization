"""TEST-ONLY stand-in for the compiled mamba_ssm package (CPU, synthetic end-to-end test). Not Mamba."""
import torch

__version__ = '0.0-e2e-stub'


class Mamba(torch.nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.proj = torch.nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.proj(x)
