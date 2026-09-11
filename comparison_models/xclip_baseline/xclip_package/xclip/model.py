import torch.nn as nn
from transformers import XCLIPModel,XCLIPProcessor

class XCLIPWrapper(nn.Module):
    """Returns the HF video-by-text logits, including video-conditioned prompts."""
    def __init__(self,model_name='microsoft/xclip-base-patch32'):
        super().__init__()
        self.model=XCLIPModel.from_pretrained(model_name)
        self.processor=XCLIPProcessor.from_pretrained(model_name)
    def forward(self,**inputs):
        return self.model(**inputs).logits_per_video
