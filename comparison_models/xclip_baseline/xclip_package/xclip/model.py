# Save this file as: xclip/model.py

import torch
import torch.nn as nn
from transformers import XCLIPProcessor, XCLIPModel

class XCLIPWrapper(nn.Module):
    """A wrapper to handle the model and processor together."""
    def __init__(self, model_name: str = "microsoft/xclip-base-patch32"):
        super().__init__()
        self.model = XCLIPModel.from_pretrained(model_name)
        self.processor = XCLIPProcessor.from_pretrained(model_name)

    def forward(self, **inputs):
        """
        The forward pass of the model.
        ** This is the line we are changing. **
        Instead of returning the whole output object, we explicitly return
        only the logits tensor that we need for the calculation.
        """

        outputs = self.model(**inputs)
        # Select the first token's output, which represents the whole sentence
        text_embeds_pooled = outputs.text_embeds[:, 0]
        return outputs.video_embeds, text_embeds_pooled