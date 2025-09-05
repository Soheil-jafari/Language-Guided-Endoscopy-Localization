\
import torch
import torch.nn.functional as F

def clip_style_infonce(video_embeds, text_embeds, temperature=1.0):
    """
    Symmetric InfoNCE (like CLIP): cross-entropy over similarities in both directions.
    video_embeds: (B, D) L2-normalized
    text_embeds:  (B, D) L2-normalized
    Returns loss scalar.
    """
    logits = (video_embeds @ text_embeds.t()) / temperature  # (B, B)
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_v2t = F.cross_entropy(logits, targets)
    loss_t2v = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_v2t + loss_t2v)
