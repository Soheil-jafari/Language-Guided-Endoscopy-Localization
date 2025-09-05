import torch, torch.nn as nn
from transformers import RobertaModel
from .matcher import HungarianMatcher
from .transformer import Transformer
from .loss import SetCriterion
from .position_encoding import build_position_encoding


class MomentDETR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')
        self.video_proj = nn.Linear(cfg.v_feat_dim, cfg.hidden_dim)
        self.text_proj = nn.Linear(cfg.t_feat_dim, cfg.hidden_dim)
        self.pos_embed = build_position_encoding(cfg)
        self.transformer = Transformer(d_model=cfg.hidden_dim, nhead=cfg.nheads, num_encoder_layers=cfg.enc_layer,
                                       num_decoder_layers=cfg.dec_layer, dim_feedforward=cfg.dim_feedforward,
                                       dropout=cfg.dropout, normalize_before=cfg.pre_norm)
        self.query_embed = nn.Embedding(cfg.max_v_len, cfg.hidden_dim)

        # --- MODIFICATION START ---
        # We need separate heads for start and width prediction
        self.span_head = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU(),
                                       nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
        self.span_start_head = nn.Linear(cfg.hidden_dim, 1)
        self.span_width_head = nn.Linear(cfg.hidden_dim, 1)  # Changed from 'end' to 'width'
        # --- MODIFICATION END ---

        self.class_head = nn.Linear(cfg.hidden_dim, 2)
        matcher = HungarianMatcher(cost_span=cfg.l1_loss_coef, cost_giou=cfg.giou_loss_coef)
        self.criterion = SetCriterion(matcher,
                                      weight_dict={'loss_span': cfg.l1_loss_coef, 'loss_giou': cfg.giou_loss_coef,
                                                   'loss_ce': 1}, eos_coef=cfg.eos_coef)

    def forward(self, video_feats, video_mask, query, query_mask, targets=None):
        video_feats, query_feats = self.video_proj(video_feats), self.text_encoder(query, attention_mask=query_mask)[0]
        query_feats = self.text_proj(query_feats)
        pos_embed = self.pos_embed(video_feats, video_mask)
        hs = \
        self.transformer(video_feats, video_mask, self.query_embed.weight, pos_embed, query_feats, query_mask == 0)[0]

        # --- MODIFICATION START ---
        # This is the new, robust logic for span prediction
        hs_span = self.span_head(hs)
        pred_start = self.span_start_head(hs_span).sigmoid()
        pred_width = self.span_width_head(hs_span).sigmoid()

        # Guarantees that (start + width) is always <= 1.0
        pred_width = pred_width * (1 - pred_start)

        pred_center = pred_start + pred_width / 2
        # --- MODIFICATION END ---

        pred_spans = torch.stack([pred_center.squeeze(-1), pred_width.squeeze(-1)], dim=-1)
        pred_logits = self.class_head(hs)
        outputs = {'pred_logits': pred_logits[-1], 'pred_spans': pred_spans[-1]}
        if self.training: outputs['loss_dict'] = self.criterion(outputs, targets)
        return outputs

