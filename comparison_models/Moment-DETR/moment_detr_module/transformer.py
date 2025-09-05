import copy, torch, torch.nn as nn, torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=1024, dropout=0.1, normalize_before=True):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, "relu", normalize_before)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, nn.LayerNorm(d_model) if normalize_before else None)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, "relu", normalize_before)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, nn.LayerNorm(d_model), return_intermediate=True)
        self._reset_parameters()
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
    def forward(self, src, mask, query_embed, pos_embed, text_src, text_mask):
        bs, l, c = src.shape
        src, pos_embed, query_embed, text_src = src.permute(1, 0, 2), pos_embed.permute(1, 0, 2), query_embed.unsqueeze(1).repeat(1, bs, 1), text_src.permute(1, 0, 2)
        memory = self.encoder(src, src_key_padding_mask=~mask, pos=pos_embed, text_memory=text_src, text_memory_key_padding_mask=text_mask)
        hs = self.decoder(torch.zeros_like(query_embed), memory, memory_key_padding_mask=~mask, pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 0, 2)

class TransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers, norm=None): super().__init__(); self.layers, self.num_layers, self.norm = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)]), num_layers, norm
    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None, text_memory=None, text_memory_key_padding_mask=None):
        output = src
        for layer in self.layers: output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos, text_memory=text_memory, text_memory_key_padding_mask=text_memory_key_padding_mask)
        if self.norm is not None: output = self.norm(output)
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, layer, num_layers, norm=None, return_intermediate=False): super().__init__(); self.layers, self.num_layers, self.norm, self.return_intermediate = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)]), num_layers, norm, return_intermediate
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        output, intermediate = tgt, []
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, pos=pos, query_pos=query_pos)
            if self.return_intermediate: intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate: intermediate.pop(); intermediate.append(output)
        if self.return_intermediate: return torch.stack(intermediate)
        return output.unsqueeze(0)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation="relu", normalize_before=True):
        super().__init__()
        self.self_attn, self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout), nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1, self.dropout, self.linear2 = nn.Linear(d_model, dim_feedforward), nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout1, self.dropout2, self.dropout3 = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(dropout)
        self.activation, self.normalize_before = F.relu, normalize_before
    def with_pos(self, t, p): return t if p is None else t + p

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None, text_memory=None,
                text_memory_key_padding_mask=None):
        if self.normalize_before:
            src_norm = self.norm1(src)
            q = k = self.with_pos(src_norm, pos)
            src2 = self.self_attn(q, k, value=src_norm, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)

            src_norm = self.norm3(src)  # norm for cross-attention
            src2 = self.cross_attn(self.with_pos(src_norm, pos), text_memory, text_memory,
                                   key_padding_mask=text_memory_key_padding_mask)[0]
            src = src + self.dropout3(src2)

            src_norm = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
            src = src + self.dropout2(src2)
            return src
        else:  # The original implementation was post-normalization
            src2 = self.self_attn(self.with_pos(src, pos), self.with_pos(src, pos), src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
            src = self.norm1(src + self.dropout1(src2))
            src2 = self.cross_attn(self.with_pos(src, pos), text_memory, text_memory,
                                   key_padding_mask=text_memory_key_padding_mask)[0]
            src = self.norm3(src + self.dropout(src2))
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            return self.norm2(src + self.dropout2(src2))

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation="relu", normalize_before=True):
        super().__init__()
        self.self_attn, self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout), nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1, self.dropout, self.linear2 = nn.Linear(d_model, dim_feedforward), nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout1, self.dropout2, self.dropout3 = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(dropout)
        self.activation, self.normalize_before = F.relu, normalize_before
    def with_pos(self, t, p): return t if p is None else t + p

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, pos=None, query_pos=None):
        if self.normalize_before:
            tgt_norm = self.norm1(tgt)
            q = k = self.with_pos(tgt_norm, query_pos)
            tgt2 = self.self_attn(q, k, value=tgt_norm, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)

            tgt_norm = self.norm2(tgt)
            tgt2 = self.multihead_attn(self.with_pos(tgt_norm, query_pos), self.with_pos(memory, pos), memory,
                                       attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
            tgt = tgt + self.dropout2(tgt2)

            tgt_norm = self.norm3(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt_norm))))
            tgt = tgt + self.dropout3(tgt2)
            return tgt
        else:  # Original implementation
            tgt2 = self.self_attn(self.with_pos(tgt, query_pos), self.with_pos(tgt, query_pos), tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            tgt = self.norm1(tgt + self.dropout1(tgt2))
            tgt2 = self.multihead_attn(self.with_pos(tgt, query_pos), self.with_pos(memory, pos), memory,
                                       attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
            tgt = self.norm2(tgt + self.dropout2(tgt2))
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            return self.norm3(tgt + self.dropout3(tgt2))

