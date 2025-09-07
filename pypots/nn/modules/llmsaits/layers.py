# Created by Wenjie Du <wdu@time-series.ai>
# Modified by Franck Junior Aboya <mesabo18@gmail.com>
# License: BSD-3-Clause

from math import sqrt
import torch
import torch.nn as nn


class ReprogrammingLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: int = None,
        d_llm: int = None,
        attention_dropout: float = 0.1,
        use_residual: bool = True,
        use_layernorm: bool = True,
        use_gating: bool = True,
        use_adapter: bool = True,
    ):
        super().__init__()

        d_k = d_k or (d_model // n_heads)
        self.d_model = d_model
        self.d_llm = d_llm
        self.n_heads = n_heads
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm
        self.use_gating = use_gating
        self.use_adapter = use_adapter

        self.query_projection = nn.Linear(d_model, d_k * n_heads)
        self.key_projection = nn.Linear(d_llm, d_k * n_heads)
        self.value_projection = nn.Linear(d_llm, d_k * n_heads)
        self.out_projection = nn.Linear(d_k * n_heads, d_llm)
        self.dropout = nn.Dropout(attention_dropout)

        if self.use_layernorm:
            self.norm = nn.LayerNorm(d_llm)
        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(2 * d_llm, d_llm),
                nn.Sigmoid()
            )
        if self.use_adapter:
            self.adapter = nn.Sequential(
                nn.Linear(d_model, d_llm),
                nn.LayerNorm(d_llm),
                nn.GELU()
            )

        self.activation = nn.GELU()

    def forward(self, target_embedding, source_embedding, value_embedding=None):
        """
        target_embedding: [B*, L, d_model]
        source_embedding: [B*, S, d_llm]
        value_embedding:  [B*, S, d_llm] or None
        """
        B_, L, _ = target_embedding.shape
        _, S, _ = source_embedding.shape
        H = self.n_heads

        if value_embedding is None:
            value_embedding = source_embedding

        if self.use_adapter:
            target_proj = self.adapter(target_embedding)  # [B*, L, d_llm]
        else:
            target_proj = target_embedding  # [B*, L, d_model]

        # Multi-head projections
        target = self.query_projection(target_embedding).view(B_, L, H, -1)  # [B*, L, H, d_k]
        source = self.key_projection(source_embedding).view(B_, S, H, -1)    # [B*, S, H, d_k]
        value = self.value_projection(value_embedding).view(B_, S, H, -1)    # [B*, S, H, d_k]

        # Cross-attention
        out = self.reprogramming(target, source, value)      # [B*, L, H, d_k]
        out = out.reshape(B_, L, -1)                          # [B*, L, H*d_k]
        out = self.out_projection(out)                        # [B*, L, d_llm]

        if self.use_residual:
            out = out + target_proj

        if self.use_layernorm:
            out = self.norm(out)

        out = self.activation(out)

        if self.use_gating:
            fusion = torch.cat([target_proj, out], dim=-1)  # [B*, L, d_model + d_llm]
            gate_weight = self.gate(fusion)                 # [B*, L, d_llm]
            out = gate_weight * out + (1 - gate_weight) * target_proj

        return out

    def reprogramming(self, target, source, value):
        """
        target: [B*, L, H, d_k]
        source: [B*, S, H, d_k]
        value:  [B*, S, H, d_k]
        """
        B_, L, H, E = target.shape
        scale = 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", target, source)  # [B*, H, L, S]
        attn_weights = self.dropout(torch.softmax(scale * scores, dim=-1))  # [B*, H, L, S]

        out = torch.einsum("bhls,bshe->blhe", attn_weights, value)  # [B*, L, H, d_k]
        return out