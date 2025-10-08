import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        # Generate sinusoidal position encoding
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [seq_len, dim//2]
        # Compute cos and sin embeddings
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        # Concatenate for applying rotation
        cos_emb = torch.cat((cos, cos), dim=-1)  # [seq_len, dim]
        sin_emb = torch.cat((sin, sin), dim=-1)  # [seq_len, dim]
        return (
            cos_emb[None, None, :, :],
            sin_emb[None, None, :, :],
        )  # [1, 1, seq_len, dim]


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)


class MutilHeadSelfAttn(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.ln1_q = nn.LayerNorm(hidden_dim)
        self.ln1_k = nn.LayerNorm(hidden_dim)
        self.ln1_v = nn.LayerNorm(hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.rope = RotaryPositionEmbedding(self.head_dim)

        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Linear(hidden_dim, hidden_dim)

    def self_attn(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) # 原始为除以 d_k

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, 1e-10)
        attn = attn - attn.max(dim=-1, keepdim=True)[0]  # for numerical stability
        attn = F.softmax(attn, dim=-1)

        if self.dropout is not None:
            attn = self.dropout(attn)

        x = torch.matmul(attn, v)
        return x

    def forward(self, q, k, v, mask=None):
        bs = q.shape[0]
        q_len, k_len = q.shape[1], k.shape[1]

        ln_q = self.ln1_q(q)
        ln_k = self.ln1_k(k)
        ln_v = self.ln1_v(v)

        query = (
            self.query(ln_q).view(bs, -1, self.num_heads, self.head_dim).transpose(1, 2)
        )  # [bs, num_heads, q_len, head_dim]
        key = self.key(ln_k).view(bs, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = (
            self.value(ln_v).view(bs, -1, self.num_heads, self.head_dim).transpose(1, 2)
        )

        # Apply Rotary Position Embedding
        cos_q, sin_q = self.rope(q_len, query.device)
        cos_k, sin_k = self.rope(k_len, key.device)
        query = apply_rotary_pos_emb(query, cos_q, sin_q)
        key = apply_rotary_pos_emb(key, cos_k, sin_k)

        attn_out = self.self_attn(query, key, value, mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bs, -1, self.hidden_dim)

        ffn_x = self.ln2(attn_out)
        ffn_out = self.relu(self.ffn(ffn_x))
        out = ffn_x + self.ln3(ffn_out)
        return out