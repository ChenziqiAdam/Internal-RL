"""
Phase 3: Autoregressive Causal Transformer
Architecture (Table A16):
- 6 layers, dim=256, 4 heads, head_dim=64, MLP=512
- Relative positional encodings (rotary / T5-style)
- Predicts next action (cross-entropy) + optionally next obs (λ * CE)
- Input: interleaved (obs, action) tokens
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from env import OBS_DIM, NUM_COLORS

NUM_ACTIONS = 4
TRANSFORMER_DIM = 256
NUM_HEADS = 4
HEAD_DIM = 64
MLP_DIM = 512
NUM_LAYERS = 6


# ── Rotary Positional Encoding ──────────────────────────────────────────────

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, seq_len: int):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


# ── Multi-Head Self-Attention with RoPE ─────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim

        self.qkv = nn.Linear(dim, 3 * inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, dim, bias=False)
        self.rotary = RotaryEmbedding(head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) for t in qkv]

        cos, sin = self.rotary(T)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # Causal mask via scaled dot-product attention (PyTorch 2.0+)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.out_proj(out)


# ── Transformer Block ────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, mlp_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, head_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ── Full Autoregressive Model ────────────────────────────────────────────────

class AutoregressiveTransformer(nn.Module):
    """
    Processes interleaved (obs, action) token sequences.
    Token sequence: obs_0, act_0, obs_1, act_1, ...
    Predicts: act_t from (obs_0..obs_t) and optionally obs_{t+1}.

    obs tokens and action tokens are embedded into dim-dimensional vectors.
    The residual stream after each layer is accessible for linear probing.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        num_actions: int = NUM_ACTIONS,
        dim: int = TRANSFORMER_DIM,
        num_layers: int = NUM_LAYERS,
        num_heads: int = NUM_HEADS,
        head_dim: int = HEAD_DIM,
        mlp_dim: int = MLP_DIM,
        predict_obs: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.predict_obs = predict_obs

        # Input embeddings
        self.obs_embed = nn.Linear(obs_dim, dim)
        self.action_embed = nn.Embedding(num_actions, dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, head_dim, mlp_dim)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(dim)

        # Output heads
        self.action_head = nn.Linear(dim, num_actions)
        if predict_obs:
            self.obs_head = nn.Linear(dim, obs_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def embed_sequence(
        self, obs_seq: torch.Tensor, action_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        obs_seq: (B, T+1, obs_dim)
        action_seq: (B, T) — integer actions

        Returns interleaved token sequence (B, 2T+1, dim) and token type mask.
        Layout: [obs_0, act_0, obs_1, act_1, ..., obs_T]
        """
        B, Tp1, _ = obs_seq.shape
        T = Tp1 - 1

        obs_emb = self.obs_embed(obs_seq)          # (B, T+1, dim)
        act_emb = self.action_embed(action_seq)     # (B, T, dim)

        # Interleave: obs_0, act_0, obs_1, act_1, ..., obs_T
        tokens = torch.zeros(B, 2 * T + 1, self.dim, device=obs_seq.device)
        tokens[:, 0::2, :] = obs_emb              # even positions = obs
        tokens[:, 1::2, :] = act_emb              # odd positions = actions

        # is_action_token: (2T+1,) bool
        is_action = torch.zeros(2 * T + 1, dtype=torch.bool)
        is_action[1::2] = True

        return tokens, is_action

    def forward(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        return_residuals: bool = False,
    ):
        """
        obs_seq: (B, T+1, obs_dim)
        action_seq: (B, T) int64

        Returns:
          action_logits: (B, T, num_actions)  — predictions at obs positions
          obs_logits: (B, T, obs_dim) or None
          residuals: list of (B, 2T+1, dim) per layer if return_residuals
        """
        tokens, is_action = self.embed_sequence(obs_seq, action_seq)

        residuals = [] if return_residuals else None
        x = tokens
        for layer in self.layers:
            x = layer(x)
            if return_residuals:
                residuals.append(x)

        x = self.final_norm(x)

        # Action predictions: from obs token positions (even), predict next action
        # Position 2t predicts action at step t
        T = action_seq.shape[1]
        obs_positions = x[:, 0::2, :][:, :T, :]  # (B, T, dim) — obs tokens 0..T-1
        action_logits = self.action_head(obs_positions)  # (B, T, num_actions)

        obs_logits = None
        if self.predict_obs:
            # Predict obs from action token positions
            act_positions = x[:, 1::2, :]  # (B, T, dim)
            obs_logits = self.obs_head(act_positions)  # (B, T, obs_dim)

        return action_logits, obs_logits, residuals

    def get_residual_stream(
        self, obs_seq: torch.Tensor, action_seq: torch.Tensor
    ) -> list:
        """
        Returns list of residual streams, one per layer.
        Each is (B, T, dim) at obs token positions only (for subgoal probing).
        """
        _, _, residuals = self.forward(obs_seq, action_seq, return_residuals=True)
        T = action_seq.shape[1]
        # Extract obs token residuals
        return [r[:, 0::2, :][:, :T, :] for r in residuals]  # list of (B, T, dim)
