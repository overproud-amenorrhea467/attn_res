"""
Attention Residuals with Grouped Query Attention (GQA) Transformer
===================================================================

Implementation of the "Attention Residuals" paper (Kimi Team, 2026),
integrated with Multi-Head Grouped Query Attention.

Algorithmic Pseudocode
----------------------

### Full Attention Residuals (Full AttnRes)

    FOR each layer l in 1..L:
        # Collect all previous layer outputs as keys/values
        V = [h1, f1(h1), f2(h2), ..., f_{l-1}(h_{l-1})]   # shape [l, d]
        K = RMSNorm(V)                                       # normalize keys
        q_l = w_l                                            # learned pseudo-query ∈ R^d

        # Compute softmax attention over depth
        logits_i = dot(q_l, K_i)  for i in 0..l-1            # [l]
        alpha = softmax(logits)                               # [l]

        # Weighted aggregation → input to layer l
        h_l = sum(alpha_i * V_i)                              # [d]

        # Apply layer transformation (Attention or MLP)
        output_l = f_l(LayerNorm(h_l))

### Block Attention Residuals (Block AttnRes)

    Partition L layers into N blocks of S = L/N layers each.
    blocks = [embedding]    # b_0 = h_1
    partial_block = 0       # running sum within current block

    FOR each layer l in 1..L:
        # --- Inter-block attention ---
        V = stack(blocks + [partial_block])       # [N_cur+1, d]
        K = RMSNorm(V)
        logits = einsum('d, n d -> n', w_l, K)
        h_l = einsum('n, n d -> d', softmax(logits), V)

        # --- Block boundary check ---
        IF l hits block boundary:
            blocks.append(partial_block)
            partial_block = 0

        # --- Layer computation (PreNorm + GQA or MLP) ---
        output_l = f_l(LayerNorm(h_l))
        partial_block = partial_block + output_l

### Grouped Query Attention (GQA)

    Given H query heads and G key/value groups (G divides H):
        Q = x @ W_Q → [B, T, H, d_head]
        K = x @ W_K → [B, T, G, d_head]
        V = x @ W_V → [B, T, G, d_head]

        # Expand K, V: each group serves H/G query heads
        K = repeat(K, groups → H/G copies)
        V = repeat(V, groups → H/G copies)

        # Standard scaled dot-product attention
        attn = softmax(Q @ K^T / sqrt(d_head)) @ V
        output = attn.reshape(B, T, d_model) @ W_O

### Two-Phase Inference for Block AttnRes (Algorithm 1)

    FOR each block n:
        Phase 1 — Parallel inter-block attention:
            Q = stack([w_l for l in block_n])                    # [S, d]
            K, V = stack(block_reps[0..n-1])                     # [n, d]
            {o1_l, max1_l, lse1_l} = ATTN_WITH_STATS(Q, K, V)   # batched

        Phase 2 — Sequential intra-block + online softmax merge:
            partial = 0
            FOR i, l in enumerate(block_n):
                IF i == 0:
                    h_l = o1_l / lse1_l
                ELSE:
                    o2_l, max2_l, lse2_l = ATTN_WITH_STATS(w_l, partial, partial)
                    m = max(max1_l, max2_l)
                    h_l = (exp(max1-m)*o1 + exp(max2-m)*o2) /
                           (exp(max1-m)*lse1 + exp(max2-m)*lse2)
                output_l = f_l(LayerNorm(h_l))
                partial = partial + output_l

References
----------
- Attention Residuals (Kimi Team, 2026): https://github.com/MoonshotAI/Attention-Residuals
- GQA: Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models", 2023
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# =============================================================================
# Configuration
# =============================================================================


class AttnResMode(Enum):
    """Which Attention Residual variant to use."""

    NONE = "none"  # Standard residual connections
    FULL = "full"  # Full AttnRes: attend over all previous layer outputs
    BLOCK = "block"  # Block AttnRes: attend over block-level representations


@dataclass
class TransformerConfig:
    """Configuration for the Transformer with Attention Residuals and GQA.

    Attributes:
        d_model: Hidden dimension of the model.
        n_layers: Total number of transformer sub-layers (each Attn and MLP
            counts as one layer, so a standard transformer block = 2 layers).
        n_heads: Number of query attention heads.
        n_kv_heads: Number of key/value heads for Grouped Query Attention.
            Must divide ``n_heads``. When ``n_kv_heads == n_heads`` this is
            standard MHA; when ``n_kv_heads == 1`` this is Multi-Query Attention.
        vocab_size: Size of the token vocabulary.
        max_seq_len: Maximum sequence length for positional encodings.
        ffn_mult: Multiplier for the feed-forward hidden dimension
            (``d_ff = ffn_mult * d_model``).
        dropout: Dropout probability applied after attention and FFN.
        attn_res_mode: Which Attention Residual variant to use.
        n_blocks: Number of AttnRes blocks (only used when
            ``attn_res_mode == BLOCK``). Layers are evenly divided.
        eps: Epsilon for RMSNorm.
        rope_theta: Base frequency for Rotary Position Embeddings.
    """

    d_model: int = 512
    n_layers: int = 16  # total sub-layers (attn + mlp)
    n_heads: int = 8
    n_kv_heads: int = 2  # GQA groups
    vocab_size: int = 32000
    max_seq_len: int = 2048
    ffn_mult: float = 2.667  # ≈ 8/3 for SwiGLU
    dropout: float = 0.0
    attn_res_mode: AttnResMode = AttnResMode.BLOCK
    n_blocks: int = 8
    eps: float = 1e-6
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by "
            f"n_kv_heads ({self.n_kv_heads})"
        )
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by "
            f"n_heads ({self.n_heads})"
        )
        if self.attn_res_mode == AttnResMode.BLOCK:
            assert self.n_blocks >= 1, "n_blocks must be >= 1"

    @property
    def d_head(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads

    @property
    def n_kv_groups(self) -> int:
        """Number of query heads per KV group."""
        return self.n_heads // self.n_kv_heads

    @property
    def d_ff(self) -> int:
        """Feed-forward intermediate dimension (rounded to multiple of 8)."""
        raw = int(self.ffn_mult * self.d_model)
        return ((raw + 7) // 8) * 8  # align for GPU efficiency

    @property
    def block_size(self) -> int:
        """Number of sub-layers per AttnRes block."""
        if self.attn_res_mode != AttnResMode.BLOCK:
            return self.n_layers
        return self.n_layers // self.n_blocks


# =============================================================================
# Core Building Blocks
# =============================================================================


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes the input by its RMS value, scaled by a learnable parameter.

    Args:
        dim: Feature dimension to normalize over.
        eps: Small constant for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Normalized tensor of the same shape.
        """
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


class RMSNormNoWeight(nn.Module):
    """RMSNorm without a learnable scale parameter.

    Used inside the AttnRes kernel function φ to normalize keys so that
    layers with large-magnitude outputs do not dominate the softmax.

    Args:
        eps: Small constant for numerical stability.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS normalization without learned scale.

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Normalized tensor of the same shape.
        """
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype)


# =============================================================================
# Rotary Position Embedding (RoPE)
# =============================================================================


def precompute_rope_freqs(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Precompute complex-valued RoPE frequencies.

    Args:
        dim: Head dimension (must be even).
        max_seq_len: Maximum sequence length to precompute for.
        theta: Base frequency.
        device: Target device.

    Returns:
        Complex tensor of shape ``(max_seq_len, dim // 2)`` containing
        the rotation factors ``exp(i * pos * freq)``.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    positions = torch.arange(max_seq_len, device=device).float()
    angles = torch.outer(positions, freqs)  # [max_seq_len, dim//2]
    return torch.polar(torch.ones_like(angles), angles)  # e^{i*angle}


def apply_rope(x: Tensor, freqs: Tensor) -> Tensor:
    """Apply Rotary Position Embedding to query or key tensor.

    Args:
        x: Tensor of shape ``(B, T, H, d_head)``.
        freqs: Precomputed complex frequencies of shape ``(T, d_head // 2)``.

    Returns:
        Rotated tensor of the same shape.
    """
    # Reshape to complex pairs
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs.unsqueeze(0).unsqueeze(2)  # [1, T, 1, d_head//2]
    x_rotated = x_complex * freqs
    return torch.view_as_real(x_rotated).reshape_as(x).to(x.dtype)


# =============================================================================
# Grouped Query Attention (GQA)
# =============================================================================


class GroupedQueryAttention(nn.Module):
    """Multi-Head Grouped Query Attention.

    Supports the full spectrum from Multi-Head Attention (n_kv_heads == n_heads)
    to Multi-Query Attention (n_kv_heads == 1) and everything in between.

    Each KV group is shared across ``n_heads // n_kv_heads`` query heads.

    Args:
        cfg: Transformer configuration.
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.w_q = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.w_k = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=False)
        self.w_v = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=False)
        self.w_o = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)
        self.attn_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: Tensor,
        rope_freqs: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute grouped query attention.

        Args:
            x: Input tensor of shape ``(B, T, d_model)``.
            rope_freqs: RoPE frequencies of shape ``(T, d_head // 2)``.
            mask: Optional causal mask of shape ``(1, 1, T, T)`` or broadcastable.

        Returns:
            Output tensor of shape ``(B, T, d_model)``.
        """
        B, T, _ = x.shape
        cfg = self.cfg

        # Project to Q, K, V
        q = self.w_q(x).view(B, T, cfg.n_heads, cfg.d_head)
        k = self.w_k(x).view(B, T, cfg.n_kv_heads, cfg.d_head)
        v = self.w_v(x).view(B, T, cfg.n_kv_heads, cfg.d_head)

        # Apply RoPE
        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)

        # Expand KV heads to match query heads
        # Each KV group serves (n_heads // n_kv_heads) query heads
        if cfg.n_kv_heads != cfg.n_heads:
            k = k.unsqueeze(3).expand(B, T, cfg.n_kv_heads, cfg.n_kv_groups, cfg.d_head)
            k = k.reshape(B, T, cfg.n_heads, cfg.d_head)
            v = v.unsqueeze(3).expand(B, T, cfg.n_kv_heads, cfg.n_kv_groups, cfg.d_head)
            v = v.reshape(B, T, cfg.n_heads, cfg.d_head)

        # Transpose for attention: (B, H, T, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(cfg.d_head)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, T, T]

        if mask is not None:
            attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # [B, H, T, d_head]
        out = out.transpose(1, 2).contiguous().view(B, T, cfg.n_heads * cfg.d_head)

        return self.w_o(out)


# =============================================================================
# Feed-Forward Network (SwiGLU)
# =============================================================================


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network.

    Implements the gated linear unit variant:
        FFN(x) = (Swish(x @ W_gate) * (x @ W_up)) @ W_down

    Args:
        cfg: Transformer configuration.
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Apply SwiGLU feed-forward transformation.

        Args:
            x: Input of shape ``(B, T, d_model)``.

        Returns:
            Output of shape ``(B, T, d_model)``.
        """
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))


# =============================================================================
# Attention Residuals Operator
# =============================================================================


class AttnResOperator(nn.Module):
    """Depth-wise Attention Residual operator.

    Computes softmax attention over a set of source representations (either
    individual layer outputs for Full AttnRes or block-level summaries for
    Block AttnRes) using a single learned pseudo-query vector ``w_l``.

    The kernel function is:
        φ(q, k) = exp(q^T · RMSNorm(k))

    with softmax normalization over all sources.

    Args:
        d_model: Hidden dimension.
        eps: Epsilon for RMSNorm on keys.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.pseudo_query = nn.Parameter(torch.zeros(d_model))
        self.key_norm = RMSNormNoWeight(eps=eps)

    def forward(self, sources: Tensor) -> Tensor:
        """Compute weighted aggregation of source representations.

        The pseudo-query is initialized to zero so that initial attention
        weights are uniform (equivalent to equal-weight averaging), which
        prevents training volatility.

        Args:
            sources: Tensor of shape ``(N_src, B, T, d)`` containing the
                source representations to attend over (e.g. block reps
                plus partial sum, or all previous layer outputs).

        Returns:
            Aggregated representation of shape ``(B, T, d)``.
        """
        # K = RMSNorm(sources), no learned weight
        K = self.key_norm(sources)  # [N_src, B, T, d]

        # Compute logits: dot product of pseudo-query with each normalized source
        # pseudo_query: [d], K: [N_src, B, T, d] → logits: [N_src, B, T]
        logits = torch.einsum("d, n b t d -> n b t", self.pseudo_query, K)

        # Softmax over sources (depth dimension)
        weights = F.softmax(logits, dim=0)  # [N_src, B, T]

        # Weighted sum
        out = torch.einsum("n b t, n b t d -> b t d", weights, sources)
        return out


# =============================================================================
# Transformer Sub-Layer (with AttnRes integration)
# =============================================================================


class TransformerSubLayer(nn.Module):
    """A single transformer sub-layer: either Attention or FFN.

    Each sub-layer is wrapped with PreNorm and receives its input via the
    Attention Residuals mechanism (when enabled) rather than a plain additive
    residual.

    Args:
        cfg: Transformer configuration.
        layer_idx: Index of this sub-layer (0-based) within the full stack.
        is_attention: If True, this sub-layer contains GQA; otherwise FFN.
    """

    def __init__(
        self,
        cfg: TransformerConfig,
        layer_idx: int,
        is_attention: bool,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.is_attention = is_attention

        # PreNorm
        self.norm = RMSNorm(cfg.d_model, eps=cfg.eps)

        # Core transformation
        if is_attention:
            self.fn = GroupedQueryAttention(cfg)
        else:
            self.fn = SwiGLUFFN(cfg)

        # AttnRes operator (one per sub-layer for depth-wise attention)
        if cfg.attn_res_mode != AttnResMode.NONE:
            self.attn_res = AttnResOperator(cfg.d_model, eps=cfg.eps)

    def forward(
        self,
        sources: Tensor,
        rope_freqs: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply the sub-layer transformation.

        This method first computes the AttnRes aggregation over ``sources``
        to produce the input ``h_l``, then applies PreNorm + the core
        function (GQA or FFN).

        Args:
            sources: Source representations for AttnRes, of shape
                ``(N_src, B, T, d_model)``.
            rope_freqs: RoPE frequencies for attention sub-layers.
            mask: Optional causal mask.

        Returns:
            The sub-layer output ``f_l(Norm(h_l))`` of shape ``(B, T, d_model)``.
        """
        # Compute input via AttnRes or plain sum
        if self.cfg.attn_res_mode != AttnResMode.NONE:
            h = self.attn_res(sources)
        else:
            # Standard residual: h_l = sum of all sources (uniform weights)
            h = sources.sum(dim=0)

        # PreNorm + transformation
        normed = self.norm(h)
        if self.is_attention:
            output = self.fn(normed, rope_freqs, mask)
        else:
            output = self.fn(normed)

        return output


# =============================================================================
# Full Model: Transformer with Attention Residuals + GQA
# =============================================================================


class AttnResTransformer(nn.Module):
    """Transformer Language Model with Attention Residuals and GQA.

    The model interleaves Grouped Query Attention and SwiGLU FFN sub-layers,
    connected via learned depth-wise attention (AttnRes) instead of fixed
    additive residual connections.

    Supports three residual modes:
        - ``NONE``: Standard additive residuals (baseline).
        - ``FULL``: Full AttnRes — each sub-layer attends over *all*
          previous sub-layer outputs. Memory: O(L·d).
        - ``BLOCK``: Block AttnRes — sub-layers are grouped into N blocks;
          inter-block attention operates over N block-level summaries plus
          the current partial sum. Memory: O(N·d).

    Args:
        cfg: Transformer configuration.

    Example::

        >>> cfg = TransformerConfig(
        ...     d_model=256, n_layers=8, n_heads=8, n_kv_heads=2,
        ...     vocab_size=1000, attn_res_mode=AttnResMode.BLOCK, n_blocks=4,
        ... )
        >>> model = AttnResTransformer(cfg)
        >>> tokens = torch.randint(0, 1000, (2, 64))
        >>> logits = model(tokens)  # (2, 64, 1000)
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Token embedding
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Build sub-layers: alternating Attention and FFN
        self.layers = nn.ModuleList()
        for i in range(cfg.n_layers):
            is_attn = i % 2 == 0  # even = attention, odd = FFN
            self.layers.append(TransformerSubLayer(cfg, i, is_attn))

        # Output head
        self.out_norm = RMSNorm(cfg.d_model, eps=cfg.eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying (embedding ↔ lm_head)
        self.lm_head.weight = self.token_emb.weight

        # Precompute RoPE frequencies (registered as buffer, not parameter)
        rope_freqs = precompute_rope_freqs(cfg.d_head, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_freqs", rope_freqs, persistent=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model parameters.

        - Embeddings and linear layers use normal initialization.
        - AttnRes pseudo-queries are initialized to zero (critical for
          stable training; ensures uniform initial weights).
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, AttnResOperator):
                # CRITICAL: zero init for pseudo-query ensures uniform
                # initial attention → equivalent to equal-weight average
                nn.init.zeros_(module.pseudo_query)

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Build a causal attention mask.

        Args:
            seq_len: Sequence length.
            device: Target device.

        Returns:
            Mask of shape ``(1, 1, T, T)`` with ``-inf`` for positions
            that should not be attended to.
        """
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    def _forward_standard(
        self,
        x: Tensor,
        rope_freqs: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Forward pass with standard additive residual connections.

        h_l = h_{l-1} + f_{l-1}(Norm(h_{l-1}))

        Args:
            x: Token embeddings of shape ``(B, T, d)``.
            rope_freqs: RoPE frequencies.
            mask: Causal mask.

        Returns:
            Final hidden state of shape ``(B, T, d)``.
        """
        h = x
        for layer in self.layers:
            # sources is just [h] for standard residuals → sum = h
            sources = h.unsqueeze(0)
            output = layer(sources, rope_freqs, mask)
            h = h + output
        return h

    def _forward_full_attnres(
        self,
        x: Tensor,
        rope_freqs: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Forward pass with Full Attention Residuals.

        Each sub-layer attends over the embedding and all previous sub-layer
        outputs to compute its input:
            h_l = Σ α_{i→l} · v_i

        where v_0 = embedding, v_i = f_i(h_i) for i ≥ 1.

        Args:
            x: Token embeddings of shape ``(B, T, d)``.
            rope_freqs: RoPE frequencies.
            mask: Causal mask.

        Returns:
            Final hidden state of shape ``(B, T, d)``.
        """
        B, T, d = x.shape

        # Store all layer outputs: v_0 = embedding, v_i = f_i(h_i)
        layer_outputs: list[Tensor] = [x]  # v_0

        for layer in self.layers:
            # Stack all previous outputs as sources
            sources = torch.stack(layer_outputs, dim=0)  # [i+1, B, T, d]
            output = layer(sources, rope_freqs, mask)
            layer_outputs.append(output)

        # Final aggregation: attend over all outputs for the output layer
        all_sources = torch.stack(layer_outputs, dim=0)
        # For the final output, we just sum (or could add another AttnRes)
        # Following the paper: final output aggregates all block reps
        return all_sources.sum(dim=0)

    def _forward_block_attnres(
        self,
        x: Tensor,
        rope_freqs: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Forward pass with Block Attention Residuals.

        Layers are partitioned into N blocks of S sub-layers. Within each
        block, outputs are accumulated via standard summation. Across blocks,
        each sub-layer attends over block-level summaries plus the current
        partial sum.

        This reduces memory from O(L·d) to O(N·d).

        Args:
            x: Token embeddings of shape ``(B, T, d)``.
            rope_freqs: RoPE frequencies.
            mask: Causal mask.

        Returns:
            Final hidden state of shape ``(B, T, d)``.
        """
        cfg = self.cfg
        block_size = cfg.block_size

        # b_0 = token embedding (always included as a source)
        blocks: list[Tensor] = [x]  # completed block representations
        partial_block: Optional[Tensor] = None  # running intra-block sum

        for i, layer in enumerate(self.layers):
            # Assemble sources: completed blocks + partial sum (if any)
            if partial_block is not None:
                source_list = blocks + [partial_block]
            else:
                source_list = list(blocks)

            sources = torch.stack(source_list, dim=0)  # [N_src, B, T, d]

            # AttnRes → PreNorm → layer function
            output = layer(sources, rope_freqs, mask)

            # Check if we've hit a block boundary BEFORE accumulating
            # (block boundary happens every block_size layers)
            if i > 0 and i % block_size == 0 and partial_block is not None:
                blocks.append(partial_block)
                partial_block = None

            # Accumulate into the partial block sum
            if partial_block is None:
                partial_block = output
            else:
                partial_block = partial_block + output

        # Final output: attend over all completed blocks + last partial block
        if partial_block is not None:
            final_sources = torch.stack(blocks + [partial_block], dim=0)
        else:
            final_sources = torch.stack(blocks, dim=0)

        # Sum for the final output (could also use a final AttnRes operator)
        return final_sources.sum(dim=0)

    def forward(
        self,
        tokens: Tensor,
        targets: Optional[Tensor] = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Full forward pass: tokens → logits (and optional loss).

        Args:
            tokens: Input token IDs of shape ``(B, T)``.
            targets: Optional target token IDs of shape ``(B, T)`` for
                computing cross-entropy loss.

        Returns:
            If ``targets`` is None: logits of shape ``(B, T, vocab_size)``.
            If ``targets`` is provided: tuple of ``(logits, loss)``.
        """
        B, T = tokens.shape
        assert (
            T <= self.cfg.max_seq_len
        ), f"Sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"

        # Token embeddings
        x = self.token_emb(tokens)  # [B, T, d_model]

        # RoPE frequencies for this sequence length
        rope_freqs = self.rope_freqs[:T]

        # Causal mask
        mask = self._build_causal_mask(T, x.device)

        # Route to the appropriate forward method
        if self.cfg.attn_res_mode == AttnResMode.NONE:
            h = self._forward_standard(x, rope_freqs, mask)
        elif self.cfg.attn_res_mode == AttnResMode.FULL:
            h = self._forward_full_attnres(x, rope_freqs, mask)
        elif self.cfg.attn_res_mode == AttnResMode.BLOCK:
            h = self._forward_block_attnres(x, rope_freqs, mask)
        else:
            raise ValueError(f"Unknown attn_res_mode: {self.cfg.attn_res_mode}")

        # Output projection
        h = self.out_norm(h)
        logits = self.lm_head(h)  # [B, T, vocab_size]

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
            )
            return logits, loss

        return logits

    def count_parameters(self) -> dict[str, int]:
        """Count trainable parameters by component.

        Returns:
            Dictionary mapping component names to parameter counts.
        """
        counts: dict[str, int] = {}
        counts["embedding"] = sum(p.numel() for p in self.token_emb.parameters())
        counts["layers"] = sum(p.numel() for p in self.layers.parameters())
        counts["output"] = sum(p.numel() for p in self.out_norm.parameters())
        # lm_head shares weights with embedding, so not counted separately

        attn_res_params = 0
        for module in self.modules():
            if isinstance(module, AttnResOperator):
                attn_res_params += sum(p.numel() for p in module.parameters())
        counts["attn_res"] = attn_res_params

        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts


# =============================================================================
# Two-Phase Inference Helper (Algorithm 1 from the paper)
# =============================================================================


def two_phase_block_attnres_inference(
    pseudo_queries: Tensor,
    block_reps: Tensor,
    key_norm: RMSNormNoWeight,
) -> tuple[Tensor, Tensor, Tensor]:
    """Phase 1 of Algorithm 1: batched inter-block attention with statistics.

    Computes attention of all S pseudo-queries against the block
    representations in a single batched operation, returning softmax
    statistics (max, log-sum-exp) for online softmax merging in Phase 2.

    Args:
        pseudo_queries: Stacked pseudo-query vectors of shape ``(S, d)``
            for all layers in the current block.
        block_reps: Block-level representations of shape ``(N, B, T, d)``
            for all completed blocks.
        key_norm: RMSNorm (no weight) to normalize keys.

    Returns:
        Tuple of:
            - outputs: Unnormalized attention outputs ``(S, B, T, d)``.
            - maxes: Per-query max logits ``(S, B, T)``.
            - lse: Per-query log-sum-exp ``(S, B, T)``.
    """
    K = key_norm(block_reps)  # [N, B, T, d]
    # logits: [S, N, B, T]
    logits = torch.einsum("s d, n b t d -> s n b t", pseudo_queries, K)

    # Compute stable softmax statistics
    maxes = logits.max(dim=1).values  # [S, B, T]
    shifted = logits - maxes.unsqueeze(1)
    exp_shifted = shifted.exp()
    lse = exp_shifted.sum(dim=1)  # [S, B, T]

    # Unnormalized weighted sum
    outputs = torch.einsum("s n b t, n b t d -> s b t d", exp_shifted, block_reps)

    return outputs, maxes, lse


def online_softmax_merge(
    o1: Tensor,
    m1: Tensor,
    l1: Tensor,
    o2: Tensor,
    m2: Tensor,
    l2: Tensor,
) -> Tensor:
    """Merge two attention outputs via online softmax (Milakov & Gimelshein, 2018).

    Given two unnormalized attention outputs with their max-logits and
    log-sum-exp statistics, computes the correctly normalized combined result
    without revisiting the original logits.

    Args:
        o1: First unnormalized output ``(..., d)``.
        m1: First max logits ``(...)``.
        l1: First sum-of-exp ``(...)``.
        o2: Second unnormalized output ``(..., d)``.
        m2: Second max logits ``(...)``.
        l2: Second sum-of-exp ``(...)``.

    Returns:
        Merged, normalized output ``(..., d)``.
    """
    m = torch.maximum(m1, m2)
    exp1 = (m1 - m).exp().unsqueeze(-1)  # [..., 1]
    exp2 = (m2 - m).exp().unsqueeze(-1)

    l1_adj = (m1 - m).exp() * l1
    l2_adj = (m2 - m).exp() * l2
    denom = (l1_adj + l2_adj).unsqueeze(-1)  # [..., 1]

    return (exp1 * o1 + exp2 * o2) / denom
