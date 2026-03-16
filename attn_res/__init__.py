"""
Public package interface for the Attention Residuals Transformer.

This exposes the main user-facing classes and utilities so that users can write:

    from attn_res import (
        AttnResMode,
        TransformerConfig,
        AttnResTransformer,
        RMSNorm,
        RMSNormNoWeight,
        GroupedQueryAttention,
        SwiGLUFFN,
        AttnResOperator,
        precompute_rope_freqs,
        apply_rope,
        two_phase_block_attnres_inference,
        online_softmax_merge,
    )
"""

from attn_res.main import (
    AttnResMode,
    TransformerConfig,
    RMSNorm,
    RMSNormNoWeight,
    precompute_rope_freqs,
    apply_rope,
    GroupedQueryAttention,
    SwiGLUFFN,
    AttnResOperator,
    AttnResTransformer,
    two_phase_block_attnres_inference,
    online_softmax_merge,
)

__all__ = [
    "AttnResMode",
    "TransformerConfig",
    "RMSNorm",
    "RMSNormNoWeight",
    "precompute_rope_freqs",
    "apply_rope",
    "GroupedQueryAttention",
    "SwiGLUFFN",
    "AttnResOperator",
    "AttnResTransformer",
    "two_phase_block_attnres_inference",
    "online_softmax_merge",
]

