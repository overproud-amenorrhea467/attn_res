# attn_res

A clean, single-file PyTorch implementation of **Attention Residuals** (Kimi Team, MoonshotAI, 2026), integrated with Grouped Query Attention (GQA), SwiGLU feed-forward networks, and Rotary Position Embeddings (RoPE).

> **Reference:** [Attention Residuals — MoonshotAI](https://github.com/MoonshotAI/Attention-Residuals/blob/master/Attention_Residuals.pdf)


---

## Installation

Clone the repository and install the single dependency:

```bash
uv pip install attn_res
```

No additional dependencies are required. The entire model lives in `attn_res/main.py`.

---

## Quick Start

```python
import torch
from attn_res.main import AttnResTransformer, TransformerConfig, AttnResMode

# Configure the model
cfg = TransformerConfig(
    d_model=512,
    n_layers=16,       # 8 transformer blocks (attention + FFN sub-layers each)
    n_heads=8,
    n_kv_heads=2,      # GQA: 4 query heads per KV group
    vocab_size=32000,
    max_seq_len=2048,
    attn_res_mode=AttnResMode.BLOCK,   # Full | Block | None
    n_blocks=8,        # number of AttnRes blocks (Block mode only)
)

model = AttnResTransformer(cfg)

# Inference
tokens = torch.randint(0, cfg.vocab_size, (1, 128))
logits = model(tokens)                    # (1, 128, vocab_size)

# Training (with targets)
targets = torch.randint(0, cfg.vocab_size, (1, 128))
logits, loss = model(tokens, targets)
loss.backward()

# Inspect parameter distribution
print(model.count_parameters())
# {'embedding': ..., 'layers': ..., 'output': ..., 'attn_res': ..., 'total': ...}
```

Run the included smoke test to verify all three modes against all configurations:

```bash
python example.py
```

Expected output:

```
======================================================================
Attention Residuals + GQA Transformer — Smoke Test
======================================================================

Mode:   none
  Logits shape : (2, 64, 1000)
  Loss         : 6.9079
  Forward time : 12.3 ms
  Total params : 3,141,000
  AttnRes params: 0

Mode:   full
  Logits shape : (2, 64, 1000)
  ...

Mode:  block
  Logits shape : (2, 64, 1000)
  ...

======================================================================
All modes passed!
======================================================================
```

Switch between residual modes by changing `attn_res_mode`:

| `AttnResMode` | Description |
|---|---|
| `NONE` | Standard additive residual connections (baseline) |
| `FULL` | Each sub-layer attends over *all* previous sub-layer outputs |
| `BLOCK` | Sub-layers attend over *block-level summaries* (memory-efficient) |

---

## Architecture Overview

The model is a decoder-only transformer language model composed of the following stack:

```
Token IDs  →  Embedding
           →  [Sub-Layer 0: GQA]         ─┐
           →  [Sub-Layer 1: SwiGLU FFN]   │  Block 0
           →  [Sub-Layer 2: GQA]         ─┘
           →  [Sub-Layer 3: SwiGLU FFN]   │  Block 1
           →  ...
           →  RMSNorm  →  LM Head  →  Logits
```

Each pair of sub-layers (GQA + FFN) constitutes one logical transformer block. The total number of sub-layers is `n_layers`, so there are `n_layers / 2` transformer blocks.

Connections between sub-layers are governed by the **Attention Residuals mechanism** rather than fixed additive residuals. The key principle is that the input to each sub-layer \( l \) is computed as a **learned, softmax-weighted combination** of prior representations, rather than a simple element-wise sum.

---

## The Attention Residuals Mechanism

### Motivation: Limitations of Additive Residuals

Standard transformer residual connections take the form:

\[
h_l = h_{l-1} + f_l\!\left(\text{Norm}(h_{l-1})\right)
\]

While powerful, this formulation treats all prior representations equally — the contribution of layer \( l-1 \) to layer \( l \) is always a uniform additive offset, with no ability for the network to dynamically re-weight or selectively retrieve information from earlier layers. The gradient signal for very deep layers must also propagate through a long chain of additions, which can lead to vanishing updates.

**Attention Residuals** (Kimi Team, 2026) replaces this fixed aggregation with a learned, depth-wise attention operation over past representations, turning the inter-layer routing itself into a differentiable retrieval problem.

---

### Full Attention Residuals

In the **Full AttnRes** variant, each sub-layer \( l \) computes its input \( h_l \) by attending over the embedding \( v_0 \) and all prior sub-layer outputs \( v_1, \ldots, v_{l-1} \):

\[
h_l = \sum_{i=0}^{l-1} \alpha_{i \to l} \cdot v_i, \qquad
\alpha_{i \to l} = \frac{\exp\!\left(w_l^\top \phi(v_i)\right)}{\sum_{j=0}^{l-1} \exp\!\left(w_l^\top \phi(v_j)\right)}
\]

where \( w_l \in \mathbb{R}^d \) is a **learned pseudo-query** unique to layer \( l \), and \( \phi(\cdot) = \text{RMSNorm}(\cdot) \) is a parameter-free key normalization function. The sub-layer transformation is then:

\[
\text{output}_l = f_l\!\left(\text{Norm}(h_l)\right)
\]

Algorithmic pseudocode:

```
FOR each layer l in 1..L:
    V = [h_0, output_1, output_2, ..., output_{l-1}]   # all past representations
    K = RMSNorm(V)                                       # normalize keys
    logits_i = dot(w_l, K_i)   for i in 0..l-1
    alpha    = softmax(logits)
    h_l      = sum(alpha_i * V_i)
    output_l = f_l(LayerNorm(h_l))
```

Memory cost for Full AttnRes is **O(L · B · T · d)** — the full stack of layer outputs must be kept alive throughout the forward pass.

**Implementation** (`_forward_full_attnres`):

```python
layer_outputs: list[Tensor] = [x]          # v_0 = embedding

for layer in self.layers:
    sources = torch.stack(layer_outputs, dim=0)   # [l+1, B, T, d]
    output  = layer(sources, rope_freqs, mask)
    layer_outputs.append(output)
```

---

### Block Attention Residuals

For practical deployment, **Block AttnRes** reduces the memory footprint from O(L · d) to **O(N · d)** by partitioning the \( L \) sub-layers into \( N \) blocks of \( S = L / N \) sub-layers each.

Rather than storing all individual layer outputs, the mechanism maintains:
- `blocks`: A list of **completed block summaries** \( b_0, b_1, \ldots \) — each is the sum of outputs within a finished block.
- `partial_block`: The **running intra-block partial sum** for the current (incomplete) block.

The input to sub-layer \( l \) is then:

\[
h_l = \sum_{n=0}^{N_{\text{cur}}} \alpha_{n \to l} \cdot s_n, \qquad
s_n = \begin{cases} \text{embedding} & n = 0 \\ \sum_{i \in \text{block}_n} \text{output}_i & n \geq 1 \end{cases}
\]

where \( N_{\text{cur}} \) is the number of completed blocks at layer \( l \), and the final source is always the current `partial_block`.

Algorithmic pseudocode:

```
blocks       = [embedding]     # b_0 = token embedding
partial      = None

FOR each layer l in 1..L:
    source_list = blocks + ([partial] if partial is not None else [])
    sources     = stack(source_list)                     # [N_cur+1, B, T, d]
    h_l         = AttnResOperator(sources)               # softmax over depth
    output_l    = f_l(LayerNorm(h_l))

    IF block boundary reached:
        blocks.append(partial)
        partial = None

    partial = output_l  if partial is None  else  partial + output_l
```

**Implementation** (`_forward_block_attnres`):

```python
blocks: list[Tensor] = [x]       # b_0 = embedding
partial_block: Tensor | None = None

for i, layer in enumerate(self.layers):
    source_list = blocks + ([partial_block] if partial_block is not None else [])
    sources     = torch.stack(source_list, dim=0)          # [N_src, B, T, d]
    output      = layer(sources, rope_freqs, mask)

    if i > 0 and i % block_size == 0 and partial_block is not None:
        blocks.append(partial_block)
        partial_block = None

    partial_block = output if partial_block is None else partial_block + output
```

---

### The Pseudo-Query and Key Normalization

The `AttnResOperator` module is the heart of the mechanism. It implements:

\[
\text{AttnRes}(S) = \sum_{n} \underbrace{\text{softmax}_n\!\left(w_l^\top \text{RMSNorm}(s_n)\right)}_{\alpha_n} \cdot s_n
\]

Two design choices from the paper are faithfully reproduced:

**1. Learned pseudo-query, zero-initialized.**
Each sub-layer \( l \) has a single learnable vector \( w_l \in \mathbb{R}^d \) (the pseudo-query). It is initialized to **zero**, which causes the initial softmax weights to be exactly uniform:

\[
w_l = \mathbf{0} \implies \alpha_n = \frac{1}{N_{\text{src}}} \quad \forall n
\]

This means that at the start of training, Attention Residuals behaves identically to an equal-weight average — a safe initialization that prevents early training instability before gradients have shaped the routing weights.

**2. Parameter-free key normalization (`RMSNormNoWeight`).**
Keys are normalized by their RMS value without any learned scale:

\[
\phi(v) = \frac{v}{\sqrt{\frac{1}{d}\sum_j v_j^2 + \epsilon}}
\]

This prevents representations with large magnitudes (which tend to appear in later layers) from systematically dominating the attention logits, ensuring the softmax competition is fair across layers of different depth and scale.

```python
class AttnResOperator(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        self.pseudo_query = nn.Parameter(torch.zeros(d_model))   # zero init
        self.key_norm     = RMSNormNoWeight(eps=eps)              # no learned scale

    def forward(self, sources):           # sources: [N_src, B, T, d]
        K       = self.key_norm(sources)  # [N_src, B, T, d]
        logits  = einsum("d, n b t d -> n b t", self.pseudo_query, K)
        weights = softmax(logits, dim=0)  # softmax over depth dimension
        return  einsum("n b t, n b t d -> b t d", weights, sources)
```

---

### Two-Phase Inference and Online Softmax Merging

The paper (Algorithm 1) proposes a two-phase inference scheme for Block AttnRes that separates inter-block and intra-block attention to enable parallelism:

**Phase 1 — Batched inter-block attention.**
All \( S \) pseudo-queries in block \( n \) are batched against the \( n \) completed block representations in a single matrix multiply, producing unnormalized attention outputs along with softmax statistics (max-logit and sum-of-exp):

\[
\{o^{(1)}_l,\, m^{(1)}_l,\, \ell^{(1)}_l\} = \text{ATTN\_STATS}\!\left(\{w_l\}_{l \in \text{block}_n},\; \{b_0, \ldots, b_{n-1}\}\right)
\]

**Phase 2 — Sequential intra-block attention with online softmax merge.**
The partial block sum grows incrementally. For each layer \( l \) in block \( n \):

\[
o^{(2)}_l,\, m^{(2)}_l,\, \ell^{(2)}_l = \text{ATTN\_STATS}(w_l,\, \text{partial}_l,\, \text{partial}_l)
\]

The two partial results are combined using the **online softmax identity** (Milakov & Gimelshein, 2018):

\[
h_l = \frac{e^{m^{(1)}-m} \cdot o^{(1)}_l + e^{m^{(2)}-m} \cdot o^{(2)}_l}{e^{m^{(1)}-m} \cdot \ell^{(1)}_l + e^{m^{(2)}-m} \cdot \ell^{(2)}_l}, \quad m = \max(m^{(1)}, m^{(2)})
\]

This merging is numerically equivalent to computing the full attention from scratch, but avoids re-materializing the logits. The implementation is provided in `two_phase_block_attnres_inference` and `online_softmax_merge`:

```python
def online_softmax_merge(o1, m1, l1, o2, m2, l2):
    m    = torch.maximum(m1, m2)
    exp1 = (m1 - m).exp().unsqueeze(-1)
    exp2 = (m2 - m).exp().unsqueeze(-1)
    denom = ((m1 - m).exp() * l1 + (m2 - m).exp() * l2).unsqueeze(-1)
    return (exp1 * o1 + exp2 * o2) / denom
```

---

## Component Reference

### Grouped Query Attention (GQA)

`GroupedQueryAttention` implements the GQA formulation of Ainslie et al. (2023). With \( H \) query heads and \( G \) key/value heads (\( G \mid H \)):

\[
Q = xW_Q \in \mathbb{R}^{B \times T \times H \times d_h}, \quad
K = xW_K \in \mathbb{R}^{B \times T \times G \times d_h}, \quad
V = xW_V \in \mathbb{R}^{B \times T \times G \times d_h}
\]

Each KV group is broadcast to serve \( H/G \) query heads, then standard scaled dot-product attention is applied:

\[
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right) V
\]

Setting `n_kv_heads == n_heads` recovers standard Multi-Head Attention; setting `n_kv_heads == 1` gives Multi-Query Attention. The KV head expansion is done via a zero-copy `.expand()` + `.reshape()` to avoid materializing duplicate weights:

```python
k = k.unsqueeze(3).expand(B, T, n_kv_heads, n_kv_groups, d_head)
k = k.reshape(B, T, n_heads, d_head)
```

RoPE is applied to both \( Q \) and \( K \) prior to the attention computation.

---

### SwiGLU Feed-Forward Network

Each FFN sub-layer uses the **SwiGLU** activation (Shazeer, 2020):

\[
\text{FFN}(x) = \left(\text{SiLU}(xW_{\text{gate}}) \odot xW_{\text{up}}\right) W_{\text{down}}
\]

The intermediate dimension is set to \( d_{ff} = \lfloor 8d/3 \rfloor \) (rounded up to the nearest multiple of 8 for GPU alignment), matching the convention from LLaMA-style architectures. All projection matrices are bias-free.

---

### Rotary Position Embeddings (RoPE)

Positional information is injected via **RoPE** (Su et al., 2021). Complex-valued rotation factors are precomputed once:

\[
f_j = \frac{1}{\theta^{2j/d}}, \quad
R_{t,j} = e^{i \cdot t \cdot f_j}
\]

and applied to query and key vectors by interpreting consecutive pairs of dimensions as complex numbers:

\[
\tilde{x}_{t,j} = x_{t,j} \cdot e^{i \cdot t \cdot f_j}
\]

Frequencies are registered as a non-persistent buffer (not saved to checkpoints), and only the slice `[:T]` is used at runtime for sequences shorter than `max_seq_len`.

---

## Configuration

All hyperparameters are exposed through `TransformerConfig`:

| Parameter | Default | Description |
|---|---|---|
| `d_model` | 512 | Hidden dimension |
| `n_layers` | 16 | Total sub-layers (attn + FFN, so 8 transformer blocks) |
| `n_heads` | 8 | Number of query attention heads |
| `n_kv_heads` | 2 | Number of KV heads for GQA (`n_heads / n_kv_heads` = queries per group) |
| `vocab_size` | 32000 | Vocabulary size |
| `max_seq_len` | 2048 | Maximum sequence length |
| `ffn_mult` | 2.667 | FFN intermediate multiplier (\( \approx 8/3 \) for SwiGLU) |
| `dropout` | 0.0 | Dropout after attention and FFN |
| `attn_res_mode` | `BLOCK` | Residual mode: `NONE`, `FULL`, or `BLOCK` |
| `n_blocks` | 8 | Number of AttnRes blocks (Block mode only) |
| `eps` | 1e-6 | Epsilon for all RMSNorm layers |
| `rope_theta` | 10000.0 | RoPE base frequency |

Derived properties: `d_head = d_model / n_heads`, `n_kv_groups = n_heads / n_kv_heads`, `block_size = n_layers / n_blocks`.

---

## Memory and Complexity Analysis

| Mode | Sources stored | Memory (depth dim) | Notes |
|---|---|---|---|
| `NONE` | Current hidden state only | \( O(B \cdot T \cdot d) \) | Standard transformer |
| `FULL` | All \( L \) layer outputs | \( O(L \cdot B \cdot T \cdot d) \) | Maximal expressivity, high memory |
| `BLOCK` | \( N \) block summaries + partial | \( O(N \cdot B \cdot T \cdot d) \) | \( N \ll L \), memory-efficient |

For `FULL` AttnRes, the attention residual operator adds \( L \) pseudo-query parameters (one \( d \)-dimensional vector per sub-layer). For `BLOCK` AttnRes, it adds \( L \) pseudo-queries but only maintains \( N + 1 \) source tensors in memory at any time. The AttnRes parameters themselves are negligible: \( L \times d \) scalars total, versus millions of parameters in the projection matrices.

---

## References

- Kimi Team, MoonshotAI. *Attention Residuals* (2026). [PDF](https://github.com/MoonshotAI/Attention-Residuals/blob/master/Attention_Residuals.pdf)
- Ainslie, J. et al. *GQA: Training Generalized Multi-Query Transformer Models* (2023). arXiv:2305.13245.
- Su, J. et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021). arXiv:2104.09864.
- Shazeer, N. *GLU Variants Improve Transformer* (2020). arXiv:2002.05202.
- Milakov, M. & Gimelshein, N. *Online normalizer calculation for softmax* (2018). arXiv:1805.02867.
- Zhang, B. & Sennrich, R. *Root Mean Square Layer Normalization* (2019). arXiv:1910.07467.
