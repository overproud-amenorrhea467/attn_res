import time

import torch

from attn_res import AttnResMode, AttnResTransformer, TransformerConfig

"""Run a quick forward pass to verify the model works."""

print("=" * 70)
print("Attention Residuals + GQA Transformer — Smoke Test")
print("=" * 70)

device = "cuda" if torch.cuda.is_available() else "cpu"

for mode in AttnResMode:
    cfg = TransformerConfig(
        d_model=256,
        n_layers=12,  # 6 transformer blocks (attn + mlp each)
        n_heads=8,
        n_kv_heads=2,  # GQA with 4 query heads per KV group
        vocab_size=1000,
        max_seq_len=512,
        attn_res_mode=mode,
        n_blocks=4,  # for Block AttnRes
    )

    model = AttnResTransformer(cfg).to(device)
    tokens = torch.randint(0, cfg.vocab_size, (2, 64), device=device)
    targets = torch.randint(0, cfg.vocab_size, (2, 64), device=device)

    # Forward pass
    t0 = time.perf_counter()
    logits, loss = model(tokens, targets)
    t1 = time.perf_counter()

    params = model.count_parameters()

    print(f"\nMode: {mode.value:>6s}")
    print(f"  Logits shape : {tuple(logits.shape)}")
    print(f"  Loss         : {loss.item():.4f}")
    print(f"  Forward time : {(t1 - t0) * 1000:.1f} ms")
    print(f"  Total params : {params['total']:,}")
    print(f"  AttnRes params: {params['attn_res']:,}")

print("\n" + "=" * 70)
print("All modes passed!")
print("=" * 70)
