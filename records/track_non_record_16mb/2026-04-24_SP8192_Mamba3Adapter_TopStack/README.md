# Non-record: SP8192 Top Stack + Mamba-3 Adapter Majority

This is a non-record hybrid experiment built from the current SP8192 frontier recipe and augmented with real Mamba-3 MIMO adapters.

## What changed from the SP8192 top stack

- Keeps SP8192 tokenizer/data path, 11x512 transformer stack, XSA, 3-layer depth recurrence, parallel residuals, QK gain, MuonEq-R, GPTQ SDClip, Brotli compression, and legal score-first TTT support.
- Adds real Mamba-3 MIMO adapters to the last 6 of 11 physical blocks by default, so more than half the stack is Mamba-augmented.
- Defaults the adapter to a bottleneck dimension of 48 with d_state 64, headdim 32, MIMO rank 2. The first 64-dim smoke was functional but exported at 16,212,587 bytes, so 48 is the legal-size default.
- Uses `MLP_MULT=3.75` in the RunPod recipe to buy artifact room for the Mamba path.
- Disables `torch.compile` by default (`COMPILE_ENABLED=0`) because TileLang Mamba kernels are less reliable under Dynamo fullgraph capture.

## Default proof run

Use `runpod/sp8192_mamba_topstack_runpod.sh proof` for a 2k-step sanity run. If it is not clearly ahead of the Scylla Mamba line by 1k-2k steps, do not spend the full run.

## Sparse training option

Magnitude sparsity is available but disabled by default. It linearly ramps from `SPARSE_START_STEP` to `SPARSE_END_STEP`, reapplies masks after optimizer steps, and only touches matrices selected by `SPARSE_INCLUDE`.

Recommended Mamba-capacity probe:

```bash
MAMBA_ADAPTER_DIM=56 SPARSE_ENABLED=1 SPARSE_TARGET=0.10 SPARSE_INCLUDE=mamba_adapter \
  bash runpod/sp8192_mamba_topstack_runpod.sh proof
```

Riskier big-sparse probe:

```bash
MODEL_DIM=608 MLP_MULT=3.75 MAMBA_ADAPTER_DIM=64 \
SPARSE_ENABLED=1 SPARSE_TARGET=0.40 SPARSE_START_STEP=1000 SPARSE_END_STEP=6000 SPARSE_INCLUDE=mlp,attn \
  bash runpod/sp8192_mamba_topstack_runpod.sh proof
```

This is the intended ~50M-parameter big sparse probe. The sparsity schedule leaves embeddings, norms, gates, and tiny scalar/control tensors dense.

## Default full run

Use `runpod/sp8192_mamba_topstack_runpod.sh full` on 8xH100. This trains the hybrid with SP8192 data and exports a legal-style artifact.

## Verification status

Implemented and syntax-checked locally. Full GPU smoke/training must be run on RunPod because this path requires CUDA, FlashAttention3, Mamba-3/TileLang kernels, and SP8192 data.
