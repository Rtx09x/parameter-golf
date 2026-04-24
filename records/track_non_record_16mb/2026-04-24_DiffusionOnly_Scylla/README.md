# Non-record: Causal Diffusion-Only + Scylla

**Status:** implemented and syntax-checked locally. RunPod smoke/full training still needs to be run.

This is a serious non-record diffusion-only lane for Parameter Golf. It is intentionally not Mamba diffusion. The model is a causal denoising transformer that predicts each next token from:

- prefix token context,
- an independent noisy current-token slot,
- a diffusion time-step embedding.

Validation stays legal because `forward_logits()` uses only prefix tokens and a fixed independent mask/noise slot. It does not see future validation tokens and does not use bidirectional masked-token reconstruction.

## Architecture

Default V1:

| Component | Value |
|---|---:|
| Backbone | causal diffusion-denoising transformer |
| Layers | `8` |
| Model dim | `512` |
| Heads | `8` |
| MLP mult | `2.4` |
| Attention | QK-Gain `5.0` |
| Residual style | parallel attention + MLP residual lanes |
| Depth recurrence | repeat layers `3,4,5` once |
| Diffusion steps | `8` |
| Eval noise step | `8` |
| Train/eval context | `1024` |
| Train tokens / step | `131,072` |
| Tokenizer | Scylla TokenMonster |
| Vocab size | `998` |
| Extra input features | BigramHash `2816 x 112`, causal SmearGate |
| Export | packed int6 + LZMA |

## Legal Scoring Contract

The legal BPB path is unchanged:

```text
prefix tokens -> forward_logits(prefix) -> next-token cross entropy -> BPB
```

The diffusion path is legal because the noisy current-token slot is independent of the true validation target at eval. There is no eval-time bidirectional denoising, no full-sequence reconstruction, no future-token access, and no tokenizer byte-accounting change.

## What Was Borrowed From The Top Runs

The current leaderboard is dominated by SP8192 stacks with recurrence, parallel residuals, legal score-first TTT, tuned quantization, and strict export discipline. This branch borrows the reliability lessons, not the exact architecture:

- fixed full-validation BPB,
- Scylla byte-accounting metadata,
- parallel residual blocks,
- middle-layer depth recurrence with shared weights,
- high QK gain,
- higher default weight decay for compression headroom,
- explicit roundtrip eval after int6+LZMA export,
- size logging before claiming legality,
- RunPod smoke before full training.

## Run

Smoke:

```bash
bash runpod/diffusion_only_scylla_runpod.sh smoke
```

Full 1x H100 SXM:

```bash
NPROC_PER_NODE=1 bash runpod/diffusion_only_scylla_runpod.sh full
```

## Evidence Required Before Treating It As A Real Result

- `diffusion_only_forward` smoke output,
- train log from full run,
- `final_int6_roundtrip_exact`,
- `Total submission size int6pack+lzma`,
- `final_int6_sliding_window_s64_exact` if full sliding eval is enabled.

## Current Submission Status

No final BPB has been produced yet. `submission.json` is intentionally pending until RunPod smoke/full runs complete.
