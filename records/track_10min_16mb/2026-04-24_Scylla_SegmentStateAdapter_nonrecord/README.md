# Non-record: Scylla + Segment-State Adapter

**Status:** architecture patch only. No new full training run has been completed yet.

This is a non-record architecture experiment built on the current Scylla record stack. The goal is not to change the leaderboard denominator or retokenization contract. The goal is to test whether a tiny boundary-conditioned state adapter can improve the already-strong autoregressive baseline while staying easy to audit.

## Base

Inherited from `2026-03-31_Scylla_FullGPTQ_XSA11_FA3_0.9485`:

- Scylla TokenMonster-derived 998-token vocabulary
- tokenizer metadata byte accounting via `candidate.meta.npz`
- Full Hessian GPTQ int6 export path
- XSA on all 11 layers
- coprime-stride multi-shard loader
- FlashAttention 3
- Parallel Muon + EMA/SWA

## New Idea

The new module is `SegmentStateAdapter`.

It is a small residual adapter placed after selected transformer blocks. It compresses hidden states into a low-rank channel, applies a causal depthwise state kernel over the sequence, optionally injects tokenizer boundary metadata, and projects the state back into the residual stream.

Default knobs:

```bash
SEGMENT_ADAPTER=1
SEGMENT_ADAPTER_LAYERS=8,9,10
SEGMENT_ADAPTER_RANK=16
SEGMENT_ADAPTER_KERNEL=64
SEGMENT_ADAPTER_GATE_INIT=-3.0
SEGMENT_ADAPTER_BOUNDARY_SIGNAL=1
```

Why this is interesting:

- It borrows the useful part of H-Net/adaptive tokenization: boundary-aware processing.
- It borrows the useful part of state-space models: cheap causal state mixing.
- It does not alter tokenizer decoding, validation byte counts, or the scorer.
- The adapter is zero-output initialized, so the enabled model starts close to the inherited Scylla baseline.

## Byte Accounting Rule

The byte path is intentionally unchanged:

- `base_bytes_lut`
- `has_leading_space_lut`
- `is_boundary_token_lut`
- validation BPB calculation
- sliding-window BPB calculation

Tokenizer metadata is used only as a model feature through `is_boundary_token_lut`. It is not used to change the denominator.

## First Run

Smoke run:

```bash
SEED=42 DATA_PATH=./data/datasets/fineweb10B_scylla \
TOKENIZER_PATH=./candidate.vocab TOKENIZER_META_PATH=./candidate.meta.npz \
VOCAB_SIZE=998 XSA_LAST_N=11 USE_GPTQ=0 TTT_ENABLED=0 \
SEGMENT_ADAPTER=1 SEGMENT_ADAPTER_LAYERS=8,9,10 \
SEGMENT_ADAPTER_RANK=16 SEGMENT_ADAPTER_KERNEL=64 \
ITERATIONS=20 VAL_LOSS_EVERY=10 MAX_WALLCLOCK_SECONDS=90 \
BIGRAM_VOCAB_SIZE=2816 BIGRAM_DIM=112 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Full run should only happen after:

1. syntax check passes,
2. adapter-off reproduces the base code path,
3. adapter-on smoke run logs `segment_adapter:enabled=1`,
4. byte audit confirms the validation byte count is unchanged by the adapter flag.

## Expected Submission Claim

If it improves: "Scylla plus a tiny boundary-conditioned state adapter."

If it does not improve: still potentially mergeable as a non-record architecture experiment, because it is a clean adapter patch with strict scorer isolation and a useful ablation story.

## Credits

- Scylla tokenizer and metadata base: @simon-marcus / PR #1143 lineage
- Training stack base: @resouer, @abaybektursun, and the merged Scylla record stack
- Segment-State Adapter patch: experimental non-record lane
