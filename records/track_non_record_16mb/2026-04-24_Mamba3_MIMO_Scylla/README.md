# Non-record: Mamba-3 MIMO + Scylla

**Status:** proof run passed; upgraded for the 8-GPU full run.

This submission is a real state-space architecture lane for Parameter Golf. It uses the official Mamba-3 block from `state-spaces/mamba` and keeps the strongest proven parts of the current public stack around it:

- Scylla TokenMonster tokenizer, vocab `998`
- public pre-tokenized `fineweb_scylla` shards
- unchanged Scylla tokenizer metadata byte accounting
- coprime multi-shard data loader
- legal validation BPB path
- compressed int6 packed artifact roundtrip

## Architecture

Default V1:

| Component | Value |
|---|---:|
| Backbone | official `Mamba3` |
| Blocks | `9` |
| Model dim | `512` |
| State dim | `128` |
| Head dim | `64` |
| Expand | `2` |
| MIMO | enabled |
| MIMO rank | `4` |
| Chunk size | `16` |
| RoPE fraction | `0.5` |
| Tokenizer | Scylla TokenMonster |
| Vocab size | `998` |
| Extra input features | BigramHash `3072 x 112`, SmearGate |
| TTT | off |

The model is a stack of RMSNorm + Mamba-3 residual blocks with tied token embedding/head.

## Proof Run

A causal, export-legal 1-GPU proof run completed before the 8-GPU patch:

```text
steps_completed: 2045
step:2000 val_loss:2.2939 val_bpb:1.2832
final_int6_roundtrip_exact val_loss:2.30995808 val_bpb:1.29215887
Total submission size int6pack+lzma: 14469827 bytes
```

This proved the Mamba-3 lane trains, exports, and stays under 16MB. The run used a wall-clock LR schedule that decayed too early, so the 8-GPU path now uses step-based warmdown.

## Run Policy

Training is time-budgeted. Export and final evaluation are not interrupted by the training wall clock.

Default full run:

```bash
ITERATIONS=20000
MAX_TRAINING_SECONDS=999999
WARMDOWN_ITERS=4000
BIGRAM_VOCAB_SIZE=3072
LZMA_PRESET=9
```

The training wall clock still only gates the train loop if explicitly set, but the learning-rate schedule is step-based. This prevents short proof budgets from crushing the LR early.

Export evaluates raw and EMA weights, logs both, and exports whichever has the better validation BPB.

## Why This Is Non-record

This is not targeted at the 10-minute SOTA leaderboard initially. It is targeted at the requested non-record state-space lane:

- real Mamba-3 implementation,
- legal artifact under 16MB if export succeeds,
- unchanged byte denominator,
- reproducible logs,
- measured BPB.

## Expected Evidence Before PR

Required:

- smoke log showing Mamba-3 import/forward/backward,
- byte-audit output,
- train log from the full run,
- `final_int6_roundtrip_exact`,
- `Total submission size int6pack+lzma`,
- `final_int6_sliding_window_s64_exact` if full sliding eval is enabled.

## Credits

- Mamba-3 implementation: official `state-spaces/mamba`
- Tokenizer/data/byte-accounting base: Scylla record lineage
- Loader/export structure: Parameter Golf record scripts
