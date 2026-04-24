# Non-record: Mamba-3 MIMO + Scylla

**Status:** implementation ready for RunPod smoke/full runs. No final BPB has been produced yet.

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
| Extra input features | BigramHash `2816 x 112`, SmearGate |
| TTT | off |

The model is a stack of RMSNorm + Mamba-3 residual blocks with tied token embedding/head.

## Run Policy

Training is time-budgeted. Export and final evaluation are not interrupted by the training wall clock.

Default full run:

```bash
ITERATIONS=20000
MAX_TRAINING_SECONDS=4800
```

So the run trains for up to 20k steps, but stops the train loop after about 80 minutes and then continues with EMA, int6 export, roundtrip eval, and optional sliding eval.

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
