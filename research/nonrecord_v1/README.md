# Non-Record V1: Scylla + Segment-State Adapter

Status: planning scaffold, not a submission yet.

## Mission

Build a mergeable Parameter Golf non-record submission that is genuinely new, but does not throw away the performance machinery from the best public work.

The target is not main leaderboard rank. The target is a clean, credible research PR:

- weird enough to fit the non-record request,
- strong enough to not look like a toy,
- runnable and auditable,
- honest about ablations and negative results.

## Current Best Base

Use the merged Scylla record as the first base:

`records/track_10min_16mb/2026-03-31_Scylla_FullGPTQ_XSA11_FA3_0.9485/`

Why:

- strongest local merged record in current `origin/main`,
- no TTT needed,
- has tokenizer metadata and explicit byte accounting,
- compact single-folder record surface,
- already contains the modern performance stack:
  - Scylla tokenizer,
  - XSA on all layers,
  - Full Hessian GPTQ,
  - FlashAttention 3,
  - Parallel Muon / banking,
  - EMA/SWA,
  - BigramHash + SmearGate.

## Proposed Novel Idea

Add a small **Segment-State Adapter**:

- It is inspired by H-Net/adaptive tokenization and state-space memory.
- It does not change the tokenizer or byte accounting.
- It uses existing tokenizer metadata such as leading-space and boundary flags.
- It maintains a lightweight recurrent segment state across tokens.
- It injects that state as a residual adapter in selected late layers.

This makes the research idea clear:

> Can a tiny recurrent segment-level memory improve or diagnose Scylla-tokenized compression without changing the tokenization or evaluation math?

## Why This Beats Starting With Diffusion

Diffusion is mergeable as a research story, but public attempts are weak on BPB. A diffusion-first run risks becoming another interesting negative result.

This V1 keeps the strong Scylla AR path and adds one weird mechanism with controlled blast radius.

## Why This Beats Starting With GDN/FLA

GatedDeltaNet / FLA is interesting, but recent public discussion shows BPB-byte-accounting risk in that family. We can borrow the state-space instinct without inheriting the scorer controversy.

## First Implementation Shape

Minimal knobs:

- `SEGMENT_ADAPTER=1`
- `SEGMENT_ADAPTER_LAYERS=8,9,10`
- `SEGMENT_ADAPTER_DIM=64`
- `SEGMENT_ADAPTER_RANK=16`
- `SEGMENT_ADAPTER_GATE_INIT=-3.0`
- `SEGMENT_ADAPTER_RESET_ON_BOUNDARY=1`

Low-risk design:

- zero or near-zero initialized residual gate,
- late layers only,
- no tokenizer changes,
- no eval-time adaptation,
- no n-gram cache,
- no validation leakage.

## Required Ablations

Run ladder:

1. `baseline_scylla_repro` - exact base command, ideally one seed.
2. `segment_adapter_smoke` - adapter enabled, short smoke.
3. `segment_adapter_1seed` - one full run.
4. `adapter_off_same_code` - same code with adapter disabled, to prove no scoring drift.
5. `byte_audit` - verify tokenizer byte totals match canonical metadata path.

## Success Criteria

Minimum mergeable non-record:

- code compiles from inside the record folder,
- one real training log,
- artifact under 16,000,000 bytes,
- exact BPB printed,
- README explains motivation and result honestly,
- byte accounting unchanged and audited.

Strong non-record:

- adapter improves or roughly matches baseline,
- includes adapter-on/off ablation,
- includes analysis of learned segment gates/state norms,
- clear relationship to requested H-Net / state-space ideas.

## Do Not Do In V1

- Do not change tokenizer yet.
- Do not add TTT yet.
- Do not add diffusion objective yet.
- Do not combine many weird ideas at once.
- Do not touch old `salvage-export` artifacts.
