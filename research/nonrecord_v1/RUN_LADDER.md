# Run Ladder

This file is the compute discipline checklist. No expensive run should happen before the earlier gates pass.

## Stage 0: Local Code Prep

Goal: make the modified record folder compile.

Commands:

```bash
cd /workspace/parameter-golf
python -m py_compile records/track_non_record_16mb/<folder>/train_gpt.py
```

Pass condition:

- no syntax error,
- no missing local imports from the record folder.

## Stage 1: Dataset Prep

Use Scylla assets from the record folder first. Do not rebuild tokenizer until necessary.

Needed files:

- `candidate.vocab`
- `candidate.meta.npz`
- Scylla-retokenized FineWeb dataset.

Open question:

- Whether we can reuse an existing Scylla dataset artifact or need to run `retokenize.py`.

## Stage 2: Baseline Reproduction

Run the unmodified Scylla record for one seed.

Expected target:

- approximately `0.9485` 3-seed mean in the record,
- one-seed variance expected.

Purpose:

- validate RunPod image,
- validate data path,
- validate byte accounting,
- validate artifact export.

## Stage 3: Adapter Smoke

Run adapter enabled with a short wallclock.

Suggested overrides:

```bash
SEGMENT_ADAPTER=1 \
SEGMENT_ADAPTER_LAYERS=8,9,10 \
SEGMENT_ADAPTER_DIM=64 \
SEGMENT_ADAPTER_RANK=16 \
MAX_WALLCLOCK_SECONDS=180 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Pass condition:

- training loss decreases,
- no NaNs,
- export path still runs or is explicitly skipped for smoke,
- byte audit line still prints.

## Stage 4: One Full Run

Run one full 8xH100 or long non-record run only after smoke passes.

Pass condition:

- artifact under `16,000,000`,
- final BPB available,
- baseline-compatible byte accounting,
- runtime and artifact sizes logged.

## Stage 5: Submission Package

Create:

- `README.md`
- `submission.json`
- `train_gpt.py`
- run logs
- optional `requirements.txt`
- byte audit note
