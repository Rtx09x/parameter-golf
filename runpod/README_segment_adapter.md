# Scylla Segment Adapter RunPod Path

This is the paid-run-safe path for the non-record Segment-State Adapter experiment.

## Notebook Cell: Smoke First

Run this in a fresh RunPod notebook:

```bash
!set -e; \
cd /workspace; \
if [ ! -d parameter-golf ]; then git clone -b boobie/nonrecord-state-space-hnet https://github.com/Rtx09x/parameter-golf.git parameter-golf; fi; \
cd /workspace/parameter-golf; \
git pull --ff-only; \
bash runpod/scylla_segment_adapter_runpod.sh smoke
```

This does:

1. install missing deps,
2. download public pre-tokenized Scylla data from Hugging Face,
3. run byte-accounting audit,
4. run a 20-iteration adapter-on smoke test on 1 GPU.

Do not run full training until smoke logs include:

- `segment_adapter:enabled=1`
- validation/audit JSON with train and val shard counts
- no crash before first validation

## Notebook Cell: Full Run

After smoke passes:

```bash
!set -e; \
cd /workspace/parameter-golf; \
git pull --ff-only; \
bash runpod/scylla_segment_adapter_runpod.sh full
```

Defaults:

- all visible GPUs,
- seed `42`,
- `ITERATIONS=6716`,
- `USE_GPTQ=1`,
- `TTT_ENABLED=0`,
- `SEGMENT_ADAPTER=1`,
- late layers `8,9,10`.

Override example:

```bash
!cd /workspace/parameter-golf && SEED=1337 SEGMENT_ADAPTER_RANK=8 bash runpod/scylla_segment_adapter_runpod.sh full
```

## Cheap Audit Only

```bash
!cd /workspace/parameter-golf && bash runpod/scylla_segment_adapter_runpod.sh audit
```

Use this if the pod setup looks suspicious and you want to validate the data path before starting training.
