# Mamba-3 MIMO Scylla RunPod Path

Run smoke first:

```bash
!set -e; \
cd /workspace; \
if [ -d parameter-golf ] && [ ! -d parameter-golf/.git ]; then mv parameter-golf parameter-golf_broken_$(date +%s); fi; \
if [ ! -d parameter-golf ]; then git clone -b boobie/mamba3-mimo-scylla https://github.com/Rtx09x/parameter-golf.git parameter-golf; fi; \
cd /workspace/parameter-golf; \
git fetch origin boobie/mamba3-mimo-scylla; \
git checkout boobie/mamba3-mimo-scylla; \
git pull --ff-only; \
bash runpod/mamba3_scylla_runpod.sh smoke
```

If smoke passes, run full:

```bash
!set -e; \
cd /workspace/parameter-golf; \
git pull --ff-only; \
bash runpod/mamba3_scylla_runpod.sh full
```

Full defaults:

- `ITERATIONS=20000`
- `MAX_TRAINING_SECONDS=4800`
- `RUN_SLIDING_EVAL=1`
- all visible GPUs unless `NPROC_PER_NODE=1` is set.

The training wall clock only stops the training loop. EMA/export/final eval run after training stops.
