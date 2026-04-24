# Diffusion-Only Scylla RunPod Path

Branch:

```bash
boobie/diffusion-only-scylla
```

Run smoke first:

```bash
!set -e; \
cd /workspace; \
if [ -d parameter-golf ] && [ ! -d parameter-golf/.git ]; then mv parameter-golf parameter-golf_broken_$(date +%s); fi; \
if [ ! -d parameter-golf ]; then git clone -b boobie/diffusion-only-scylla https://github.com/Rtx09x/parameter-golf.git parameter-golf; fi; \
cd /workspace/parameter-golf; \
git fetch origin boobie/diffusion-only-scylla; \
git checkout boobie/diffusion-only-scylla; \
git pull --ff-only; \
bash runpod/diffusion_only_scylla_runpod.sh smoke
```

If smoke passes, run full on the 1x H100 SXM pod:

```bash
!set -e; \
cd /workspace/parameter-golf; \
git pull --ff-only; \
NPROC_PER_NODE=1 bash runpod/diffusion_only_scylla_runpod.sh full
```

Full defaults:

- `NPROC_PER_NODE=1`
- `ITERATIONS=20000`
- `MAX_TRAINING_SECONDS=4800`
- `RUN_SLIDING_EVAL=1`
- Scylla TokenMonster tokenizer, vocab `998`
- causal diffusion steps `8`
- `1024` token context
- `131,072` train tokens per optimizer step
- `MLP_MULT=2.4`
- `QK_GAIN_INIT=5.0`
- `RECUR_LAYERS=3,4,5`
- `RECUR_REPEATS=1`
- `WEIGHT_DECAY=0.09`

The model is diffusion-only in the sense that every prediction is made through the causal denoising path. Validation remains legal because `forward_logits()` receives only prefix tokens plus an independent mask/noise slot, never future validation tokens.
