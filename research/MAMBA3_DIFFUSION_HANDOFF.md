# Mamba-3 + Diffusion Planning Handoff

Date: 2026-04-24

## Current Goal

Build a serious Parameter Golf non-record submission, not a toy. The submission must still follow Parameter Golf rules:

- legal 16MB artifact,
- real `val_bpb`,
- runnable `train_gpt.py`,
- logs,
- clear README,
- no scorer/byte-accounting tricks.

User wants real architecture work around:

- Mamba-3 / state-space models,
- diffusion,
- later H-Net / E2E TTT if needed.

## Current Repo / Branch

Local worktree:

`E:\Coding\00_active\parameter_golf_nonrecord`

Current serious Mamba branch:

`boobie/mamba3-mimo-scylla`

Remote branch:

`https://github.com/Rtx09x/parameter-golf/tree/boobie/mamba3-mimo-scylla`

Main record folder:

`records/track_non_record_16mb/2026-04-24_Mamba3_MIMO_Scylla`

RunPod launcher:

`runpod/mamba3_scylla_runpod.sh`

## Mamba-3 Architecture

Current legal V1:

- official `Mamba3` from `state-spaces/mamba`,
- Scylla TokenMonster tokenizer, vocab `998`,
- `fineweb_scylla` pre-tokenized data,
- unchanged Scylla byte accounting,
- `9` Mamba-3 MIMO blocks,
- `model_dim=512`,
- `d_state=128`,
- `headdim=64`,
- `expand=2`,
- `is_mimo=1`,
- `mimo_rank=4`,
- `chunk_size=16`,
- BigramHash `2816 x 112`,
- causal SmearGate,
- tied embeddings/head,
- AdamW optimizer,
- packed int6 + LZMA export.

Training full defaults:

```bash
ITERATIONS=20000
MAX_TRAINING_SECONDS=4800
NPROC_PER_NODE=1
```

Important: `MAX_TRAINING_SECONDS` stops only the training loop. EMA/export/final eval continue after the training budget.

## Important Bug Found And Fixed

There was a non-causal SmearGate bug:

```python
right = torch.roll(x, shifts=-1, dims=1)
```

That leaked the next input token while predicting `y[i] = input[i+1]`, causing absurd train loss around `0.295` by step 100.

Fixed in commit:

`7e75fca Make Mamba SmearGate causal`

Current causal version:

```python
x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
return (1 - g) * x + g * x_prev
```

Do not trust any run before that fix.

## Current Training Signal

After the causal fix, the run looks sane:

```text
step:1/20000 train_loss:6.9054
step:1/20000 val_loss:6.7945 val_bpb:3.8007
step:100/20000 train_loss:4.3669
step:200/20000 train_loss:3.3976
```

Current speed:

```text
~2.47s/step on 1 GPU
```

With `MAX_TRAINING_SECONDS=4800`, expected completed steps:

```text
~1900-2100 steps
```

First meaningful validation is expected at step 1000.

Interpretation guide:

- `val_bpb > 3.0`: weak / slow,
- `2.0-3.0`: learning but needs more time/8GPU,
- `1.5-2.0`: promising,
- `<1.5`: very strong, audit hard,
- `<1.2`: either breakthrough or bug until proven.

## Validity Guards

Already added:

`runpod/mamba3_causality_smoke.py`

It changes future tokens while keeping prefix fixed and checks that prefix logits do not change. Run this before trusting final results:

```bash
cd /workspace/parameter-golf
git pull --ff-only
REPO_ROOT=/workspace/parameter-golf python runpod/mamba3_causality_smoke.py
```

Expected good result:

```text
'mamba3_causality': 'checked'
'prefix_max_abs_diff': 0.0-ish
```

Other watch items:

- final artifact must be under `16,000,000` bytes,
- trust `final_int6_sliding_window_s64_exact`, not only non-sliding validation,
- DDP/8GPU path is not yet fully tested,
- quantization may hurt Mamba more than transformers.

## Diffusion Combo Idea

Do not use diffusion sampling at eval. That risks legality/scoring complexity and slow eval.

Use:

**Mamba-3 AR + diffusion auxiliary training**

Main legal scoring path stays AR:

```text
input tokens -> predict next token -> cross entropy -> BPB
```

Add a training-only auxiliary denoising loss:

```text
corrupt/mask input tokens -> trunk hidden states -> reconstruct original tokens at corrupted positions
```

Eval/export:

- no diffusion sampling,
- no iterative denoising,
- no scorer change,
- no byte-accounting change,
- same `forward_logits`,
- same BPB.

Suggested default knobs:

```bash
DIFFUSION_AUX=1
DIFFUSION_MASK_PROB=0.15
DIFFUSION_LOSS_WEIGHT=0.10
DIFFUSION_RANDOM_REPLACE_PROB=0.10
DIFFUSION_HEAD_TIED=1
```

Safer first variant:

```bash
DIFFUSION_LOSS_WEIGHT=0.05
DIFFUSION_MASK_PROB=0.10
```

Use tied token embedding/head for the aux reconstruction head to avoid growing artifact size.

## Recommended Diffusion Implementation

Patch `MambaGolfLM.forward` only for training:

1. Keep the normal AR loss exactly as-is.
2. If `DIFFUSION_AUX=1` and `self.training`:
   - create a corrupted copy of `input_ids`,
   - randomly choose mask positions,
   - replace selected tokens with random tokens or a fixed token ID,
   - run hidden state on corrupted input,
   - predict original `input_ids` at corrupted positions,
   - add `loss_weight * aux_loss`.
3. `forward_logits` must remain unchanged and use clean input.
4. Eval functions must not call diffusion logic.
5. Export must not require additional diffusion-only state if head is tied.

Pseudo-code:

```python
def forward(self, input_ids, target_ids):
    clean_hidden = self.hidden(input_ids)
    ar_logits = self.logits_from_hidden(clean_hidden)
    ar_loss = cross_entropy(ar_logits, target_ids)

    if self.training and self.diffusion_aux:
        corrupted, mask = corrupt(input_ids)
        h = self.hidden(corrupted)
        logits = self.logits_from_hidden(h)
        aux_loss = cross_entropy(logits[mask], input_ids[mask])
        return ar_loss + self.diffusion_loss_weight * aux_loss

    return ar_loss
```

This is legally clean because final scoring still uses the standard AR logits.

## Diffusion Experiment Plan

Do not start diffusion before reading the Mamba-only step-1000 validation.

If Mamba-only looks promising:

1. Create branch:

   `boobie/mamba3-diffusion-aux-scylla`

2. Copy from current Mamba branch.

3. Add diffusion auxiliary knobs.

4. Smoke:

   ```bash
   DIFFUSION_AUX=1 DIFFUSION_LOSS_WEIGHT=0.05 DIFFUSION_MASK_PROB=0.10 bash runpod/mamba3_scylla_runpod.sh smoke
   ```

5. Short full-ish proof:

   ```bash
   MAX_TRAINING_SECONDS=4800 DIFFUSION_AUX=1 DIFFUSION_LOSS_WEIGHT=0.05 DIFFUSION_MASK_PROB=0.10 bash runpod/mamba3_scylla_runpod.sh full
   ```

6. Compare to Mamba-only at similar steps/time.

If it improves validation BPB, run larger 8GPU/longer seeds.

## Copy-Paste Prompt For New Chat

Use this prompt to continue in another chat:

```text
We are working in E:\Coding\00_active\parameter_golf_nonrecord on OpenAI Parameter Golf.

Goal: serious non-record submission, not toy. Must obey legal BPB scoring, 16MB artifact, runnable train_gpt.py, logs, README. User wants Mamba-3/state-space and diffusion.

Current branch: boobie/mamba3-mimo-scylla
Remote: https://github.com/Rtx09x/parameter-golf/tree/boobie/mamba3-mimo-scylla
Record folder: records/track_non_record_16mb/2026-04-24_Mamba3_MIMO_Scylla
RunPod launcher: runpod/mamba3_scylla_runpod.sh

Current Mamba architecture:
- official Mamba3 from state-spaces/mamba
- Scylla tokenizer/data, vocab 998
- unchanged byte accounting
- 9 Mamba3 MIMO blocks
- dim 512, d_state 128, headdim 64, mimo_rank 4, chunk_size 16
- BigramHash 2816x112
- causal SmearGate
- tied embeddings/head
- AdamW
- packed int6+LZMA export

Important bug fixed:
Previous SmearGate used right neighbor / future token leak. Fixed in commit 7e75fca with causal previous-token-only SmearGate. Do not trust runs before this.

Current causal Mamba-only training signal:
step 1 train_loss 6.9054, val_bpb 3.8007
step 100 train_loss 4.3669
step 200 train_loss 3.3976
speed about 2.47s/step on 1 GPU
Need step 1000 val_bpb before judging.

Causality smoke added:
runpod/mamba3_causality_smoke.py
Run before trusting final results:
REPO_ROOT=/workspace/parameter-golf python runpod/mamba3_causality_smoke.py

Next idea: Mamba-3 AR + diffusion auxiliary training. Do NOT do diffusion sampling at eval. Add training-only denoising aux loss while keeping forward_logits/eval/scoring unchanged.
Suggested knobs:
DIFFUSION_AUX=1
DIFFUSION_MASK_PROB=0.10 or 0.15
DIFFUSION_LOSS_WEIGHT=0.05 or 0.10
DIFFUSION_RANDOM_REPLACE_PROB=0.10
DIFFUSION_HEAD_TIED=1

Please inspect the actual current files before editing. Implement diffusion auxiliary cleanly only after checking Mamba-only step-1000 validation signal. Keep export under 16MB and do not alter byte accounting.
```

## Short Prompt If You Only Want Planning

```text
Continue planning Parameter Golf Mamba-3 + diffusion. We have a legal Mamba-3 MIMO Scylla branch with causal fix. Current Mamba-only run is in progress; step 200 train_loss 3.3976, first serious validation is step 1000. Need a serious non-record plan: Mamba-3 AR base plus training-only diffusion auxiliary loss, no eval diffusion, no scorer changes, artifact under 16MB. Give implementation plan, risks, and run ladder.
```
