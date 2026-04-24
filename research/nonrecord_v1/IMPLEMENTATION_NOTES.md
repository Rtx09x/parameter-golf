# Implementation Notes

## First Patch Target

Started from:

`records/track_10min_16mb/2026-03-31_Scylla_FullGPTQ_XSA11_FA3_0.9485/train_gpt.py`

New non-record folder:

`records/track_10min_16mb/2026-04-24_Scylla_SegmentStateAdapter_nonrecord/`

Status: implemented and syntax-checked. No full training run yet.

## Likely Patch Points

Inspected first patch points in the Scylla record script:

- `Hyperparameters` starts at `train_gpt.py:22`; add adapter env knobs there.
- `build_sentencepiece_luts` starts at `train_gpt.py:261`; do not change scoring semantics.
- `load_tokenizer_meta_luts_np` starts at `train_gpt.py:321`; adapter can reuse metadata-derived boundary flags only as model features.
- `Block` starts at `train_gpt.py:952`; late-layer residual insertion can live after the normal block output.
- `GPT` starts at `train_gpt.py:983`; add adapter construction and non-persistent boundary buffers here.
- `GPT.forward` starts at `train_gpt.py:1102`; training path needs adapter hook.
- `GPT.forward_logits` starts at `train_gpt.py:1139`; eval/logits path must mirror training path exactly.
- Sliding byte accounting happens around `train_gpt.py:1232` and `train_gpt.py:1343`; leave unchanged except for extra audit logging.
- Main model construction passes args around `train_gpt.py:1759` and `train_gpt.py:2095`; both train and eval models need the same adapter args.

## Adapter Sketch

Input:

- hidden state `x`,
- token ids,
- boundary flags from metadata or token-derived LUT.

State:

- small per-sequence recurrent segment vector,
- reset or decay at token boundaries,
- update from current hidden projection.

Output:

- low-rank residual adapter,
- gated near zero at init.

Sketch:

```python
seg = seg * decay
seg = torch.where(boundary, reset_value, seg)
seg = seg + update_proj(x)
x = x + gate * out_proj(seg)
```

The actual implementation must be vectorized and cheap enough not to destroy step count.

## Implemented V1

The V1 adapter is intentionally simpler than a hand-rolled recurrent loop:

- low-rank hidden projection,
- causal depthwise state kernel,
- optional tokenizer-boundary signal from `is_boundary_token_lut`,
- low-rank projection back to the residual stream,
- sigmoid gate initialized from `SEGMENT_ADAPTER_GATE_INIT`,
- zero-initialized output projection so the enabled model starts close to baseline.

The byte path is unchanged. Boundary metadata is only a model feature.

## Expected Failure Modes

- too slow per step,
- artifact grows too much,
- adapter learns nothing because gate stays near zero,
- adapter hurts because extra objective-free recurrence destabilizes quantization,
- byte audit passes but BPB regresses.

All of those can still make a good non-record if documented cleanly.
