# Byte Accounting Audit Plan

This is mandatory because tokenizer and state-space submissions are easy to invalidate with subtle BPB mistakes.

## Rule

For V1, do not change tokenization or evaluation math.

The adapter may use metadata as features, but the score denominator must remain the Scylla metadata/canonical path.

## Checks

1. Compare total bytes from runtime metadata against decoded validation bytes on a sample.
2. Confirm leading-space handling is applied exactly once.
3. Confirm special, unused, and byte tokens follow the existing Scylla record behavior.
4. Confirm adapter-disabled code produces identical BPB to base within normal floating variance.
5. Log:
   - total tokens,
   - total bytes,
   - bytes/token,
   - val_loss,
   - val_bpb.

## Danger Signs

- `val_loss / log(2) / val_bpb` changes unexpectedly between baseline and adapter.
- bytes/token jumps without a tokenizer change.
- scorer code touches adapter state or eval-time learned statistics.
- metadata files differ from the record baseline without a written reason.
