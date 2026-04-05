from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path


def fail(message: str) -> None:
    print(f"[preflight] ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RunPod preflight for Parameter Golf runs")
    parser.add_argument("--script", required=True, help="Training script path")
    parser.add_argument("--data-path", required=True, help="Dataset directory")
    parser.add_argument("--tokenizer-path", required=True, help="Tokenizer model path")
    parser.add_argument("--vocab-size", required=True, type=int, help="Expected tokenizer vocab size")
    parser.add_argument("--min-train-shards", type=int, default=1, help="Minimum number of train shards required")
    parser.add_argument("--min-val-shards", type=int, default=1, help="Minimum number of val shards required")
    parser.add_argument("--min-gpus", type=int, default=1, help="Minimum visible CUDA devices required")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script = Path(args.script)
    if not script.is_file():
        fail(f"training script not found: {script}")

    data_path = Path(args.data_path)
    if not data_path.is_dir():
        fail(f"dataset directory not found: {data_path}")

    tokenizer_path = Path(args.tokenizer_path)
    if not tokenizer_path.is_file():
        fail(f"tokenizer not found: {tokenizer_path}")

    train_files = sorted(glob.glob(str(data_path / "fineweb_train_*.bin")))
    val_files = sorted(glob.glob(str(data_path / "fineweb_val_*.bin")))
    if len(train_files) < args.min_train_shards:
        fail(
            f"found {len(train_files)} train shards under {data_path}, "
            f"need at least {args.min_train_shards}"
        )
    if len(val_files) < args.min_val_shards:
        fail(
            f"found {len(val_files)} val shards under {data_path}, "
            f"need at least {args.min_val_shards}"
        )

    try:
        import sentencepiece as spm
    except Exception as exc:  # pragma: no cover
        fail(f"sentencepiece import failed: {exc}")

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        fail(f"torch import failed: {exc}")

    try:
        import flash_attn_interface  # noqa: F401
    except Exception as exc:  # pragma: no cover
        fail(f"flash_attn_interface import failed: {exc}")

    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() is false")
    gpu_count = torch.cuda.device_count()
    if gpu_count < args.min_gpus:
        fail(f"only {gpu_count} visible CUDA device(s), need at least {args.min_gpus}")

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    actual_vocab = int(sp.vocab_size())
    if actual_vocab != args.vocab_size:
        fail(f"tokenizer vocab mismatch: expected {args.vocab_size}, got {actual_vocab}")

    print("[preflight] OK")
    print(f"[preflight] script={script}")
    print(f"[preflight] data_path={data_path}")
    print(f"[preflight] tokenizer_path={tokenizer_path}")
    print(f"[preflight] train_shards={len(train_files)}")
    print(f"[preflight] val_shards={len(val_files)}")
    print(f"[preflight] vocab_size={actual_vocab}")
    print(f"[preflight] torch={torch.__version__}")
    print(f"[preflight] cuda_device_count={gpu_count}")
    for idx in range(gpu_count):
        print(f"[preflight] gpu[{idx}]={torch.cuda.get_device_name(idx)}")


if __name__ == "__main__":
    main()
