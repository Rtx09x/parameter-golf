from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch


def load_train_module(path: Path):
    if "flash_attn_interface" not in sys.modules:
        stub = types.ModuleType("flash_attn_interface")
        stub.flash_attn_func = lambda *args, **kwargs: None
        sys.modules["flash_attn_interface"] = stub
    spec = importlib.util.spec_from_file_location("segment_adapter_train_gpt", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load train script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_eval_bytes(tokens: torch.Tensor, base: torch.Tensor, leading: torch.Tensor, boundary: torch.Tensor, seq_len: int):
    total_seqs = (tokens.numel() - 1) // seq_len
    usable = total_seqs * seq_len
    x = tokens[:usable].reshape(-1, seq_len).long()
    y = tokens[1: usable + 1].reshape(-1, seq_len).long()
    prev_ids = x.reshape(-1)
    tgt_ids = y.reshape(-1)
    token_bytes = base[tgt_ids].to(dtype=torch.int64)
    token_bytes += (leading[tgt_ids] & ~boundary[prev_ids]).to(dtype=torch.int64)
    return {
        "tokens": int(tgt_ids.numel()),
        "bytes": int(token_bytes.sum().item()),
        "tokens_per_byte": float(tgt_ids.numel() / max(int(token_bytes.sum().item()), 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Scylla data and byte accounting before paid training.")
    parser.add_argument("--train-gpt", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--tokenizer-meta-path", required=True)
    parser.add_argument("--vocab-size", type=int, default=998)
    parser.add_argument("--seq-len", type=int, default=2048)
    args = parser.parse_args()

    train_gpt = load_train_module(Path(args.train_gpt).resolve())
    data_path = Path(args.data_path).resolve()
    train_files = sorted(glob.glob(str(data_path / "fineweb_train_*.bin")))
    val_files = sorted(glob.glob(str(data_path / "fineweb_val_*.bin")))
    if not train_files:
        raise FileNotFoundError(f"No train shards found in {data_path}")
    if not val_files:
        raise FileNotFoundError(f"No validation shards found in {data_path}")

    (base, leading, boundary), metadata = train_gpt.load_tokenizer_luts(
        args.tokenizer_path,
        args.tokenizer_meta_path,
        args.vocab_size,
        torch.device("cpu"),
        validate_meta=False,
    )
    val_tokens = train_gpt.load_validation_tokens(str(data_path / "fineweb_val_*.bin"), args.seq_len)
    report = {
        "data_path": str(data_path),
        "train_shards": len(train_files),
        "val_shards": len(val_files),
        "vocab_size": args.vocab_size,
        "tokenizer_kind": metadata.get("tokenizer_kind", "unknown"),
        "tokenizer_meta_path": metadata.get("meta_path"),
        "seq_len": args.seq_len,
        "eval_byte_accounting": compute_eval_bytes(val_tokens, base, leading, boundary, args.seq_len),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
