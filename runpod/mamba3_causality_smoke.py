from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch


def load_train_module(repo_root: Path):
    path = repo_root / "records/track_non_record_16mb/2026-04-24_Mamba3_MIMO_Scylla/train_gpt.py"
    spec = importlib.util.spec_from_file_location("mamba3_train_gpt", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load train script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    repo_root = Path(os.environ.get("REPO_ROOT", "/workspace/parameter-golf")).resolve()
    mod = load_train_module(repo_root)
    args = mod.Hyperparameters()
    args.num_layers = int(os.environ.get("CAUSALITY_NUM_LAYERS", 1))
    args.model_dim = int(os.environ.get("MODEL_DIM", args.model_dim))
    args.vocab_size = int(os.environ.get("VOCAB_SIZE", args.vocab_size))
    args.bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", args.bigram_vocab_size))
    args.bigram_dim = int(os.environ.get("BIGRAM_DIM", args.bigram_dim))
    args.d_state = int(os.environ.get("MAMBA_D_STATE", args.d_state))
    args.headdim = int(os.environ.get("MAMBA_HEADDIM", args.headdim))
    args.is_mimo = bool(int(os.environ.get("MAMBA_IS_MIMO", str(int(args.is_mimo)))))
    args.mimo_rank = int(os.environ.get("MAMBA_MIMO_RANK", args.mimo_rank))
    args.chunk_size = int(os.environ.get("MAMBA_CHUNK_SIZE", args.chunk_size))

    torch.manual_seed(1234)
    model = mod.MambaGolfLM(args).to("cuda").eval()
    seq_len = int(os.environ.get("CAUSALITY_SEQ_LEN", 128))
    split = int(os.environ.get("CAUSALITY_SPLIT", 64))
    x1 = torch.randint(0, args.vocab_size, (2, seq_len), device="cuda")
    x2 = x1.clone()
    x2[:, split:] = torch.randint(0, args.vocab_size, x2[:, split:].shape, device="cuda")

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        y1 = model.forward_logits(x1).float()
        y2 = model.forward_logits(x2).float()
    max_prefix_diff = (y1[:, :split] - y2[:, :split]).abs().max().item()
    max_suffix_diff = (y1[:, split:] - y2[:, split:]).abs().max().item()
    print({
        "mamba3_causality": "checked",
        "prefix_max_abs_diff": max_prefix_diff,
        "suffix_max_abs_diff": max_suffix_diff,
        "split": split,
        "seq_len": seq_len,
    })
    if max_prefix_diff > float(os.environ.get("CAUSALITY_TOL", "1e-3")):
        raise RuntimeError(f"Causality check failed: prefix logits changed by {max_prefix_diff}")


if __name__ == "__main__":
    main()
