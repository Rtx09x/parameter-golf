from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch


def load_train_module(repo_root: Path):
    path = repo_root / "records/track_non_record_16mb/2026-04-24_DiffusionOnly_Scylla/train_gpt.py"
    spec = importlib.util.spec_from_file_location("diffusion_only_train_gpt", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load train script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    repo_root = Path(os.environ.get("REPO_ROOT", "/workspace/parameter-golf")).resolve()
    mod = load_train_module(repo_root)
    args = mod.Hyperparameters()
    args.vocab_size = int(os.environ.get("SMOKE_VOCAB_SIZE", 256))
    args.model_dim = int(os.environ.get("SMOKE_MODEL_DIM", 128))
    args.num_layers = int(os.environ.get("SMOKE_NUM_LAYERS", 2))
    args.num_heads = int(os.environ.get("SMOKE_NUM_HEADS", 4))
    args.bigram_vocab_size = int(os.environ.get("SMOKE_BIGRAM_VOCAB_SIZE", 512))
    args.bigram_dim = int(os.environ.get("SMOKE_BIGRAM_DIM", 32))
    args.max_seq_len = int(os.environ.get("SMOKE_SEQ_LEN", 64))
    args.diffusion_steps = int(os.environ.get("SMOKE_DIFFUSION_STEPS", 4))
    args.eval_noise_step = args.diffusion_steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1234)
    model = mod.CausalDiffusionLM(args).to(device)
    seq_len = args.max_seq_len
    split = seq_len // 2
    x = torch.randint(0, args.vocab_size, (2, seq_len), device=device)
    y = torch.randint(0, args.vocab_size, (2, seq_len), device=device)
    loss = model(x, y)
    loss.backward()

    model.eval()
    x1 = torch.randint(0, args.vocab_size, (2, seq_len), device=device)
    x2 = x1.clone()
    x2[:, split:] = torch.randint(0, args.vocab_size, x2[:, split:].shape, device=device)
    with torch.inference_mode():
        y1 = model.forward_logits(x1).float()
        y2 = model.forward_logits(x2).float()
    prefix_diff = (y1[:, :split] - y2[:, :split]).abs().max().item()
    suffix_diff = (y1[:, split:] - y2[:, split:]).abs().max().item()
    print({
        "diffusion_only_forward": "checked",
        "device": str(device),
        "loss": float(loss.detach().cpu()),
        "prefix_max_abs_diff": prefix_diff,
        "suffix_max_abs_diff": suffix_diff,
    })
    if prefix_diff > float(os.environ.get("CAUSALITY_TOL", "1e-4")):
        raise RuntimeError(f"Causality check failed: prefix logits changed by {prefix_diff}")


if __name__ == "__main__":
    main()
