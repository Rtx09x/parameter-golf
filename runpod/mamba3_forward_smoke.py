from __future__ import annotations

import inspect
import os

import torch


def load_mamba3_class():
    try:
        from mamba_ssm import Mamba3
        return Mamba3
    except Exception:
        pass
    from mamba_ssm.modules.mamba3 import Mamba3
    return Mamba3


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    Mamba3 = load_mamba3_class()
    dim = int(os.environ.get("MODEL_DIM", 512))
    kwargs = {
        "d_model": dim,
        "d_state": int(os.environ.get("MAMBA_D_STATE", 128)),
        "headdim": int(os.environ.get("MAMBA_HEADDIM", 64)),
        "expand": int(os.environ.get("MAMBA_EXPAND", 2)),
        "is_mimo": bool(int(os.environ.get("MAMBA_IS_MIMO", "1"))),
        "mimo_rank": int(os.environ.get("MAMBA_MIMO_RANK", 4)),
        "chunk_size": int(os.environ.get("MAMBA_CHUNK_SIZE", 16)),
        "rope_fraction": float(os.environ.get("MAMBA_ROPE_FRACTION", 0.5)),
        "is_outproj_norm": False,
        "ngroups": int(os.environ.get("MAMBA_NGROUPS", 1)),
        "dtype": torch.bfloat16,
    }
    try:
        sig = inspect.signature(Mamba3)
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if not accepts_kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    except (TypeError, ValueError):
        pass
    model = Mamba3(**kwargs).to("cuda")
    x = torch.randn(2, 128, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = model(x)
    if isinstance(y, tuple):
        y = y[0]
    loss = y.float().square().mean()
    loss.backward()
    print({
        "mamba3_forward_backward": "ok",
        "shape": tuple(y.shape),
        "loss": float(loss.detach().cpu()),
        "params": sum(p.numel() for p in model.parameters()),
    })


if __name__ == "__main__":
    main()
