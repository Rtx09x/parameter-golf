from __future__ import annotations

import glob
import inspect
import io
import lzma
import math
import os
import random
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP


class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb_scylla")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./candidate.vocab")
    tokenizer_meta_path = os.environ.get("TOKENIZER_META_PATH", "./candidate.meta.npz")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 42))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 998))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    max_training_seconds = float(os.environ.get("MAX_TRAINING_SECONDS", os.environ.get("MAX_WALLCLOCK_SECONDS", 4800.0)))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 3))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 4000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 100))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 131_072))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    sliding_batch_seqs = int(os.environ.get("SLIDING_BATCH_SEQS", 16))
    run_sliding_eval = bool(int(os.environ.get("RUN_SLIDING_EVAL", "1")))

    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    d_state = int(os.environ.get("MAMBA_D_STATE", 128))
    headdim = int(os.environ.get("MAMBA_HEADDIM", 64))
    expand = int(os.environ.get("MAMBA_EXPAND", 2))
    is_mimo = bool(int(os.environ.get("MAMBA_IS_MIMO", "1")))
    mimo_rank = int(os.environ.get("MAMBA_MIMO_RANK", 4))
    chunk_size = int(os.environ.get("MAMBA_CHUNK_SIZE", 16))
    rope_fraction = float(os.environ.get("MAMBA_ROPE_FRACTION", 0.5))
    ngroups = int(os.environ.get("MAMBA_NGROUPS", 1))
    mamba_dtype = os.environ.get("MAMBA_DTYPE", "bfloat16")
    local_attn_layers = int(os.environ.get("LOCAL_ATTN_LAYERS", 2))
    local_attn_dim = int(os.environ.get("LOCAL_ATTN_DIM", 256))
    local_attn_window = int(os.environ.get("LOCAL_ATTN_WINDOW", 128))
    local_attn_heads = int(os.environ.get("LOCAL_ATTN_HEADS", 8))

    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 3072))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 112))

    lr = float(os.environ.get("LR", 0.0025))
    embed_lr = float(os.environ.get("EMBED_LR", 0.0015))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.08))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))
    warmdown_frac = float(os.environ.get("WARMDOWN_FRAC", 0.72))
    ema_enabled = bool(int(os.environ.get("EMA_ENABLED", "1")))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))

    int6_clip_range = int(os.environ.get("INT6_CLIP_RANGE", 31))
    quant_min_numel = int(os.environ.get("QUANT_MIN_NUMEL", 256))
    lzma_preset = int(os.environ.get("LZMA_PRESET", 9))
    temp_scaling = bool(int(os.environ.get("TEMP_SCALING", "1")))
    temp_grid = os.environ.get("TEMP_GRID", "0.85,0.90,0.95,1.00,1.05,1.10")
    temp_eval_during_train = bool(int(os.environ.get("TEMP_EVAL_DURING_TRAIN", "0")))


def log_temp_sweep(args, model, rank, world_size, device, val_tokens, base, leading, boundary, log, prefix: str) -> tuple[float, float, float]:
    best_temp = 1.0
    best_loss = float("inf")
    best_bpb = float("inf")
    for temp in parse_temp_grid(args.temp_grid):
        loss, bpb = eval_val(args, model, rank, world_size, device, val_tokens, base, leading, boundary, logit_temp=temp)
        log(f"{prefix}_temp_grid temp:{temp:.4f} val_loss:{loss:.4f} val_bpb:{bpb:.4f}")
        if bpb < best_bpb:
            best_temp = temp
            best_loss = loss
            best_bpb = bpb
    log(f"{prefix}_best_temp:{best_temp:.4f} val_loss:{best_loss:.4f} val_bpb:{best_bpb:.4f}")
    return best_temp, best_loss, best_bpb


TOKENIZER_META_FORMAT_VERSION = 1
DATAFILE_MAGIC = 20240520
DATAFILE_VERSION = 1
SHARD_HEADER_BYTES = 256 * np.dtype("<i4").itemsize
SHARD_NTOKENS_CACHE: dict[str, int] = {}
MMAP_CACHE: dict[str, np.memmap] = {}


def log_once(rank: int, msg: str, logfile: str | None = None, console: bool = True) -> None:
    if rank != 0:
        return
    if console:
        print(msg, flush=True)
    if logfile is not None:
        with open(logfile, "a", encoding="utf-8") as f:
            print(msg, file=f)


def load_mamba3_class():
    try:
        from mamba_ssm import Mamba3
        return Mamba3
    except Exception:
        pass
    try:
        from mamba_ssm.modules.mamba3 import Mamba3
        return Mamba3
    except Exception as exc:
        raise RuntimeError(
            "Mamba3 import failed. Install official state-spaces/mamba from source with "
            "MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall "
            "git+https://github.com/state-spaces/mamba.git --no-build-isolation"
        ) from exc


def dtype_from_name(name: str) -> torch.dtype:
    table = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name.lower() not in table:
        raise ValueError(f"Unsupported dtype {name!r}")
    return table[name.lower()]


def load_tokenizer_meta_luts_np(meta_path: Path, vocab_size: int):
    def scalar(value):
        arr = np.asarray(value)
        if arr.ndim == 0:
            return arr.item()
        item = arr.reshape(-1)[0]
        return item.item() if hasattr(item, "item") else item

    with np.load(meta_path, allow_pickle=False) as data:
        format_version = int(scalar(data["format_version"]))
        if format_version != TOKENIZER_META_FORMAT_VERSION:
            raise ValueError(f"Unsupported tokenizer meta format_version={format_version}")
        meta_vocab_size = int(scalar(data["vocab_size"]))
        tokenizer_kind = str(scalar(data["tokenizer_kind"]))
        source_model_name = str(scalar(data["source_model_name"]))
        base_bytes_np = np.asarray(data["base_bytes"], dtype=np.int16)
        has_leading_space_np = np.asarray(data["has_leading_space"], dtype=np.bool_)
        is_boundary_token_np = np.asarray(data["is_boundary_token"], dtype=np.bool_)
    table_size = max(meta_vocab_size, vocab_size)
    if base_bytes_np.shape[0] < table_size:
        padded_base_bytes = np.zeros((table_size,), dtype=np.int16)
        padded_has_leading_space = np.zeros((table_size,), dtype=np.bool_)
        padded_is_boundary = np.ones((table_size,), dtype=np.bool_)
        padded_base_bytes[: base_bytes_np.shape[0]] = base_bytes_np
        padded_has_leading_space[: has_leading_space_np.shape[0]] = has_leading_space_np
        padded_is_boundary[: is_boundary_token_np.shape[0]] = is_boundary_token_np
        base_bytes_np = padded_base_bytes
        has_leading_space_np = padded_has_leading_space
        is_boundary_token_np = padded_is_boundary
    metadata = {
        "format_version": format_version,
        "tokenizer_kind": tokenizer_kind,
        "source_model_name": source_model_name,
        "vocab_size": meta_vocab_size,
        "meta_path": str(meta_path),
    }
    return base_bytes_np, has_leading_space_np, is_boundary_token_np, metadata


def load_tokenizer_luts(tokenizer_meta_path: str, vocab_size: int, device: torch.device):
    meta_path = Path(tokenizer_meta_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"TOKENIZER_META_PATH does not exist: {meta_path}")
    base, leading, boundary, metadata = load_tokenizer_meta_luts_np(meta_path, vocab_size)
    return (
        torch.tensor(base, dtype=torch.int16, device=device),
        torch.tensor(leading, dtype=torch.bool, device=device),
        torch.tensor(boundary, dtype=torch.bool, device=device),
    ), metadata


def load_data_shard(file: Path) -> Tensor:
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != DATAFILE_MAGIC or int(header[1]) != DATAFILE_VERSION:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = SHARD_HEADER_BYTES + num_tokens * np.dtype("<u2").itemsize
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    toks = np.fromfile(file, dtype="<u2", count=num_tokens, offset=SHARD_HEADER_BYTES)
    if toks.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(toks.astype(np.uint16, copy=False))


def read_num_tokens(file: Path) -> int:
    key = str(file)
    cached = SHARD_NTOKENS_CACHE.get(key)
    if cached is not None:
        return cached
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != DATAFILE_MAGIC or int(header[1]) != DATAFILE_VERSION:
        raise ValueError(f"Unexpected shard header for {file}")
    n = int(header[2])
    SHARD_NTOKENS_CACHE[key] = n
    return n


def get_shard_memmap(file: Path) -> np.memmap:
    key = str(file)
    mm = MMAP_CACHE.get(key)
    if mm is not None:
        return mm
    n = read_num_tokens(file)
    mm = np.memmap(file, mode="r", dtype="<u2", offset=SHARD_HEADER_BYTES, shape=(n,))
    MMAP_CACHE[key] = mm
    return mm


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.num_tokens = np.array([read_num_tokens(f) for f in self.files], dtype=np.int64)
        seed = 0
        for f in self.files:
            for b in str(f).encode():
                seed = ((seed ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.cfg: tuple[int, int, int, int] | None = None
        self.eligible_shards: np.ndarray | None = None
        self.base_block_counts: np.ndarray | None = None
        n = len(self.files)
        self.cursor_phase = np.zeros(n, dtype=np.int64)
        self.cursor_block_count = np.zeros(n, dtype=np.int64)
        self.cursor_next = np.zeros(n, dtype=np.int64)
        self.cursor_start = np.zeros(n, dtype=np.int64)
        self.cursor_stride = np.ones(n, dtype=np.int64)
        self.cursor_init = np.zeros(n, dtype=np.bool_)
        self.batches_built = 0

    def pick_coprime_stride(self, n: int) -> int:
        if n <= 1:
            return 1
        while True:
            s = int(self.rng.integers(1, n))
            if math.gcd(s, n) == 1:
                return s

    def reset_cursor(self, si: int, seq_len: int) -> None:
        nt = int(self.num_tokens[si])
        max_phase = min(seq_len - 1, max(0, nt - seq_len - 1))
        phase = int(self.rng.integers(max_phase + 1)) if max_phase > 0 else 0
        bc = max(1, (nt - 1 - phase) // seq_len)
        self.cursor_phase[si] = phase
        self.cursor_block_count[si] = bc
        self.cursor_next[si] = 0
        self.cursor_start[si] = int(self.rng.integers(bc)) if bc > 1 else 0
        self.cursor_stride[si] = self.pick_coprime_stride(bc)
        self.cursor_init[si] = True

    def ensure_cursor(self, si: int, seq_len: int) -> None:
        if not self.cursor_init[si] or self.cursor_next[si] >= self.cursor_block_count[si]:
            self.reset_cursor(si, seq_len)

    def take_from_shard(self, si: int, seq_len: int, count: int, out: list[tuple[int, int]]) -> None:
        rem = count
        while rem > 0:
            self.ensure_cursor(si, seq_len)
            bc = int(self.cursor_block_count[si])
            ni = int(self.cursor_next[si])
            take = min(rem, bc - ni)
            phase = int(self.cursor_phase[si])
            start = int(self.cursor_start[si])
            stride = int(self.cursor_stride[si])
            for j in range(take):
                bi = (start + (ni + j) * stride) % bc
                out.append((si, phase + bi * seq_len))
            self.cursor_next[si] = ni + take
            rem -= take

    def init_pipeline(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> None:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        num_seqs = local_tokens // seq_len
        if num_seqs <= 0:
            raise ValueError("train_batch_tokens too small for seq_len/world_size/grad_accum")
        global_num_seqs = num_seqs * self.world_size
        self.cfg = (local_tokens, seq_len, num_seqs, global_num_seqs)
        block_counts = (self.num_tokens - 1) // seq_len
        self.eligible_shards = np.nonzero(block_counts > 0)[0].astype(np.int64)
        self.base_block_counts = block_counts[self.eligible_shards].astype(np.int64)

    def sample_global_windows(self) -> list[tuple[int, int]]:
        assert self.cfg is not None and self.eligible_shards is not None and self.base_block_counts is not None
        _, seq_len, _, global_num_seqs = self.cfg
        ec = int(self.eligible_shards.size)
        progress = min(self.batches_built / 1800.0, 1.0)
        remaining = np.empty(ec, dtype=np.float64)
        for i, si in enumerate(self.eligible_shards.tolist()):
            if self.cursor_init[si]:
                rem = int(self.cursor_block_count[si]) - int(self.cursor_next[si])
                remaining[i] = float(max(rem, 1))
            else:
                remaining[i] = float(self.base_block_counts[i])
        alpha = 0.90 - 0.40 * progress
        weights = np.power(remaining, alpha)
        weights = weights / max(float(weights.sum()), 1.0)
        low = min(max(8, self.world_size), ec, global_num_seqs)
        high = min(max(32, self.world_size * 8), ec, global_num_seqs)
        mix = max(1, min(int(round(low + progress * (high - low))), ec, global_num_seqs))
        choices = self.rng.choice(ec, size=mix, replace=False, p=weights)
        shards = self.eligible_shards[choices]
        probs = weights[choices].copy()
        probs /= probs.sum()
        counts = np.ones(mix, dtype=np.int64)
        extra = global_num_seqs - mix
        if extra > 0:
            counts += self.rng.multinomial(extra, probs).astype(np.int64)
        buckets: list[list[tuple[int, int]]] = []
        for si, cnt in zip(shards.tolist(), counts.tolist()):
            bucket: list[tuple[int, int]] = []
            self.take_from_shard(int(si), seq_len, int(cnt), bucket)
            if bucket:
                if len(bucket) > 1:
                    perm = self.rng.permutation(len(bucket))
                    bucket = [bucket[int(k)] for k in perm.tolist()]
                buckets.append(bucket)
        windows: list[tuple[int, int]] = []
        active = [i for i, b in enumerate(buckets) if b]
        while active:
            order = self.rng.permutation(len(active))
            new_active: list[int] = []
            for oi in order.tolist():
                bi = active[oi]
                if buckets[bi]:
                    windows.append(buckets[bi].pop())
                if buckets[bi]:
                    new_active.append(bi)
            active = new_active
        return windows

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        if self.cfg is None:
            self.init_pipeline(global_tokens, seq_len, grad_accum_steps)
        _, _, num_seqs, _ = self.cfg
        windows = self.sample_global_windows()[self.rank::self.world_size]
        x = torch.empty((num_seqs, seq_len), dtype=torch.int64)
        y = torch.empty((num_seqs, seq_len), dtype=torch.int64)
        for slot, (si, pos) in enumerate(windows):
            mm = get_shard_memmap(self.files[si])
            window = torch.as_tensor(np.array(mm[pos:pos + seq_len + 1], dtype=np.int64))
            x[slot] = window[:-1]
            y[slot] = window[1:]
        self.batches_built += 1
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


class RMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev


class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = nn.Linear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()
    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


class Mamba3ResidualBlock(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        Mamba3 = load_mamba3_class()
        kwargs = {
            "d_model": args.model_dim,
            "d_state": args.d_state,
            "headdim": args.headdim,
            "expand": args.expand,
            "is_mimo": args.is_mimo,
            "mimo_rank": args.mimo_rank,
            "chunk_size": args.chunk_size,
            "rope_fraction": args.rope_fraction,
            "is_outproj_norm": False,
            "ngroups": args.ngroups,
            "dtype": dtype_from_name(args.mamba_dtype),
        }
        try:
            sig = inspect.signature(Mamba3)
            accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if not accepts_kwargs:
                kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        except (TypeError, ValueError):
            pass
        self.norm = RMSNorm()
        self.mixer = Mamba3(**kwargs)
        self.resid_scale = nn.Parameter(torch.ones(args.model_dim, dtype=torch.float32))
    def forward(self, x: Tensor) -> Tensor:
        y = self.mixer(self.norm(x))
        if isinstance(y, tuple):
            y = y[0]
        return x + self.resid_scale.to(dtype=x.dtype)[None, None, :] * y


class LocalAttentionAdapter(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        inner_dim = args.local_attn_dim if args.local_attn_dim > 0 else args.model_dim
        if inner_dim % args.local_attn_heads != 0:
            raise ValueError("LOCAL_ATTN_DIM must be divisible by LOCAL_ATTN_HEADS")
        self.num_heads = args.local_attn_heads
        self.inner_dim = inner_dim
        self.head_dim = inner_dim // args.local_attn_heads
        self.window = args.local_attn_window
        self.norm = RMSNorm()
        self.qkv = nn.Linear(args.model_dim, 3 * inner_dim, bias=False)
        self.proj = nn.Linear(inner_dim, args.model_dim, bias=False)
        self.gate = nn.Parameter(torch.tensor(-4.0, dtype=torch.float32))
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: Tensor) -> Tensor:
        b, t, c = x.shape
        qkv = self.qkv(self.norm(x)).view(b, t, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_mask = None
        if self.window > 0 and self.window < t:
            row = torch.arange(t, device=x.device)[:, None]
            col = torch.arange(t, device=x.device)[None, :]
            attn_mask = col < (row - self.window + 1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, self.inner_dim)
        y = self.proj(y)
        return x + torch.sigmoid(self.gate).to(dtype=x.dtype) * y


class MambaGolfLM(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=args.tied_embed_init_std)
        self.bigram = (
            BigramHashEmbedding(args.bigram_vocab_size, args.bigram_dim, args.model_dim)
            if args.bigram_vocab_size > 0 else None
        )
        self.smear = SmearGate(args.model_dim)
        self.blocks = nn.ModuleList([Mamba3ResidualBlock(args) for _ in range(args.num_layers)])
        self.local_adapters = nn.ModuleList([LocalAttentionAdapter(args) for _ in range(max(args.local_attn_layers, 0))])
        self.final_norm = RMSNorm()
        self.lm_head = None if args.tie_embeddings else nn.Linear(args.model_dim, args.vocab_size, bias=False)
        self.logit_softcap = args.logit_softcap
    def hidden(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        for block in self.blocks:
            x = block(x)
        for adapter in self.local_adapters:
            x = adapter(x)
        return self.final_norm(x)
    def logits_from_hidden(self, x: Tensor) -> Tensor:
        if self.lm_head is None:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)
    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.hidden(input_ids)
        logits = self.logits_from_hidden(x).reshape(-1, self.args.vocab_size)
        return F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="mean")
    def forward_logits(self, input_ids: Tensor) -> Tensor:
        return self.logits_from_hidden(self.hidden(input_ids))


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for seq_len={seq_len}")
    return tokens[: usable + 1]


def byte_count_for_targets(x: Tensor, y: Tensor, base: Tensor, leading: Tensor, boundary: Tensor) -> Tensor:
    prev_ids = x.reshape(-1)
    tgt_ids = y.reshape(-1)
    token_bytes = base[tgt_ids].to(dtype=torch.int16)
    token_bytes += (leading[tgt_ids] & ~boundary[prev_ids]).to(dtype=torch.int16)
    return token_bytes.to(torch.float64).sum()


def parse_temp_grid(grid: str) -> list[float]:
    temps: list[float] = []
    for item in grid.split(","):
        item = item.strip()
        if not item:
            continue
        temp = float(item)
        if temp <= 0:
            raise ValueError(f"Temperature must be positive, got {temp}")
        temps.append(temp)
    return temps or [1.0]


def forward_logits_any(model: nn.Module, input_ids: Tensor) -> Tensor:
    target = model.module if isinstance(model, DDP) else model
    return target.forward_logits(input_ids)


def eval_val(args, model, rank, world_size, device, val_tokens, base, leading, boundary, logit_temp: float = 1.0) -> tuple[float, float]:
    seq_len = args.eval_seq_len
    local_batch_tokens = args.val_batch_size // max(world_size, 1)
    local_batch_seqs = max(1, local_batch_tokens // seq_len)
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                if abs(logit_temp - 1.0) < 1e-8:
                    batch_loss = model(x, y).detach()
                else:
                    logits = forward_logits_any(model, x).float() / logit_temp
                    batch_loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), y.reshape(-1), reduction="mean").detach()
            n = float(y.numel())
            loss_sum += batch_loss.to(torch.float64) * n
            token_count += n
            byte_count += byte_count_for_targets(x, y, base, leading, boundary)
    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, token_count, byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    val_loss = loss_sum / token_count
    val_bpb = val_loss.item() / math.log(2.0) * (token_count.item() / byte_count.item())
    if was_training:
        model.train()
    return float(val_loss.item()), float(val_bpb)


def eval_val_sliding(args, model, rank, world_size, device, val_tokens, base, leading, boundary, logit_temp: float = 1.0) -> tuple[float, float]:
    seq_len = args.eval_seq_len
    stride = args.eval_stride
    starts = list(range(0, val_tokens.numel() - seq_len, stride))
    local_starts = starts[rank::world_size]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(local_starts), args.sliding_batch_seqs):
            chunk = local_starts[i:i + args.sliding_batch_seqs]
            bsz = len(chunk)
            x = torch.empty((bsz, seq_len), dtype=torch.int64, device=device)
            y = torch.empty((bsz, seq_len), dtype=torch.int64, device=device)
            first_score = torch.empty((bsz,), dtype=torch.int64, device=device)
            for j, start in enumerate(chunk):
                end = start + seq_len + 1
                local = val_tokens[start:end].to(device=device, dtype=torch.int64, non_blocking=True)
                x[j] = local[:-1]
                y[j] = local[1:]
                first_score[j] = 0 if start == 0 else seq_len - stride
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = forward_logits_any(model, x).float() / logit_temp
            losses = F.cross_entropy(logits.reshape(-1, args.vocab_size), y.reshape(-1), reduction="none").view(bsz, seq_len)
            for j in range(bsz):
                s = int(first_score[j].item())
                if s >= seq_len:
                    continue
                yy = y[j:j + 1, s:]
                xx = x[j:j + 1, s:]
                loss_sum += losses[j, s:].to(torch.float64).sum()
                token_count += float(seq_len - s)
                byte_count += byte_count_for_targets(xx, yy, base, leading, boundary)
    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, token_count, byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    val_loss = loss_sum / token_count
    val_bpb = val_loss.item() / math.log(2.0) * (token_count.item() / byte_count.item())
    if was_training:
        model.train()
    return float(val_loss.item()), float(val_bpb)


def pack_int6_tensor(q: Tensor) -> tuple[Tensor, int]:
    flat = q.detach().cpu().to(torch.uint8).flatten().numpy()
    orig_n = int(flat.size)
    pad = (-orig_n) % 4
    if pad:
        flat = np.pad(flat, (0, pad), constant_values=32)
    vals = flat.astype(np.uint32).reshape(-1, 4)
    packed24 = vals[:, 0] | (vals[:, 1] << 6) | (vals[:, 2] << 12) | (vals[:, 3] << 18)
    out = np.empty((packed24.shape[0], 3), dtype=np.uint8)
    out[:, 0] = packed24 & 0xFF
    out[:, 1] = (packed24 >> 8) & 0xFF
    out[:, 2] = (packed24 >> 16) & 0xFF
    return torch.from_numpy(out.reshape(-1).copy()), orig_n


def unpack_int6_tensor(packed: Tensor, orig_n: int, shape: tuple[int, ...]) -> Tensor:
    data = packed.detach().cpu().to(torch.uint8).numpy().reshape(-1, 3).astype(np.uint32)
    vals = data[:, 0] | (data[:, 1] << 8) | (data[:, 2] << 16)
    out = np.empty((vals.shape[0], 4), dtype=np.uint8)
    out[:, 0] = vals & 0x3F
    out[:, 1] = (vals >> 6) & 0x3F
    out[:, 2] = (vals >> 12) & 0x3F
    out[:, 3] = (vals >> 18) & 0x3F
    return torch.from_numpy(out.reshape(-1)[:orig_n].astype(np.int16).copy()).view(*shape) - 32


def quantize_state_dict_int6(sd: dict[str, Tensor], clip_range: int, min_numel: int):
    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    for name, tensor in sd.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() < min_numel:
            kept = t.to(torch.float16) if t.is_floating_point() else t
            result[name] = kept.contiguous()
            meta[name] = {"type": "passthrough", "dtype": str(tensor.dtype).replace("torch.", "")}
            continue
        t32 = t.float()
        if t32.ndim >= 2:
            rows = t32.reshape(t32.shape[0], -1)
            amax = rows.abs().amax(dim=1).clamp_min(1e-8)
            scale = (amax / float(clip_range)).to(torch.float16)
            q = torch.clamp(torch.round(rows / scale.float()[:, None]), -clip_range, clip_range).to(torch.int16) + 32
            packed, orig_n = pack_int6_tensor(q)
            result[name + ".packed"] = packed
            result[name + ".scale"] = scale.contiguous()
            meta[name] = {"type": "int6_row", "shape": tuple(t.shape), "orig_n": orig_n, "dtype": str(tensor.dtype).replace("torch.", "")}
        else:
            amax = t32.abs().amax().clamp_min(1e-8)
            scale = (amax / float(clip_range)).to(torch.float16)
            q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int16) + 32
            packed, orig_n = pack_int6_tensor(q)
            result[name + ".packed"] = packed
            result[name + ".scale"] = scale.contiguous()
            meta[name] = {"type": "int6_tensor", "shape": tuple(t.shape), "orig_n": orig_n, "dtype": str(tensor.dtype).replace("torch.", "")}
    return result, meta


def dequantize_state_dict_int6(result: dict[str, Tensor], meta: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, info in meta.items():
        kind = info["type"]
        dtype = getattr(torch, info.get("dtype", "float32"))
        if kind == "passthrough":
            t = result[name]
            out[name] = t.to(dtype) if t.is_floating_point() else t
            continue
        shape = tuple(info["shape"])
        q = unpack_int6_tensor(result[name + ".packed"], int(info["orig_n"]), shape)
        scale = result[name + ".scale"].float()
        if kind == "int6_row":
            qf = q.float().reshape(shape[0], -1) * scale[:, None]
            out[name] = qf.reshape(shape).to(dtype)
        elif kind == "int6_tensor":
            out[name] = (q.float() * scale).reshape(shape).to(dtype)
        else:
            raise ValueError(f"Unknown quant type {kind}")
    return out


def build_optimizer(args: Hyperparameters, model: nn.Module):
    embed_params = [model.tok_emb.weight]
    if model.bigram is not None:
        embed_params.append(model.bigram.embed.weight)
    embed_ids = {id(p) for p in embed_params}
    other_params = [p for p in model.parameters() if id(p) not in embed_ids]
    return torch.optim.AdamW(
        [
            {"params": embed_params, "lr": args.embed_lr, "base_lr": args.embed_lr},
            {"params": other_params, "lr": args.lr, "base_lr": args.lr},
        ],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        fused=torch.cuda.is_available(),
    )


def lr_multiplier(args: Hyperparameters, step: int, elapsed_s: float) -> float:
    if step <= args.warmup_steps:
        return max(step / max(args.warmup_steps, 1), 1e-4)
    warmdown_steps = args.warmdown_iters
    if warmdown_steps <= 0:
        warmdown_steps = max(int(args.iterations * args.warmdown_frac), 1)
    warmdown_steps = min(max(warmdown_steps, 1), max(args.iterations, 1))
    start = max(args.iterations - warmdown_steps, 0)
    return max((args.iterations - step) / warmdown_steps, 0.0) if step >= start else 1.0


def clone_ema_state(model: nn.Module) -> dict[str, Tensor]:
    return {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items() if v.is_floating_point()}


def update_ema_state(ema: dict[str, Tensor], model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        state = model.state_dict()
        for k, avg in ema.items():
            avg.mul_(decay).add_(state[k].detach().float().cpu(), alpha=1.0 - decay)


def apply_ema_state(model: nn.Module, ema: dict[str, Tensor]) -> None:
    sd = model.state_dict()
    for k, avg in ema.items():
        sd[k] = avg.to(dtype=sd[k].dtype, device=sd[k].device)
    model.load_state_dict(sd, strict=True)


def main() -> None:
    args = Hyperparameters()
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mamba-3 training")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    logfile = None
    if rank == 0:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile, flush=True)
    log = lambda msg, console=True: log_once(rank, msg, logfile, console)

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    (base_bytes_lut, leading_lut, boundary_lut), tok_meta = load_tokenizer_luts(args.tokenizer_meta_path, args.vocab_size, device)
    dataset_dir = Path(args.data_path).resolve()
    train_shards = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.eval_seq_len)
    log(f"tokenizer: kind={tok_meta.get('tokenizer_kind', 'unknown')} vocab={tok_meta.get('vocab_size', '?')}")
    log(f"train_loader:dataset:{dataset_dir.name} train_shards:{train_shards}")
    log(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    base_model = MambaGolfLM(args).to(device=device)
    optimizer = build_optimizer(args, base_model)
    model = DDP(base_model, device_ids=[local_rank]) if distributed else base_model
    grad_accum_steps = 8 // world_size if 8 % world_size == 0 else 1
    grad_scale = 1.0 / grad_accum_steps
    loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    ema_state = clone_ema_state(base_model) if args.ema_enabled and rank == 0 else None

    log(f"model:mamba3_mimo layers:{args.num_layers} dim:{args.model_dim} d_state:{args.d_state} headdim:{args.headdim} expand:{args.expand} mimo:{int(args.is_mimo)} rank:{args.mimo_rank} chunk:{args.chunk_size}")
    log(f"local_attention:layers:{args.local_attn_layers} dim:{args.local_attn_dim} heads:{args.local_attn_heads} window:{args.local_attn_window}")
    log(f"model_params:{sum(p.numel() for p in base_model.parameters())}")
    log(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log(f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} iterations:{args.iterations} max_training_seconds:{args.max_training_seconds:.3f} warmdown_iters:{args.warmdown_iters}")
    log(f"optimizer:AdamW lr:{args.lr} embed_lr:{args.embed_lr} wd:{args.weight_decay} ema:{int(args.ema_enabled)}")
    log(f"export:int6_clip:{args.int6_clip_range} lzma_preset:{args.lzma_preset} temp_scaling:{int(args.temp_scaling)} temp_eval_during_train:{int(args.temp_eval_during_train)} temp_grid:{args.temp_grid}")
    log(f"seed:{args.seed}")

    if args.warmup_steps > 0:
        for warmup_step in range(args.warmup_steps):
            x, y = loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                warmup_loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            warmup_loss.backward()
            optimizer.zero_grad(set_to_none=True)
            log(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    start_time = time.perf_counter()
    last_step = 0
    for step in range(1, args.iterations + 1):
        elapsed_s = time.perf_counter() - start_time
        if args.max_training_seconds > 0 and elapsed_s >= args.max_training_seconds:
            log(f"training_time_budget_reached before_step:{step} elapsed_s:{elapsed_s:.1f}")
            break
        lr_mul = lr_multiplier(args, step, elapsed_s)
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * lr_mul
        optimizer.zero_grad(set_to_none=True)
        train_loss = torch.zeros((), device=device)
        for _ in range(grad_accum_steps):
            x, y = loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            (loss * grad_scale).backward()
            train_loss += loss.detach()
        torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        optimizer.step()
        if args.ema_enabled:
            if distributed:
                dist.barrier()
            if rank == 0:
                update_ema_state(ema_state, base_model, args.ema_decay)
        train_loss /= grad_accum_steps
        last_step = step
        elapsed_s = time.perf_counter() - start_time
        if step == 1 or step % args.train_log_every == 0:
            log(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} elapsed_s:{elapsed_s:.1f} step_avg_ms:{1000.0 * elapsed_s / step:.2f} lr_mul:{lr_mul:.4f}")
        if step == 1 or step % args.val_loss_every == 0:
            val_loss, val_bpb = eval_val(args, model, rank, world_size, device, val_tokens, base_bytes_lut, leading_lut, boundary_lut)
            log(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} elapsed_s:{elapsed_s:.1f}")
            if args.temp_eval_during_train:
                log_temp_sweep(args, model, rank, world_size, device, val_tokens, base_bytes_lut, leading_lut, boundary_lut, log, f"step:{step}/{args.iterations}")

    torch.cuda.synchronize()
    log(f"training_done steps_completed:{last_step} train_elapsed_s:{time.perf_counter() - start_time:.1f}")
    mem_alloc = torch.cuda.max_memory_allocated() // (1024 * 1024)
    mem_reserved = torch.cuda.max_memory_reserved() // (1024 * 1024)
    log(f"peak memory allocated:{mem_alloc} MiB reserved:{mem_reserved} MiB")

    raw_state_cpu = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
    raw_loss, raw_bpb = eval_val(args, model, rank, world_size, device, val_tokens, base_bytes_lut, leading_lut, boundary_lut)
    log(f"DIAGNOSTIC raw val_loss:{raw_loss:.4f} val_bpb:{raw_bpb:.4f}")
    export_choice = "raw"
    export_loss, export_bpb = raw_loss, raw_bpb

    if args.ema_enabled:
        if distributed:
            dist.barrier()
        if rank == 0:
            log("ema:evaluating EMA weights")
            apply_ema_state(base_model, ema_state)
        if distributed:
            obj = [base_model.state_dict() if rank == 0 else None]
            dist.broadcast_object_list(obj, src=0)
            if rank != 0:
                base_model.load_state_dict(obj[0], strict=True)
        ema_loss, ema_bpb = eval_val(args, model, rank, world_size, device, val_tokens, base_bytes_lut, leading_lut, boundary_lut)
        log(f"DIAGNOSTIC ema val_loss:{ema_loss:.4f} val_bpb:{ema_bpb:.4f}")
        if ema_bpb < raw_bpb:
            export_choice = "ema"
            export_loss, export_bpb = ema_loss, ema_bpb
        else:
            base_model.load_state_dict(raw_state_cpu, strict=True)
            if distributed:
                dist.barrier()

    log(f"export_choice:{export_choice} val_loss:{export_loss:.4f} val_bpb:{export_bpb:.4f}")

    if rank == 0:
        export_sd = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
        torch.save(export_sd, "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(Path(__file__).read_text(encoding="utf-8").encode("utf-8"))
        log(f"Serialized model: {model_bytes} bytes")
        log(f"Code size: {code_bytes} bytes")
        q_state, q_meta = quantize_state_dict_int6(export_sd, args.int6_clip_range, args.quant_min_numel)
        buf = io.BytesIO()
        torch.save({"w": q_state, "m": q_meta}, buf)
        quant_blob = lzma.compress(buf.getvalue(), preset=args.lzma_preset)
        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob)
        log(f"Serialized model int6pack+lzma: {len(quant_blob)} bytes")
        log(f"Total submission size int6pack+lzma: {len(quant_blob) + code_bytes} bytes")
    if distributed:
        dist.barrier()

    with open("final_model.int6.ptz", "rb") as f:
        payload = torch.load(io.BytesIO(lzma.decompress(f.read())), map_location="cpu")
    deq_sd = dequantize_state_dict_int6(payload["w"], payload["m"])
    eval_model = MambaGolfLM(args).to(device=device)
    eval_model.load_state_dict(deq_sd, strict=True)
    eval_model.eval()
    temp_candidates = parse_temp_grid(args.temp_grid) if args.temp_scaling else [1.0]
    best_temp = 1.0
    best_loss = float("inf")
    best_bpb = float("inf")
    for temp in temp_candidates:
        q_loss, q_bpb = eval_val(args, eval_model, rank, world_size, device, val_tokens, base_bytes_lut, leading_lut, boundary_lut, logit_temp=temp)
        log(f"temp_grid temp:{temp:.4f} val_loss:{q_loss:.4f} val_bpb:{q_bpb:.4f}")
        if q_bpb < best_bpb:
            best_temp = temp
            best_loss = q_loss
            best_bpb = q_bpb
    q_loss, q_bpb = best_loss, best_bpb
    log(f"final_temperature:{best_temp:.4f}")
    log(f"final_int6_roundtrip val_loss:{q_loss:.4f} val_bpb:{q_bpb:.4f}")
    log(f"final_int6_roundtrip_exact val_loss:{q_loss:.8f} val_bpb:{q_bpb:.8f}")
    if args.run_sliding_eval:
        sw_loss, sw_bpb = eval_val_sliding(args, eval_model, rank, world_size, device, val_tokens, base_bytes_lut, leading_lut, boundary_lut, logit_temp=best_temp)
        log(f"final_int6_sliding_window_s{args.eval_stride} val_loss:{sw_loss:.4f} val_bpb:{sw_bpb:.4f}")
        log(f"final_int6_sliding_window_s{args.eval_stride}_exact val_loss:{sw_loss:.8f} val_bpb:{sw_bpb:.8f}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
