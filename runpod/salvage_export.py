from __future__ import annotations

import argparse
import importlib.util
import io
import lzma
import os
import subprocess
import sys
import time
import types
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = REPO_ROOT / "records" / "track_non_record_16mb" / "2026-04-04_11L_XSA11_EMA_GPTQ_1xH100_PCIe" / "train_gpt.py"
BUNDLES_ROOT = REPO_ROOT / "bundles"
LOGS_ROOT = REPO_ROOT / "logs"
DEFAULT_RUN_ID = "salvage_export_sp1024"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Salvage export from final_model_pre_quant.pt")
    parser.add_argument(
        "--checkpoint",
        default="/workspace/final_model_pre_quant.pt",
        help="Path to the pre-quant checkpoint saved from the interrupted 8x run.",
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help="Run id used for log and bundle names.",
    )
    parser.add_argument(
        "--gptq-calib-batches",
        type=int,
        default=64,
        help="Autoregressive self-generated calibration sequences for GPTQ.",
    )
    parser.add_argument(
        "--target-mb",
        type=float,
        default=15.9,
        help="Target compressed submission size in MiB.",
    )
    return parser.parse_args()


def log_factory(log_path: Path):
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    return log


def run_checked(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def ensure_sp1024_val_only() -> None:
    dataset_dir = REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024"
    tokenizer_path = REPO_ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model"
    has_val = dataset_dir.is_dir() and any(dataset_dir.glob("fineweb_val_*.bin"))
    if has_val and tokenizer_path.is_file():
        return
    jobs = str(max(8, min(32, os.cpu_count() or 16)))
    run_checked(
        [
            sys.executable,
            "data/cached_challenge_fineweb.py",
            "--variant",
            "sp1024",
            "--train-shards",
            "0",
            "--jobs",
            jobs,
        ]
    )


def install_flash_attn_shim() -> None:
    def flash_attn_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True) -> torch.Tensor:
        q_t = q.permute(0, 2, 1, 3).contiguous()
        k_t = k.permute(0, 2, 1, 3).contiguous()
        v_t = v.permute(0, 2, 1, 3).contiguous()
        try:
            y_t = F.scaled_dot_product_attention(
                q_t,
                k_t,
                v_t,
                is_causal=causal,
                enable_gqa=(q_t.size(1) != k_t.size(1)),
            )
        except TypeError:
            if q_t.size(1) != k_t.size(1):
                group = q_t.size(1) // k_t.size(1)
                k_t = k_t.repeat_interleave(group, dim=1)
                v_t = v_t.repeat_interleave(group, dim=1)
            y_t = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal)
        return y_t.permute(0, 2, 1, 3).contiguous()

    shim = types.ModuleType("flash_attn_interface")
    shim.flash_attn_func = flash_attn_func
    sys.modules["flash_attn_interface"] = shim


def configure_env(run_id: str, gptq_calib_batches: int, target_mb: float) -> None:
    overrides = {
        "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
        "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
        "VOCAB_SIZE": "1024",
        "RUN_ID": run_id,
        "EVAL_STRIDE": "0",
        "GPTQ_CALIB_BATCHES": str(gptq_calib_batches),
        "TARGET_MB": str(target_mb),
        "GATED_ATTENTION": "0",
        "VALUE_RESIDUAL": "0",
        "TRIGRAM": "0",
        "DTG_ENABLED": "0",
        "LAWA_ENABLED": "0",
    }
    os.environ.update(overrides)


def load_source_module():
    install_flash_attn_shim()
    spec = importlib.util.spec_from_file_location("salvage_source_train_gpt", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load source script: {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(mod, args, device: torch.device):
    mod.CastedLinear._qat_enabled = args.qat_enabled
    base_model = mod.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        mtp_num_heads=args.mtp_num_heads,
        mtp_loss_weight=args.mtp_loss_weight,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
        gated_attention=args.gated_attention,
        value_residual=args.value_residual,
        smear_enabled=args.smear_enabled,
    ).to(device).bfloat16()
    base_model.qo_bank.data = base_model.qo_bank.data.float()
    base_model.kv_bank.data = base_model.kv_bank.data.float()
    base_model.mlp_up_bank.data = base_model.mlp_up_bank.data.float()
    base_model.mlp_down_bank.data = base_model.mlp_down_bank.data.float()
    for module in base_model.modules():
        if isinstance(module, mod.CastedLinear):
            module.float()
    mod.restore_low_dim_params_to_fp32(base_model)
    return base_model


def create_bundle(run_id: str, log_path: Path) -> Path:
    BUNDLES_ROOT.mkdir(parents=True, exist_ok=True)
    bundle_path = BUNDLES_ROOT / f"{run_id}.zip"
    candidates = [
        REPO_ROOT / "final_model_pre_quant.pt",
        REPO_ROOT / "final_model.pt",
        REPO_ROOT / "final_model.int6.ptz",
        SOURCE_SCRIPT,
        log_path,
    ]
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for path in candidates:
            if path.is_file():
                zf.write(path, path.relative_to(REPO_ROOT))
    return bundle_path


def main() -> None:
    cli_args = parse_args()
    checkpoint_path = Path(cli_args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    ensure_sp1024_val_only()
    configure_env(cli_args.run_id, cli_args.gptq_calib_batches, cli_args.target_mb)

    log_path = LOGS_ROOT / f"{cli_args.run_id}.txt"
    if log_path.exists():
        log_path.unlink()
    log = log_factory(log_path)
    log(str(log_path.relative_to(REPO_ROOT)))

    mod = load_source_module()
    args = mod.Hyperparameters()
    code = SOURCE_SCRIPT.read_text(encoding="utf-8")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for salvage export")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    log(f"salvage_checkpoint:{checkpoint_path}")
    log(f"salvage_gpu:{torch.cuda.get_device_name(device)}")

    sp = mod.spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}")

    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = mod.load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = mod.build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    base_model = build_model(mod, args, device)
    export_sd = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = base_model.load_state_dict(export_sd, strict=False)
    if missing:
        log(f"salvage_missing_keys:{len(missing)}")
    if unexpected:
        log(f"salvage_unexpected_keys:{len(unexpected)}")

    n_params = sum(p.numel() for p in base_model.parameters())
    mtp_params = sum(p.numel() for p in base_model.mtp_heads.parameters())
    log(f"model_params:{n_params}")
    log(f"mtp_num_heads:{args.mtp_num_heads} mtp_loss_weight:{args.mtp_loss_weight} mtp_params:{mtp_params}")
    xsa_layers = [i for i, b in enumerate(base_model.blocks) if b.attn.use_xsa]
    log(f"XSA:last_{args.xsa_last_n} active_layers:{xsa_layers}")
    log("world_size:1 grad_accum_steps:8")
    log("sdp_backends:cudnn=False flash=False mem_efficient=True math=False")
    log(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{args.tied_embed_lr if args.tie_embeddings else args.embed_lr} "
        f"head_lr:{0.0 if args.tie_embeddings else args.head_lr} matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log(f"seed:{args.seed}")

    torch.cuda.synchronize()
    t_diag = time.perf_counter()
    diag_val_loss, diag_val_bpb = mod.eval_val(
        args,
        base_model,
        0,
        1,
        device,
        8,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        eval_seq_len=effective_eval_seq_len,
    )
    torch.cuda.synchronize()
    log(
        f"DIAGNOSTIC loaded_pre_quant val_loss:{diag_val_loss:.4f} val_bpb:{diag_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_diag):.0f}ms"
    )

    torch.save(export_sd, REPO_ROOT / "final_model.pt")
    torch.save(export_sd, REPO_ROOT / "final_model_pre_quant.pt")
    model_bytes = (REPO_ROOT / "final_model.pt").stat().st_size
    code_bytes = len(code.encode("utf-8"))
    log(f"Serialized model: {model_bytes} bytes")
    log("Saved pre-quant model checkpoint: final_model_pre_quant.pt")
    log(f"Code size: {code_bytes} bytes")

    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    unbanked_sd = mod._unbank_state_dict(sd_cpu, args.num_layers)
    log("gptq:building non-banked model for Hessian collection...")
    hessian_model = mod._HessianGPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
        smear_enabled=args.smear_enabled,
    ).to(device).bfloat16()
    for module in hessian_model.modules():
        if isinstance(module, mod.CastedLinear):
            module.float()
    mod.restore_low_dim_params_to_fp32(hessian_model)
    hessian_model.load_state_dict(
        {k: v.to(device) for k, v in unbanked_sd.items() if k in hessian_model.state_dict()},
        strict=False,
    )

    log(
        f"gptq:generating autoregressive calibration data ({args.gptq_calib_batches} seqs x {args.train_seq_len} tokens, temp=0.8)..."
    )
    t_gen = time.perf_counter()
    ar_tokens = mod.generate_autoregressive_calib(
        base_model,
        device,
        num_seqs=args.gptq_calib_batches,
        seq_len=args.train_seq_len,
        vocab_size=args.vocab_size,
        temperature=0.8,
        batch_size=min(8, args.gptq_calib_batches),
        seed=args.seed,
    )
    log(f"gptq:generated {len(ar_tokens)} sequences in {time.perf_counter() - t_gen:.1f}s")
    log("gptq:collecting hessians from autoregressive data...")
    hessians = mod.collect_hessians_from_tokens(hessian_model, ar_tokens, device)
    log(f"gptq:collected hessians for {len(hessians)} layers (AR self-gen)")
    del ar_tokens
    del hessian_model
    torch.cuda.empty_cache()

    quant_result, quant_meta = mod.mixed_quantize_int6(unbanked_sd, {"mlp", "attn"}, hessians=hessians)
    target_bytes = int(cli_args.target_mb * 1024 * 1024)
    ones_info: list[tuple[str, int, float]] = []
    for name, info in quant_meta.items():
        if not (isinstance(info, dict) and info.get("type") == "int6"):
            continue
        qk, sk = name + ".q", name + ".scale"
        if qk not in quant_result or sk not in quant_result:
            continue
        q, s = quant_result[qk], quant_result[sk]
        if s.ndim == 0:
            continue
        ones_mask = q.abs() == 1
        if not ones_mask.any():
            continue
        row_idx = torch.arange(q.shape[0]).unsqueeze(1).expand_as(q)[ones_mask]
        flat_idx = torch.arange(q.numel()).reshape(q.shape)[ones_mask]
        errors = s.float()[row_idx].pow(2)
        for flat_i, err in zip(flat_idx.tolist(), errors.tolist()):
            ones_info.append((qk, flat_i, err))

    def try_prune(num_items: int):
        tmp = {k: v.clone() for k, v in quant_result.items()}
        for i in range(min(num_items, len(ones_info))):
            tmp[ones_info[i][0]].view(-1)[ones_info[i][1]] = 0
        buf = io.BytesIO()
        torch.save({"w": tmp, "m": quant_meta}, buf)
        total = len(lzma.compress(buf.getvalue(), preset=9)) + code_bytes
        return total, tmp

    if ones_info:
        ones_info.sort(key=lambda x: x[2])
        no_sz, _ = try_prune(0)
        log(
            f"selective_prune: {len(ones_info)} ±1 candidates, unpruned={no_sz / (1024 * 1024):.2f}MB target={cli_args.target_mb}MB"
        )
        if no_sz <= target_bytes:
            log("selective_prune: already fits, no pruning needed")
        else:
            full_sz, _ = try_prune(len(ones_info))
            log(f"selective_prune: full ±1 prune={full_sz / (1024 * 1024):.2f}MB")
            if full_sz > target_bytes:
                log("selective_prune: even full prune not enough, applying all")
                _, quant_result = try_prune(len(ones_info))
            else:
                lo, hi = 0, len(ones_info)
                while lo < hi:
                    mid = (lo + hi) // 2
                    sz, _ = try_prune(mid)
                    if sz <= target_bytes:
                        hi = mid
                    else:
                        lo = mid + 1
                log(
                    f"selective_prune: pruning {lo}/{len(ones_info)} ±1 values ({100 * lo / len(ones_info):.1f}%) to fit {cli_args.target_mb}MB"
                )
                _, quant_result = try_prune(lo)

    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_blob = lzma.compress(quant_buf.getvalue(), preset=9)
    with (REPO_ROOT / "final_model.int6.ptz").open("wb") as f:
        f.write(quant_blob)
    log(f"Serialized model int6+lzma: {len(quant_blob)} bytes")
    log(f"Total submission size int6+lzma: {len(quant_blob) + code_bytes} bytes")

    quant_state = torch.load(io.BytesIO(lzma.decompress(quant_blob)), map_location="cpu")
    deq_unbanked = mod.dequantize_mixed_int6(quant_state["w"], quant_state["m"], unbanked_sd)
    deq_state = mod._rebank_state_dict(deq_unbanked, args.num_layers, sd_cpu)
    eval_model = build_model(mod, args, device)
    eval_model.load_state_dict(deq_state, strict=True)

    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = mod.eval_val(
        args,
        eval_model,
        0,
        1,
        device,
        8,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        eval_seq_len=effective_eval_seq_len,
    )
    torch.cuda.synchronize()
    log(
        f"final_int6_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log(f"final_int6_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    bundle_path = create_bundle(cli_args.run_id, log_path)
    log(f"bundle saved to {bundle_path}")


if __name__ == "__main__":
    main()
