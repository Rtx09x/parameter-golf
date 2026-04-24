#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-${RUN_MODE:-proof}}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMISSION_DIR="${REPO_ROOT}/records/track_non_record_16mb/2026-04-24_SP8192_Mamba3Adapter_TopStack"
DATA_DIR="${PGOLF_DATA_DIR:-${REPO_ROOT}/data}"

detect_gpus() {
  python - <<'PY'
import torch
print(max(1, torch.cuda.device_count() if torch.cuda.is_available() else 0))
PY
}

install_deps() {
  python -m pip install -q --upgrade pip
  python -m pip install -q packaging ninja wheel setuptools einops huggingface-hub sentencepiece brotli numpy tilelang
  python - <<'PY'
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

def check_torch():
    try:
        import torch
        return torch.__version__.startswith("2.9.1")
    except Exception:
        return False

if not check_torch():
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "--force-reinstall", "torch==2.9.1",
        "--index-url", "https://download.pytorch.org/whl/cu128",
    ])

def has_flash3():
    try:
        import flash_attn_interface
        return flash_attn_interface is not None
    except Exception:
        return False

if not has_flash3():
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "flash_attn_3", "--no-deps",
        "--find-links", "https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/",
    ])

def has_mamba3():
    try:
        from mamba_ssm import Mamba3
        return Mamba3 is not None
    except Exception:
        try:
            from mamba_ssm.modules.mamba3 import Mamba3
            return Mamba3 is not None
        except Exception:
            return False

def install_mamba3_cache():
    from huggingface_hub import hf_hub_download
    repo_id = os.environ.get("PGOLF_MAMBA3_WHEEL_REPO", "Rtx09/8gpu")
    filename = os.environ.get("PGOLF_MAMBA3_WHEEL_FILE", "pgolf_mamba3_FIXED_wheels_py312_torch291_cu128.tar.gz")
    workdir = Path(os.environ.get("PGOLF_MAMBA3_WHEEL_DIR", "/workspace/pgolf_mamba3_wheels"))
    archive = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename, local_dir=str(workdir))
    extract_dir = workdir / "extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(extract_dir)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "--force-reinstall", "--no-index", "--no-deps",
        f"--find-links={extract_dir / 'wheels'}",
        "causal-conv1d", "mamba-ssm",
    ])
    print(f"mamba3_cache: installed {filename} from {repo_id}", flush=True)

if not has_mamba3():
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "mamba-ssm", "causal-conv1d"])
    install_mamba3_cache()
if not has_mamba3():
    raise SystemExit("Mamba3 install failed. Check HF_TOKEN and the fixed wheel cache file.")
PY
}

download_data() {
  cd "${REPO_ROOT}"
  if compgen -G "${DATA_DIR}/datasets/fineweb10B_sp8192/fineweb_val_*.bin" > /dev/null && \
     compgen -G "${DATA_DIR}/datasets/fineweb10B_sp8192/fineweb_train_*.bin" > /dev/null && \
     [[ -f "${DATA_DIR}/tokenizers/fineweb_8192_bpe.model" ]]; then
    echo "SP8192 data already present at ${DATA_DIR}"
    return
  fi
  MATCHED_FINEWEB_REPO_ID="${MATCHED_FINEWEB_REPO_ID:-kevclark/parameter-golf}" \
    python data/cached_challenge_fineweb.py --variant sp8192 --train-shards "${SP8192_TRAIN_SHARDS:-80}"
}

run_train() {
  local mode="$1"
  local nproc iterations val_every log_every sliding ttt
  nproc="${NPROC_PER_NODE:-$(detect_gpus)}"
  case "${mode}" in
    smoke)
      iterations="${ITERATIONS:-20}"
      val_every="${VAL_LOSS_EVERY:-10}"
      log_every="${TRAIN_LOG_EVERY:-1}"
      sliding="${SLIDING_WINDOW_ENABLED:-0}"
      ttt="${TTT_ENABLED:-0}"
      ;;
    proof)
      iterations="${ITERATIONS:-2000}"
      val_every="${VAL_LOSS_EVERY:-1000}"
      log_every="${TRAIN_LOG_EVERY:-100}"
      sliding="${SLIDING_WINDOW_ENABLED:-0}"
      ttt="${TTT_ENABLED:-0}"
      ;;
    full)
      iterations="${ITERATIONS:-20000}"
      val_every="${VAL_LOSS_EVERY:-4000}"
      log_every="${TRAIN_LOG_EVERY:-100}"
      sliding="${SLIDING_WINDOW_ENABLED:-1}"
      ttt="${TTT_ENABLED:-1}"
      ;;
    *)
      echo "unknown mode: ${mode}" >&2
      exit 2
      ;;
  esac

  cd "${SUBMISSION_DIR}"
  DATA_DIR="${DATA_DIR}" \
  SEED="${SEED:-42}" \
  VOCAB_SIZE=8192 \
  NUM_LAYERS="${NUM_LAYERS:-11}" \
  MODEL_DIM="${MODEL_DIM:-512}" \
  MLP_MULT="${MLP_MULT:-3.75}" \
  QK_GAIN_INIT="${QK_GAIN_INIT:-5.25}" \
  MAMBA_ADAPTER_LAYERS="${MAMBA_ADAPTER_LAYERS:-6}" \
  MAMBA_ADAPTER_DIM="${MAMBA_ADAPTER_DIM:-64}" \
  MAMBA_ADAPTER_D_STATE="${MAMBA_ADAPTER_D_STATE:-64}" \
  MAMBA_ADAPTER_HEADDIM="${MAMBA_ADAPTER_HEADDIM:-32}" \
  MAMBA_ADAPTER_EXPAND="${MAMBA_ADAPTER_EXPAND:-2}" \
  MAMBA_ADAPTER_RANK="${MAMBA_ADAPTER_RANK:-2}" \
  MAMBA_ADAPTER_CHUNK_SIZE="${MAMBA_ADAPTER_CHUNK_SIZE:-16}" \
  COMPILE_ENABLED="${COMPILE_ENABLED:-0}" \
  ITERATIONS="${iterations}" \
  MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}" \
  VAL_LOSS_EVERY="${val_every}" \
  TRAIN_LOG_EVERY="${log_every}" \
  SLIDING_WINDOW_ENABLED="${sliding}" \
  TTT_ENABLED="${ttt}" \
  TTT_LR="${TTT_LR:-0.005}" \
  TTT_EPOCHS="${TTT_EPOCHS:-3}" \
  torchrun --standalone --nproc_per_node="${nproc}" train_gpt.py
}

case "${MODE}" in
  install)
    install_deps
    ;;
  data)
    install_deps
    download_data
    ;;
  smoke|proof|full)
    install_deps
    download_data
    run_train "${MODE}"
    ;;
  *)
    echo "Usage: bash runpod/sp8192_mamba_topstack_runpod.sh [install|data|smoke|proof|full]" >&2
    exit 2
    ;;
esac
