#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-${RUN_MODE:-smoke}}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMISSION_DIR="${REPO_ROOT}/records/track_non_record_16mb/2026-04-24_Mamba3_MIMO_Scylla"
DATA_ROOT="${PGOLF_DATA_ROOT:-/workspace/pgolf_data}"
DATA_PATH="${PGOLF_SCYLLA_DATA_PATH:-${DATA_ROOT}/fineweb_scylla}"
HF_DATASET="${PGOLF_HF_DATASET:-LightSpeedUp/parameter-golf-data}"

detect_gpus() {
  python - <<'PY'
import torch
n = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(max(1, n))
PY
}

install_deps() {
  python -m pip install -q --upgrade pip
  python -m pip install -q packaging ninja wheel setuptools einops huggingface-hub sentencepiece tokenmonster datasets tqdm brotli numpy
  python - <<'PY'
import importlib.util, subprocess, sys
from pathlib import Path

def torch_needs_pin():
    try:
        import torch
        return not torch.__version__.startswith("2.9.1")
    except Exception:
        return True

if torch_needs_pin():
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "--force-reinstall", "torch==2.9.1",
        "--index-url", "https://download.pytorch.org/whl/cu128",
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

def try_install_mamba3_cache():
    import os
    import shutil
    import tarfile
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        print(f"mamba3_cache: huggingface_hub unavailable: {exc}", flush=True)
        return False
    repo_id = os.environ.get("PGOLF_MAMBA3_WHEEL_REPO", "Rtx09/8gpu")
    filename = os.environ.get("PGOLF_MAMBA3_WHEEL_FILE", "pgolf_mamba3_FIXED_wheels_py312_torch291_cu128.tar.gz")
    workdir = Path(os.environ.get("PGOLF_MAMBA3_WHEEL_DIR", "/workspace/pgolf_mamba3_wheels"))
    try:
        archive = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=str(workdir),
        )
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
        return has_mamba3()
    except Exception as exc:
        print(f"mamba3_cache: unavailable, falling back to source build: {exc}", flush=True)
        return False

if not has_mamba3():
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "mamba-ssm", "causal-conv1d"])
if not has_mamba3() and not try_install_mamba3_cache():
    env = dict(__import__("os").environ)
    env["MAMBA_FORCE_BUILD"] = "TRUE"
    env["CAUSAL_CONV1D_FORCE_BUILD"] = "TRUE"
    env.setdefault("MAX_JOBS", "8")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--no-cache-dir", "--no-build-isolation", "--no-deps", "--no-binary", ":all:", "--force-reinstall",
        "git+https://github.com/Dao-AILab/causal-conv1d.git",
    ], env=env)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--no-cache-dir", "--no-build-isolation", "--no-deps", "--no-binary", ":all:", "--force-reinstall",
        "git+https://github.com/state-spaces/mamba.git",
    ], env=env)
PY
}

download_data() {
  mkdir -p "${DATA_ROOT}"
  if compgen -G "${DATA_PATH}/fineweb_val_*.bin" > /dev/null && compgen -G "${DATA_PATH}/fineweb_train_*.bin" > /dev/null; then
    echo "Scylla data already present at ${DATA_PATH}"
    return
  fi
  HF_DATASET="${HF_DATASET}" DATA_ROOT="${DATA_ROOT}" python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["HF_DATASET"],
    repo_type="dataset",
    local_dir=os.environ["DATA_ROOT"],
    allow_patterns=["fineweb_scylla/*", "tokenizers/scylla/*"],
)
PY
}

run_audit() {
  python "${REPO_ROOT}/runpod/mamba3_scylla_byte_audit.py" \
    --train-gpt "${SUBMISSION_DIR}/train_gpt.py" \
    --data-path "${DATA_PATH}" \
    --tokenizer-meta-path "${SUBMISSION_DIR}/candidate.meta.npz" \
    --vocab-size 998 \
    --seq-len "${AUDIT_SEQ_LEN:-2048}"
}

run_train() {
  local mode="$1"
  local nproc iterations max_train val_every log_every run_sliding
  if [[ "${mode}" == "smoke" ]]; then
    nproc="${NPROC_PER_NODE:-1}"
    iterations="${ITERATIONS:-20}"
    max_train="${MAX_TRAINING_SECONDS:-120}"
    val_every="${VAL_LOSS_EVERY:-10}"
    log_every="${TRAIN_LOG_EVERY:-1}"
    run_sliding="${RUN_SLIDING_EVAL:-0}"
  else
    nproc="${NPROC_PER_NODE:-$(detect_gpus)}"
    iterations="${ITERATIONS:-20000}"
    max_train="${MAX_TRAINING_SECONDS:-999999}"
    val_every="${VAL_LOSS_EVERY:-1000}"
    log_every="${TRAIN_LOG_EVERY:-100}"
    run_sliding="${RUN_SLIDING_EVAL:-1}"
  fi

  cd "${SUBMISSION_DIR}"
  SEED="${SEED:-42}" \
  DATA_PATH="${DATA_PATH}" \
  TOKENIZER_PATH="${SUBMISSION_DIR}/candidate.vocab" \
  TOKENIZER_META_PATH="${SUBMISSION_DIR}/candidate.meta.npz" \
  VOCAB_SIZE=998 \
  MODEL_DIM="${MODEL_DIM:-512}" \
  NUM_LAYERS="${NUM_LAYERS:-9}" \
  MAMBA_D_STATE="${MAMBA_D_STATE:-128}" \
  MAMBA_HEADDIM="${MAMBA_HEADDIM:-64}" \
  MAMBA_EXPAND="${MAMBA_EXPAND:-2}" \
  MAMBA_IS_MIMO="${MAMBA_IS_MIMO:-1}" \
  MAMBA_MIMO_RANK="${MAMBA_MIMO_RANK:-4}" \
  MAMBA_CHUNK_SIZE="${MAMBA_CHUNK_SIZE:-16}" \
  BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-3072}" \
  BIGRAM_DIM="${BIGRAM_DIM:-112}" \
  WARMDOWN_ITERS="${WARMDOWN_ITERS:-4000}" \
  LZMA_PRESET="${LZMA_PRESET:-9}" \
  ITERATIONS="${iterations}" \
  MAX_TRAINING_SECONDS="${max_train}" \
  VAL_LOSS_EVERY="${val_every}" \
  TRAIN_LOG_EVERY="${log_every}" \
  RUN_SLIDING_EVAL="${run_sliding}" \
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
  audit)
    install_deps
    download_data
    run_audit
    ;;
  smoke)
    install_deps
    download_data
    run_audit
    python "${REPO_ROOT}/runpod/mamba3_forward_smoke.py"
    REPO_ROOT="${REPO_ROOT}" python "${REPO_ROOT}/runpod/mamba3_causality_smoke.py"
    run_train smoke
    ;;
  full)
    install_deps
    download_data
    run_audit
    python "${REPO_ROOT}/runpod/mamba3_forward_smoke.py"
    REPO_ROOT="${REPO_ROOT}" python "${REPO_ROOT}/runpod/mamba3_causality_smoke.py"
    run_train full
    ;;
  *)
    echo "Usage: bash runpod/mamba3_scylla_runpod.sh [install|data|audit|smoke|full]" >&2
    exit 2
    ;;
esac
