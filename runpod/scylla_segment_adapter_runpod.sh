#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-${RUN_MODE:-smoke}}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMISSION_DIR="${REPO_ROOT}/records/track_10min_16mb/2026-04-24_Scylla_SegmentStateAdapter_nonrecord"
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
  python -m pip install -q huggingface-hub sentencepiece tokenmonster datasets tqdm brotli numpy
  python - <<'PY'
import importlib.util, subprocess, sys

if importlib.util.find_spec("torch") is None:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "torch==2.9.1", "--index-url", "https://download.pytorch.org/whl/cu128",
    ])

if importlib.util.find_spec("flash_attn_interface") is None:
    attempts = [
        [
            sys.executable, "-m", "pip", "install", "-q",
            "flash_attn_3", "--no-deps", "--find-links",
            "https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/",
        ],
        [
            sys.executable, "-m", "pip", "install", "-q",
            "https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl",
        ],
    ]
    last = None
    for cmd in attempts:
        try:
            subprocess.check_call(cmd)
            break
        except subprocess.CalledProcessError as exc:
            last = exc
    else:
        raise last
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
    local_dir_use_symlinks=False,
)
PY
}

run_audit() {
  python "${REPO_ROOT}/runpod/scylla_segment_byte_audit.py" \
    --train-gpt "${SUBMISSION_DIR}/train_gpt.py" \
    --data-path "${DATA_PATH}" \
    --tokenizer-path "${SUBMISSION_DIR}/candidate.vocab" \
    --tokenizer-meta-path "${SUBMISSION_DIR}/candidate.meta.npz" \
    --vocab-size 998 \
    --seq-len "${AUDIT_SEQ_LEN:-2048}"
}

run_train() {
  local mode="$1"
  local nproc iterations val_every max_seconds use_gptq gptq_ms
  if [[ "${mode}" == "smoke" ]]; then
    nproc="${NPROC_PER_NODE:-1}"
    iterations="${ITERATIONS:-20}"
    val_every="${VAL_LOSS_EVERY:-10}"
    max_seconds="${MAX_WALLCLOCK_SECONDS:-120}"
    use_gptq="${USE_GPTQ:-0}"
    gptq_ms="${GPTQ_RESERVE_MS:-0}"
  else
    nproc="${NPROC_PER_NODE:-$(detect_gpus)}"
    iterations="${ITERATIONS:-6716}"
    val_every="${VAL_LOSS_EVERY:-1000}"
    max_seconds="${MAX_WALLCLOCK_SECONDS:-600}"
    use_gptq="${USE_GPTQ:-1}"
    gptq_ms="${GPTQ_RESERVE_MS:-9000}"
  fi

  cd "${SUBMISSION_DIR}"
  SEED="${SEED:-42}" \
  DATA_PATH="${DATA_PATH}" \
  TOKENIZER_PATH="${SUBMISSION_DIR}/candidate.vocab" \
  TOKENIZER_META_PATH="${SUBMISSION_DIR}/candidate.meta.npz" \
  VOCAB_SIZE=998 \
  XSA_LAST_N=11 \
  USE_GPTQ="${use_gptq}" \
  GPTQ_RESERVE_MS="${gptq_ms}" \
  TTT_ENABLED=0 \
  BIGRAM_VOCAB_SIZE=2816 \
  BIGRAM_DIM=112 \
  SEGMENT_ADAPTER="${SEGMENT_ADAPTER:-1}" \
  SEGMENT_ADAPTER_LAYERS="${SEGMENT_ADAPTER_LAYERS:-8,9,10}" \
  SEGMENT_ADAPTER_RANK="${SEGMENT_ADAPTER_RANK:-16}" \
  SEGMENT_ADAPTER_KERNEL="${SEGMENT_ADAPTER_KERNEL:-64}" \
  SEGMENT_ADAPTER_GATE_INIT="${SEGMENT_ADAPTER_GATE_INIT:--3.0}" \
  SEGMENT_ADAPTER_BOUNDARY_SIGNAL="${SEGMENT_ADAPTER_BOUNDARY_SIGNAL:-1}" \
  ITERATIONS="${iterations}" \
  VAL_LOSS_EVERY="${val_every}" \
  MAX_WALLCLOCK_SECONDS="${max_seconds}" \
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
    run_train smoke
    ;;
  full)
    install_deps
    download_data
    run_audit
    run_train full
    ;;
  *)
    echo "Usage: bash runpod/scylla_segment_adapter_runpod.sh [install|data|audit|smoke|full]" >&2
    exit 2
    ;;
esac
