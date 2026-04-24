#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-${RUN_MODE:-smoke}}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMISSION_DIR="${REPO_ROOT}/records/track_non_record_16mb/2026-04-24_DiffusionOnly_Scylla"
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
  python -m pip install -q packaging wheel setuptools huggingface-hub tokenmonster datasets tqdm numpy
  python - <<'PY'
import subprocess, sys
try:
    import torch
    ok = torch.cuda.is_available()
except Exception:
    ok = False
if not ok:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "torch==2.9.1",
        "--index-url", "https://download.pytorch.org/whl/cu128",
    ])
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
    nproc="${NPROC_PER_NODE:-1}"
    iterations="${ITERATIONS:-20000}"
    max_train="${MAX_TRAINING_SECONDS:-4800}"
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
  NUM_LAYERS="${NUM_LAYERS:-8}" \
  NUM_HEADS="${NUM_HEADS:-8}" \
  MLP_MULT="${MLP_MULT:-2.4}" \
  TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-131072}" \
  TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}" \
  EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-1024}" \
  VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-65536}" \
  MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}" \
  DIFFUSION_STEPS="${DIFFUSION_STEPS:-8}" \
  EVAL_NOISE_STEP="${EVAL_NOISE_STEP:-8}" \
  NOISE_RANDOM_PROB="${NOISE_RANDOM_PROB:-0.25}" \
  NOISE_PREFIX_PROB="${NOISE_PREFIX_PROB:-0.25}" \
  MASK_TOKEN_ID="${MASK_TOKEN_ID:-0}" \
  QK_GAIN_INIT="${QK_GAIN_INIT:-5.0}" \
  RECUR_LAYERS="${RECUR_LAYERS:-3,4,5}" \
  RECUR_REPEATS="${RECUR_REPEATS:-1}" \
  WEIGHT_DECAY="${WEIGHT_DECAY:-0.09}" \
  BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-2816}" \
  BIGRAM_DIM="${BIGRAM_DIM:-112}" \
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
  smoke)
    install_deps
    download_data
    REPO_ROOT="${REPO_ROOT}" python "${REPO_ROOT}/runpod/diffusion_only_forward_smoke.py"
    run_train smoke
    ;;
  full)
    install_deps
    download_data
    REPO_ROOT="${REPO_ROOT}" python "${REPO_ROOT}/runpod/diffusion_only_forward_smoke.py"
    run_train full
    ;;
  *)
    echo "Usage: bash runpod/diffusion_only_scylla_runpod.sh [install|data|smoke|full]" >&2
    exit 2
    ;;
esac
