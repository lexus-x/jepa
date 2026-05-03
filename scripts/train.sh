#!/usr/bin/env bash
# VL-JEPA training script
# Supports single-GPU and multi-GPU (via torchrun) training.
#
# Usage:
#   ./scripts/train.sh                          # Single GPU
#   ./scripts/train.sh --nproc_per_node=4       # 4 GPUs on one node
#   ./scripts/train.sh --nnodes=2 --nproc_per_node=4  # Multi-node
#   SSH: ssh gpu-server 'cd vl-jepa && bash scripts/train.sh --nproc_per_node=4'

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Defaults
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
CONFIG="${CONFIG:-configs/config.yaml}"
RESUME="${RESUME:-}"
DEBUG="${DEBUG:-false}"

# ─── Parse arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --nproc_per_node=*)  NPROC_PER_NODE="${1#*=}" ;;
        --nnodes=*)          NNODES="${1#*=}" ;;
        --node_rank=*)       NODE_RANK="${1#*=}" ;;
        --master_addr=*)     MASTER_ADDR="${1#*=}" ;;
        --master_port=*)     MASTER_PORT="${1#*=}" ;;
        --config=*)          CONFIG="${1#*=}" ;;
        --resume=*)          RESUME="${1#*=}" ;;
        --debug)             DEBUG="true" ;;
        *)                   echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

# ─── Environment setup ────────────────────────────────────────────────
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

if [[ "$DEBUG" == "true" ]]; then
    export TORCH_DISTRIBUTED_DEBUG=DETAIL
    export CUDA_LAUNCH_BLOCKING=1
fi

# ─── Build command ────────────────────────────────────────────────────
PYTHON_CMD="python -m src.training.train_hydra"

HYDRA_ARGS=(
    "--config-path=${PROJECT_DIR}/configs"
    "--config-name=$(basename "$CONFIG" .yaml)"
)

if [[ -n "$RESUME" ]]; then
    HYDRA_ARGS+=("training.checkpoint.resume_from=$RESUME")
fi

# ─── Launch ───────────────────────────────────────────────────────────
if [[ "$NPROC_PER_NODE" -gt 1 || "$NNODES" -gt 1 ]]; then
    echo "Launching distributed training:"
    echo "  Nodes: $NNODES, GPUs/node: $NPROC_PER_NODE"
    echo "  Master: $MASTER_ADDR:$MASTER_PORT"
    echo "  Config: $CONFIG"

    torchrun \
        --nnodes="$NNODES" \
        --nproc_per_node="$NPROC_PER_NODE" \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        --rdzv_backend=static \
        -m src.training.train_hydra \
        "${HYDRA_ARGS[@]}"
else
    echo "Launching single-GPU training"
    echo "  Config: $CONFIG"

    python -m src.training.train_hydra \
        "${HYDRA_ARGS[@]}"
fi

echo "Training complete."
