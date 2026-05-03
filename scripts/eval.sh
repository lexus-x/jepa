#!/usr/bin/env bash
# VL-JEPA evaluation script
# Evaluates trained models on LIBERO and/or MetaWorld benchmarks.
#
# Usage:
#   ./scripts/eval.sh --checkpoint=checkpoints/best.pt
#   ./scripts/eval.sh --checkpoint=checkpoints/best.pt --suite=libero_long
#   ./scripts/eval.sh --checkpoint=checkpoints/best.pt --benchmark=metaworld
#   SSH: ssh gpu-server 'cd vl-jepa && bash scripts/eval.sh --checkpoint=best.pt'

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Defaults
CHECKPOINT=""
BENCHMARK="libero"  # "libero" or "metaworld"
SUITE="libero_spatial"
METAWORLD_BENCHMARK="MT10"
NUM_EPISODES=20
DEVICE="cuda"
SAVE_VIDEOS="false"
VIDEO_DIR="eval_videos"
CONFIG="configs/config.yaml"

# ─── Parse arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint=*)       CHECKPOINT="${1#*=}" ;;
        --benchmark=*)        BENCHMARK="${1#*=}" ;;
        --suite=*)            SUITE="${1#*=}" ;;
        --metaworld=*)        METAWORLD_BENCHMARK="${1#*=}" ;;
        --num_episodes=*)     NUM_EPISODES="${1#*=}" ;;
        --device=*)           DEVICE="${1#*=}" ;;
        --save_videos)        SAVE_VIDEOS="true" ;;
        --video_dir=*)        VIDEO_DIR="${1#*=}" ;;
        --config=*)           CONFIG="${1#*=}" ;;
        *)                    echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

# ─── Validate ─────────────────────────────────────────────────────────
if [[ -z "$CHECKPOINT" ]]; then
    echo "Error: --checkpoint is required"
    echo "Usage: $0 --checkpoint=<path_to_checkpoint.pt> [options]"
    exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# ─── Environment setup ────────────────────────────────────────────────
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4

# ─── Launch evaluation ───────────────────────────────────────────────
echo "============================================"
echo "VL-JEPA Evaluation"
echo "============================================"
echo "  Checkpoint: $CHECKPOINT"
echo "  Benchmark:  $BENCHMARK"
echo "  Device:     $DEVICE"
echo "  Episodes:   $NUM_EPISODES"
echo "============================================"

if [[ "$BENCHMARK" == "libero" ]]; then
    echo "Evaluating on LIBERO suite: $SUITE"

    EVAL_ARGS=(
        "eval.checkpoint=$CHECKPOINT"
        "eval.benchmark=libero"
        "eval.suite=$SUITE"
        "eval.num_episodes=$NUM_EPISODES"
        "eval.device=$DEVICE"
        "eval.save_videos=$SAVE_VIDEOS"
        "eval.video_dir=$VIDEO_DIR"
    )

    python -m src.eval \
        --config-path="${PROJECT_DIR}/configs" \
        --config-name="$(basename "$CONFIG" .yaml)" \
        "${EVAL_ARGS[@]}"

elif [[ "$BENCHMARK" == "metaworld" ]]; then
    echo "Evaluating on MetaWorld benchmark: $METAWORLD_BENCHMARK"

    EVAL_ARGS=(
        "eval.checkpoint=$CHECKPOINT"
        "eval.benchmark=metaworld"
        "eval.metaworld_benchmark=$METAWORLD_BENCHMARK"
        "eval.num_episodes=$NUM_EPISODES"
        "eval.device=$DEVICE"
    )

    python -m src.eval \
        --config-path="${PROJECT_DIR}/configs" \
        --config-name="$(basename "$CONFIG" .yaml)" \
        "${EVAL_ARGS[@]}"

elif [[ "$BENCHMARK" == "all" ]]; then
    echo "Running full benchmark suite (LIBERO + MetaWorld)"

    # LIBERO all suites
    for s in libero_spatial libero_object libero_goal libero_long; do
        echo ""
        echo "--- LIBERO: $s ---"
        python -m src.eval \
            --config-path="${PROJECT_DIR}/configs" \
            --config-name="$(basename "$CONFIG" .yaml)" \
            "eval.checkpoint=$CHECKPOINT" \
            "eval.benchmark=libero" \
            "eval.suite=$s" \
            "eval.num_episodes=$NUM_EPISODES" \
            "eval.device=$DEVICE"
    done

    # MetaWorld MT10
    echo ""
    echo "--- MetaWorld: MT10 ---"
    python -m src.eval \
        --config-path="${PROJECT_DIR}/configs" \
        --config-name="$(basename "$CONFIG" .yaml)" \
        "eval.checkpoint=$CHECKPOINT" \
        "eval.benchmark=metaworld" \
        "eval.metaworld_benchmark=MT10" \
        "eval.num_episodes=$NUM_EPISODES" \
        "eval.device=$DEVICE"

else
    echo "Error: Unknown benchmark '$BENCHMARK'. Use 'libero', 'metaworld', or 'all'."
    exit 1
fi

echo ""
echo "============================================"
echo "Evaluation complete."
echo "============================================"
