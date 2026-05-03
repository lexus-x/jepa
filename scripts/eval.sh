#!/usr/bin/env bash
# SE(3) Conformal Safety — evaluation script
#
# Usage:
#   ./scripts/eval.sh --policy=openvla --suite=libero_spatial
#   ./scripts/eval.sh --policy=pi0 --benchmark=mt10
#   ./scripts/eval.sh --policy=random --all

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Defaults
POLICY="random"
BENCHMARK="libero"
SUITE="libero_spatial"
MT_BENCHMARK="MT10"
NUM_EPISODES=20
DEVICE="cuda"
ALPHA=0.1
MAX_RADIUS=2.0
ALL="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --policy=*)       POLICY="${1#*=}" ;;
        --benchmark=*)    BENCHMARK="${1#*=}" ;;
        --suite=*)        SUITE="${1#*=}" ;;
        --mt=*)           MT_BENCHMARK="${1#*=}" ;;
        --num_episodes=*) NUM_EPISODES="${1#*=}" ;;
        --device=*)       DEVICE="${1#*=}" ;;
        --alpha=*)        ALPHA="${1#*=}" ;;
        --max_radius=*)   MAX_RADIUS="${1#*=}" ;;
        --all)            ALL="true" ;;
        *)                echo "Unknown: $1"; exit 1 ;;
    esac
    shift
done

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

echo "============================================"
echo "SE(3) Conformal Safety Evaluation"
echo "============================================"
echo "  Policy:       $POLICY"
echo "  Benchmark:    $BENCHMARK"
echo "  α:            $ALPHA"
echo "  Max radius:   $MAX_RADIUS"
echo "  Episodes:     $NUM_EPISODES"
echo "============================================"

if [[ "$ALL" == "true" ]]; then
    # Run all benchmarks
    for suite in libero_spatial libero_object libero_goal libero_long; do
        echo ""
        echo "--- LIBERO: $suite ---"
        python3 -m src.run_eval \
            --policy="$POLICY" \
            --benchmark=libero \
            --suite="$suite" \
            --num_episodes="$NUM_EPISODES" \
            --device="$DEVICE" \
            --alpha="$ALPHA" \
            --max_radius="$MAX_RADIUS"
    done
    echo ""
    echo "--- MetaWorld: MT10 ---"
    python3 -m src.run_eval \
        --policy="$POLICY" \
        --benchmark=metaworld \
        --mt=MT10 \
        --num_episodes="$NUM_EPISODES" \
        --device="$DEVICE" \
        --alpha="$ALPHA" \
        --max_radius="$MAX_RADIUS"
else
    python3 -m src.run_eval \
        --policy="$POLICY" \
        --benchmark="$BENCHMARK" \
        --suite="$SUITE" \
        --mt="$MT_BENCHMARK" \
        --num_episodes="$NUM_EPISODES" \
        --device="$DEVICE" \
        --alpha="$ALPHA" \
        --max_radius="$MAX_RADIUS"
fi

echo ""
echo "Done."
