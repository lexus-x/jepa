#!/usr/bin/env bash
# download_checkpoints.sh — Download V-JEPA 2 pre-trained checkpoints.
#
# Usage:
#   bash tools/download_checkpoints.sh [--output-dir DIR]
#
# Checkpoints are hosted by Meta AI on Hugging Face.
# Requires: curl or wget, and optionally huggingface-cli.

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

OUTPUT_DIR="${1:-checkpoints/vjepa2}"
HF_REPO="facebook/vjepa2-vitg-fpc64-256"
HF_BASE="https://huggingface.co/${HF_REPO}/resolve/main"

FILES=(
    "vjepa2-vitg-fpc64-256.pt"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

log() { printf "\033[1;34m[download]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[warning]\033[0m %s\n" "$*"; }
err() { printf "\033[1;31m[error]\033[0m %s\n" "$*" >&2; exit 1; }

check_tool() {
    command -v "$1" &>/dev/null || err "'$1' is required but not installed."
}

download_file() {
    local url="$1" dest="$2"
    if [[ -f "$dest" ]]; then
        log "Already exists: $dest — skipping."
        return 0
    fi
    log "Downloading: $url"
    if command -v curl &>/dev/null; then
        curl -fSL --progress-bar -o "$dest" "$url"
    elif command -v wget &>/dev/null; then
        wget --progress=bar:force -O "$dest" "$url"
    else
        err "Neither 'curl' nor 'wget' found. Please install one."
    fi
    log "Saved: $dest ($(du -sh "$dest" | cut -f1))"
}

# ── Main ──────────────────────────────────────────────────────────────────────

main() {
    # Parse --output-dir flag
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
            -h|--help)
                echo "Usage: bash tools/download_checkpoints.sh [--output-dir DIR]"
                echo ""
                echo "Downloads V-JEPA 2 checkpoints from Hugging Face."
                echo "Default output: checkpoints/vjepa2/"
                exit 0
                ;;
            *) OUTPUT_DIR="$1"; shift ;;
        esac
    done

    mkdir -p "$OUTPUT_DIR"

    log "Output directory: $OUTPUT_DIR"
    log "Repository: $HF_REPO"
    echo ""

    for fname in "${FILES[@]}"; do
        download_file "${HF_BASE}/${fname}" "${OUTPUT_DIR}/${fname}"
    done

    echo ""
    log "All checkpoints downloaded to: $OUTPUT_DIR/"
    log "Done."
}

main "$@"
