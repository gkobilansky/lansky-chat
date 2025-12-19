#!/bin/bash
#
# Shadeform automated training script for LanBot midtraining
#
# This script runs automatically on instance startup via Shadeform's
# launch_configuration. It downloads checkpoints, runs midtraining,
# and uploads results before auto-delete kicks in.
#
# Usage: Base64 encode this script and pass to Shadeform API
#   SCRIPT_B64=$(base64 -i scripts/shadeform_train.sh)
#
set -e

# ============================================================================
# CONFIGURATION - Edit these before running
# ============================================================================

# HuggingFace Hub settings (recommended for checkpoint storage)
HF_REPO="gkobilansky/lanbot-checkpoints"  # Your HF repo for checkpoints
HF_TOKEN="${HF_TOKEN:-}"                   # Set via Shadeform envs or here

# GitHub repo
GITHUB_REPO="https://github.com/gkobilansky/lansky-chat.git"
GITHUB_BRANCH="master"

# Training settings
RUN_NAME="lanbot-v2"
# Batch size will be auto-detected based on GPU type
# B200 (192GB): 64, H100 (80GB): 32, A100 (40/80GB): 16-32
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-auto}"

# WandB (optional)
WANDB_PROJECT="${WANDB_PROJECT:-lanbot}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

# Shadeform self-destruct (set via launch script envs)
SHADEFORM_API_KEY="${SHADEFORM_API_KEY:-}"
INSTANCE_NAME="${INSTANCE_NAME:-}"

# ============================================================================
# SETUP - Don't edit below unless you know what you're doing
# ============================================================================

LOG_FILE="/var/log/lanbot-training.log"
CACHE_DIR="/root/.cache/nanochat"
SHADEFORM_API="https://api.shadeform.ai/v1"

# Log everything
exec > >(tee -a "$LOG_FILE") 2>&1

# ============================================================================
# SELF-DESTRUCT FUNCTION
# ============================================================================

self_destruct() {
    echo ""
    echo "=== Self-Destruct Sequence ==="

    if [ -z "$SHADEFORM_API_KEY" ] || [ -z "$INSTANCE_NAME" ]; then
        echo "WARNING: SHADEFORM_API_KEY or INSTANCE_NAME not set, skipping self-destruct"
        echo "Instance will be deleted by spend_threshold instead"
        return 0
    fi

    echo "Looking up instance ID for: $INSTANCE_NAME"

    # Get instance ID by listing instances and finding by name
    INSTANCES_JSON=$(curl -s -X GET "$SHADEFORM_API/instances" \
        -H "X-API-KEY: $SHADEFORM_API_KEY")

    INSTANCE_ID=$(echo "$INSTANCES_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for inst in data.get('instances', []):
    if inst.get('name') == '$INSTANCE_NAME':
        print(inst.get('id', ''))
        break
" 2>/dev/null)

    if [ -z "$INSTANCE_ID" ]; then
        echo "ERROR: Could not find instance ID for $INSTANCE_NAME"
        echo "Instance will be deleted by spend_threshold instead"
        return 1
    fi

    echo "Found instance ID: $INSTANCE_ID"
    echo "Deleting instance..."

    DELETE_RESPONSE=$(curl -s -X POST "$SHADEFORM_API/instances/$INSTANCE_ID/delete" \
        -H "X-API-KEY: $SHADEFORM_API_KEY")

    echo "Delete response: $DELETE_RESPONSE"
    echo "Self-destruct initiated. Goodbye!"
}

# ============================================================================
# ERROR HANDLER - Upload logs and self-destruct on failure
# ============================================================================

handle_error() {
    local exit_code=$?
    local line_number=$1

    echo ""
    echo "!!! ERROR on line $line_number (exit code: $exit_code) !!!"
    echo "Uploading error log before self-destruct..."

    # Try to upload log even on failure (best effort)
    if [ -n "$HF_TOKEN" ] && command -v hf &> /dev/null; then
        hf upload "$HF_REPO" \
            "$LOG_FILE" \
            "logs/error-$(date +%Y%m%d-%H%M%S).log" \
            --token "$HF_TOKEN" \
            --commit-message "Error log: $RUN_NAME (line $line_number)" 2>/dev/null || true
    elif [ -n "$HF_TOKEN" ] && [ -d ".venv" ]; then
        uv run hf upload "$HF_REPO" \
            "$LOG_FILE" \
            "logs/error-$(date +%Y%m%d-%H%M%S).log" \
            --token "$HF_TOKEN" \
            --commit-message "Error log: $RUN_NAME (line $line_number)" 2>/dev/null || true
    fi

    echo "Log upload attempted. Now self-destructing to save costs..."
    self_destruct

    exit $exit_code
}

# Trap errors and call handler with line number
trap 'handle_error $LINENO' ERR

echo "=============================================="
echo "LanBot Training Script"
echo "Started: $(date)"
echo "=============================================="

# System info
echo ""
echo "=== System Info ==="
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
echo "$GPU_INFO"
echo "CPUs: $(nproc)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"

# Auto-detect batch size based on GPU type
if [ "$DEVICE_BATCH_SIZE" = "auto" ]; then
    if echo "$GPU_INFO" | grep -qi "B200"; then
        DEVICE_BATCH_SIZE=64
        echo "Detected B200 GPU - using batch size 64"
    elif echo "$GPU_INFO" | grep -qi "H200"; then
        DEVICE_BATCH_SIZE=48
        echo "Detected H200 GPU - using batch size 48"
    elif echo "$GPU_INFO" | grep -qi "H100"; then
        DEVICE_BATCH_SIZE=32
        echo "Detected H100 GPU - using batch size 32"
    elif echo "$GPU_INFO" | grep -qi "A100"; then
        DEVICE_BATCH_SIZE=16
        echo "Detected A100 GPU - using batch size 16"
    else
        DEVICE_BATCH_SIZE=8
        echo "Unknown GPU - using conservative batch size 8"
    fi
fi

# ============================================================================
# STEP 1: Clone repo and install dependencies
# ============================================================================

echo ""
echo "=== Step 1: Setting up environment ==="

cd /root

if [ -d "lansky-chat" ]; then
    echo "Repo already exists, pulling latest..."
    cd lansky-chat
    git pull origin "$GITHUB_BRANCH"
else
    echo "Cloning repo..."
    git clone --branch "$GITHUB_BRANCH" "$GITHUB_REPO"
    cd lansky-chat
fi

# Install Rust (needed for maturin/rustbpe)
echo "Installing Rust..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Install uv (fast Python package manager)
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Create venv and install dependencies
echo "Creating virtual environment and installing dependencies..."
uv venv
uv sync --extra gpu
uv pip install huggingface_hub[cli]

# Build Rust BPE tokenizer
echo "Building Rust BPE tokenizer..."
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# ============================================================================
# STEP 2: Download checkpoints from HuggingFace
# ============================================================================

echo ""
echo "=== Step 2: Downloading checkpoints ==="

mkdir -p "$CACHE_DIR/base_checkpoints/d20"
mkdir -p "$CACHE_DIR/tokenizer"

# Login to HuggingFace if token provided
if [ -n "$HF_TOKEN" ]; then
    echo "Logging into HuggingFace..."
    uv run hf auth login --token "$HF_TOKEN"
fi

# Download base checkpoint
echo "Downloading base checkpoint..."
uv run hf download "$HF_REPO" \
    --include "base_checkpoints/d20/*" \
    --local-dir "$CACHE_DIR"

# Download tokenizer
echo "Downloading tokenizer..."
uv run hf download "$HF_REPO" \
    --include "tokenizer/*" \
    --local-dir "$CACHE_DIR"

echo "Checkpoint download complete!"
ls -la "$CACHE_DIR/"

# ============================================================================
# STEP 3: Download synthetic training data
# ============================================================================

echo ""
echo "=== Step 3: Downloading training data ==="

# Download identity conversations (generated by gen_synthetic_data.py)
uv run hf download "$HF_REPO" \
    --include "*.jsonl" \
    --local-dir "$CACHE_DIR"

echo "Training data files:"
ls -la "$CACHE_DIR"/*.jsonl 2>/dev/null || echo "No JSONL files found - will use defaults"

# ============================================================================
# STEP 4: Run midtraining
# ============================================================================

echo ""
echo "=== Step 4: Starting midtraining ==="
echo "Run name: $RUN_NAME"
echo "Device batch size: $DEVICE_BATCH_SIZE"
echo "Start time: $(date)"

# Set WandB if configured, otherwise disable it
if [ -n "$WANDB_API_KEY" ]; then
    export WANDB_API_KEY
    export WANDB_PROJECT
    echo "WandB logging enabled (project: $WANDB_PROJECT)"
else
    export WANDB_MODE=disabled
    echo "WandB disabled (no API key)"
fi

# Run midtraining
uv run python -m scripts.mid_train \
    --run="$RUN_NAME" \
    --device_batch_size="$DEVICE_BATCH_SIZE"

MID_EXIT_CODE=$?
echo "Midtraining exit code: $MID_EXIT_CODE"
echo "Midtraining completed: $(date)"

# ============================================================================
# STEP 5: Upload results
# ============================================================================

echo ""
echo "=== Step 5: Uploading results ==="

if [ $MID_EXIT_CODE -eq 0 ]; then
    echo "Uploading mid checkpoints to HuggingFace..."

    # Find the latest checkpoint
    LATEST_MID=$(ls -t "$CACHE_DIR/mid_checkpoints/d20/"model_*.pt 2>/dev/null | head -1)

    if [ -n "$LATEST_MID" ]; then
        echo "Found checkpoint: $LATEST_MID"

        # Upload mid checkpoints
        uv run hf upload "$HF_REPO" \
            "$CACHE_DIR/mid_checkpoints" \
            "mid_checkpoints" \
            --commit-message "Midtraining run: $RUN_NAME ($(date +%Y-%m-%d))"

        echo "Upload complete!"
    else
        echo "ERROR: No checkpoint found to upload"
    fi
else
    echo "ERROR: Midtraining failed, skipping upload"
fi

# Upload logs regardless
echo "Uploading training log..."
uv run hf upload "$HF_REPO" \
    "$LOG_FILE" \
    "logs/training-$(date +%Y%m%d-%H%M%S).log" \
    --commit-message "Training log: $RUN_NAME"

# ============================================================================
# DONE
# ============================================================================

echo ""
echo "=============================================="
echo "Training script completed!"
echo "End time: $(date)"
echo "Exit code: $MID_EXIT_CODE"
echo "=============================================="
echo ""
echo "Check HuggingFace repo for results: https://huggingface.co/$HF_REPO"

# ============================================================================
# STEP 6: Self-destruct
# ============================================================================

# Delete instance to stop billing (auto_delete is backup if this fails)
self_destruct

exit $MID_EXIT_CODE
