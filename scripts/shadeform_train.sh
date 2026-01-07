#!/bin/bash
#
# Shadeform automated training script for LanBot
#
# Supports all training phases: midtraining, SFT, and RL.
# This script runs automatically on instance startup via Shadeform's
# launch_configuration. It downloads checkpoints, runs training,
# and uploads results before auto-delete kicks in.
#
# Usage: Base64 encode this script and pass to Shadeform API
#   SCRIPT_B64=$(base64 -i scripts/shadeform_train.sh)
#
# Training phases (set via TRAINING_PHASE env var):
#   mid     - Midtraining only (personality injection)
#   sft     - Supervised Fine-Tuning only (task completion)
#   rl      - Reinforcement Learning only (math reinforcement)
#   mid+sft - Midtraining then SFT
#   sft+rl  - SFT then RL
#   all     - Full pipeline: mid → sft → rl
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

# Training phase: mid, sft, rl, mid+sft, sft+rl, all
TRAINING_PHASE="${TRAINING_PHASE:-sft}"

# Training settings
RUN_NAME="${RUN_NAME:-lanbot-v3}"
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
# CLEANUP HANDLER - Always self-destruct on exit (success or failure)
# ============================================================================

SELF_DESTRUCT_DONE=false

cleanup_on_exit() {
    local exit_code=$?

    # Avoid running twice
    if [ "$SELF_DESTRUCT_DONE" = true ]; then
        return
    fi
    SELF_DESTRUCT_DONE=true

    echo ""
    echo "=== Cleanup on exit (code: $exit_code) ==="

    # Try to upload log (best effort)
    if [ -n "$HF_TOKEN" ] && [ -f "$LOG_FILE" ]; then
        local log_type="success"
        if [ $exit_code -ne 0 ]; then
            log_type="error"
        fi

        echo "Uploading $log_type log..."
        if command -v hf &> /dev/null; then
            hf upload "$HF_REPO" \
                "$LOG_FILE" \
                "logs/${log_type}-$(date +%Y%m%d-%H%M%S).log" \
                --token "$HF_TOKEN" \
                --commit-message "Training log: $RUN_NAME ($log_type)" 2>/dev/null || true
        elif [ -d "/root/lansky-chat/.venv" ]; then
            cd /root/lansky-chat 2>/dev/null || true
            uv run hf upload "$HF_REPO" \
                "$LOG_FILE" \
                "logs/${log_type}-$(date +%Y%m%d-%H%M%S).log" \
                --token "$HF_TOKEN" \
                --commit-message "Training log: $RUN_NAME ($log_type)" 2>/dev/null || true
        fi
    fi

    echo "Self-destructing to save costs..."
    self_destruct
}

# Always run cleanup on exit (normal, error, or signal)
trap cleanup_on_exit EXIT

echo "=============================================="
echo "LanBot Training Script"
echo "Phase: $TRAINING_PHASE"
echo "Started: $(date)"
echo "=============================================="

# System info
echo ""
echo "=== System Info ==="
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
echo "$GPU_INFO"
echo "CPUs: $(nproc)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"

# Count number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs: $NUM_GPUS"

# Auto-detect batch size based on GPU type
# Note: SFT has variable sequence lengths, so we use conservative batch sizes
# Midtraining can use larger batches due to fixed sequence lengths
if [ "$DEVICE_BATCH_SIZE" = "auto" ]; then
    if echo "$GPU_INFO" | grep -qi "B200"; then
        DEVICE_BATCH_SIZE=32
        echo "Detected B200 GPU - using batch size 32"
    elif echo "$GPU_INFO" | grep -qi "H200"; then
        DEVICE_BATCH_SIZE=24
        echo "Detected H200 GPU - using batch size 24"
    elif echo "$GPU_INFO" | grep -qi "H100"; then
        DEVICE_BATCH_SIZE=16
        echo "Detected H100 GPU - using batch size 16"
    elif echo "$GPU_INFO" | grep -qi "A100"; then
        DEVICE_BATCH_SIZE=8
        echo "Detected A100 GPU - using batch size 8"
    else
        DEVICE_BATCH_SIZE=4
        echo "Unknown GPU - using conservative batch size 4"
    fi
fi

# Compute batch parameters for each training script (grad_accum_steps=1 for efficiency)
# - mid_train.py uses total_batch_size (in tokens)
# - chat_sft.py uses target_examples_per_step (in examples)
# - chat_rl.py uses examples_per_step (in examples)
MAX_SEQ_LEN=2048
TOTAL_BATCH_SIZE=$((DEVICE_BATCH_SIZE * MAX_SEQ_LEN * NUM_GPUS))
TARGET_EXAMPLES_PER_STEP=$((DEVICE_BATCH_SIZE * NUM_GPUS))
EXAMPLES_PER_STEP=$((16 * NUM_GPUS))  # RL uses smaller batches, scale with GPUs
echo "Target examples per step (SFT): $TARGET_EXAMPLES_PER_STEP"
echo "Total batch size (mid): $TOTAL_BATCH_SIZE tokens"
echo "Examples per step (RL): $EXAMPLES_PER_STEP"

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

mkdir -p "$CACHE_DIR"

# Login to HuggingFace if token provided
if [ -n "$HF_TOKEN" ]; then
    echo "Logging into HuggingFace..."
    uv run hf auth login --token "$HF_TOKEN"
fi

# Download tokenizer (always needed)
echo "Downloading tokenizer..."
uv run hf download "$HF_REPO" \
    --include "tokenizer/*" \
    --local-dir "$CACHE_DIR"

# Download checkpoints based on training phase
case "$TRAINING_PHASE" in
    mid|all|mid+sft)
        echo "Downloading base checkpoint (needed for midtraining)..."
        uv run hf download "$HF_REPO" \
            --include "base_checkpoints/d20/*" \
            --local-dir "$CACHE_DIR"
        ;;
    sft|sft+rl)
        echo "Downloading mid checkpoint (needed for SFT)..."
        uv run hf download "$HF_REPO" \
            --include "mid_checkpoints/d20/*" \
            --local-dir "$CACHE_DIR"
        ;;
    rl)
        echo "Downloading SFT checkpoint (needed for RL)..."
        uv run hf download "$HF_REPO" \
            --include "chatsft_checkpoints/d20/*" \
            --local-dir "$CACHE_DIR"
        ;;
    *)
        echo "ERROR: Unknown training phase: $TRAINING_PHASE"
        echo "Valid phases: mid, sft, rl, mid+sft, sft+rl, all"
        exit 1
        ;;
esac

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
# TRAINING FUNCTIONS
# ============================================================================

# Set WandB if configured, otherwise disable it
setup_wandb() {
    if [ -n "$WANDB_API_KEY" ]; then
        export WANDB_API_KEY
        export WANDB_PROJECT
        echo "WandB logging enabled (project: $WANDB_PROJECT)"
    else
        export WANDB_MODE=disabled
        echo "WandB disabled (no API key)"
    fi
}

# Helper to run training with torchrun for multi-GPU or python for single GPU
# Each training script has different batch parameters:
#   mid_train.py: --total_batch_size (in tokens)
#   chat_sft.py: --target_examples_per_step (in examples)
#   chat_rl.py: --examples_per_step (in examples)
run_training() {
    local module="$1"
    local run_suffix="$2"
    shift 2
    local extra_args=("$@")

    # Build batch size argument based on which script we're running
    local batch_arg=""
    case "$module" in
        scripts.mid_train)
            batch_arg="--total_batch_size=$TOTAL_BATCH_SIZE"
            ;;
        scripts.chat_sft)
            batch_arg="--target_examples_per_step=$TARGET_EXAMPLES_PER_STEP"
            ;;
        scripts.chat_rl)
            batch_arg="--examples_per_step=$EXAMPLES_PER_STEP"
            ;;
    esac

    if [ "$NUM_GPUS" -gt 1 ]; then
        echo "Using torchrun with $NUM_GPUS GPUs"
        uv run torchrun --standalone --nproc_per_node="$NUM_GPUS" \
            -m "$module" -- \
            --run="${RUN_NAME}-${run_suffix}" \
            --device_batch_size="$DEVICE_BATCH_SIZE" \
            $batch_arg \
            "${extra_args[@]}"
    else
        echo "Using single GPU"
        uv run python -m "$module" \
            --run="${RUN_NAME}-${run_suffix}" \
            --device_batch_size="$DEVICE_BATCH_SIZE" \
            $batch_arg \
            "${extra_args[@]}"
    fi
}

# Run midtraining phase
run_midtraining() {
    echo ""
    echo "=== Running Midtraining ==="
    echo "Run name: ${RUN_NAME}-mid"
    echo "Device batch size: $DEVICE_BATCH_SIZE"
    echo "Total batch size: $TOTAL_BATCH_SIZE tokens"
    echo "Start time: $(date)"

    run_training scripts.mid_train mid

    local exit_code=$?
    echo "Midtraining exit code: $exit_code"
    echo "Midtraining completed: $(date)"

    if [ $exit_code -eq 0 ]; then
        echo "Uploading mid checkpoints..."
        uv run hf upload "$HF_REPO" \
            "$CACHE_DIR/mid_checkpoints" \
            "mid_checkpoints" \
            --commit-message "Midtraining: $RUN_NAME ($(date +%Y-%m-%d))"
    fi

    return $exit_code
}

# Run SFT phase
run_sft() {
    echo ""
    echo "=== Running SFT (Supervised Fine-Tuning) ==="
    echo "Run name: ${RUN_NAME}-sft"
    echo "Device batch size: $DEVICE_BATCH_SIZE"
    echo "Target examples per step: $TARGET_EXAMPLES_PER_STEP"
    echo "Start time: $(date)"

    run_training scripts.chat_sft sft

    local exit_code=$?
    echo "SFT exit code: $exit_code"
    echo "SFT completed: $(date)"

    if [ $exit_code -eq 0 ]; then
        echo "Uploading SFT checkpoints..."
        uv run hf upload "$HF_REPO" \
            "$CACHE_DIR/chatsft_checkpoints" \
            "chatsft_checkpoints" \
            --commit-message "SFT: $RUN_NAME ($(date +%Y-%m-%d))"
    fi

    return $exit_code
}

# Run RL phase
run_rl() {
    echo ""
    echo "=== Running RL (Reinforcement Learning) ==="
    echo "Run name: ${RUN_NAME}-rl"
    echo "Device batch size: $DEVICE_BATCH_SIZE"
    echo "Examples per step: $EXAMPLES_PER_STEP"
    echo "Start time: $(date)"

    run_training scripts.chat_rl rl

    local exit_code=$?
    echo "RL exit code: $exit_code"
    echo "RL completed: $(date)"

    if [ $exit_code -eq 0 ]; then
        echo "Uploading RL checkpoints..."
        uv run hf upload "$HF_REPO" \
            "$CACHE_DIR/chatrl_checkpoints" \
            "chatrl_checkpoints" \
            --commit-message "RL: $RUN_NAME ($(date +%Y-%m-%d))"
    fi

    return $exit_code
}

# ============================================================================
# STEP 4: Run training phase(s)
# ============================================================================

echo ""
echo "=== Step 4: Training (phase: $TRAINING_PHASE) ==="
setup_wandb

TRAINING_EXIT_CODE=0

case "$TRAINING_PHASE" in
    mid)
        run_midtraining
        TRAINING_EXIT_CODE=$?
        ;;
    sft)
        run_sft
        TRAINING_EXIT_CODE=$?
        ;;
    rl)
        run_rl
        TRAINING_EXIT_CODE=$?
        ;;
    mid+sft)
        run_midtraining && run_sft
        TRAINING_EXIT_CODE=$?
        ;;
    sft+rl)
        run_sft && run_rl
        TRAINING_EXIT_CODE=$?
        ;;
    all)
        run_midtraining && run_sft && run_rl
        TRAINING_EXIT_CODE=$?
        ;;
esac

echo ""
echo "Training phase(s) completed with exit code: $TRAINING_EXIT_CODE"

# ============================================================================
# DONE - EXIT trap will handle log upload and self-destruct
# ============================================================================

echo ""
echo "=============================================="
echo "Training script completed!"
echo "Phase: $TRAINING_PHASE"
echo "End time: $(date)"
echo "Exit code: $TRAINING_EXIT_CODE"
echo "=============================================="
echo ""
echo "Check HuggingFace repo for results: https://huggingface.co/$HF_REPO"

# Exit with training exit code - EXIT trap will handle cleanup
exit $TRAINING_EXIT_CODE
