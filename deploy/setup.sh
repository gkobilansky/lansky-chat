#!/bin/bash
# Simple VPS deployment script for LanBot
# Run on a fresh Ubuntu 22.04+ VPS

set -e

# Load config from .env if exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Defaults (override in .env)
HF_REPO="${HF_REPO:-gkobilansky/lanbot-checkpoints}"
MODEL_SOURCE="${MODEL_SOURCE:-mid}"
GITHUB_REPO="${GITHUB_REPO:-https://github.com/gkobilansky/lansky-chat.git}"

echo "=== LanBot VPS Setup ==="
echo "HF_REPO: $HF_REPO"
echo "MODEL_SOURCE: $MODEL_SOURCE"

# Install system dependencies
sudo apt update
sudo apt install -y python3.11 python3.11-venv git curl

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.cargo/env

# Clone repo (or pull if exists)
if [ ! -d "lansky-chat" ]; then
    git clone "$GITHUB_REPO"
fi
cd lansky-chat

# Copy .env if provided
if [ -f ../.env ]; then
    cp ../.env .
fi

# Create venv and install dependencies (CPU-only for VPS)
uv venv
source .venv/bin/activate
uv pip install -e ".[cpu]"

# Download model from HuggingFace
echo "Downloading model from HuggingFace..."
pip install huggingface_hub
python -c "
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${HF_REPO}',
    local_dir='checkpoints',
    allow_patterns=['${MODEL_SOURCE}_checkpoints/*', 'tokenizer/*'],
    token=os.environ.get('HF_TOKEN')
)
"

echo "=== Setup complete! ==="
echo "To start the server: python -m scripts.chat_web --port 8000"
echo "To install as service: sudo cp deploy/lanbot.service /etc/systemd/system/"
