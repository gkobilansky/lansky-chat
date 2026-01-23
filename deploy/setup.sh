#!/bin/bash
# Simple VPS deployment script for LanBot
# Run on a fresh Ubuntu 22.04+ VPS

set -e

echo "=== LanBot VPS Setup ==="

# Install system dependencies
sudo apt update
sudo apt install -y python3.11 python3.11-venv git curl

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.cargo/env

# Clone repo (or pull if exists)
if [ ! -d "lansky-chat" ]; then
    git clone https://github.com/gkobilansky/lansky-chat.git
fi
cd lansky-chat

# Create venv and install dependencies (CPU-only for VPS)
uv venv
source .venv/bin/activate
uv pip install -e ".[cpu]"

# Download model from HuggingFace
echo "Downloading model from HuggingFace..."
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='gkobilansky/lanbot-checkpoints',
    local_dir='checkpoints',
    allow_patterns=['mid_checkpoints/*', 'tokenizer/*']
)
"

echo "=== Setup complete! ==="
echo "To start the server: python -m scripts.chat_web --port 8000"
echo "To install as service: sudo cp deploy/lanbot.service /etc/systemd/system/"
