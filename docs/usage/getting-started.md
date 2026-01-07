# Getting Started

## Prerequisites

### Required
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Git

### For Cloud Training (Recommended)
- [Shadeform](https://shadeform.ai) account and API key
- [HuggingFace](https://huggingface.co) account and token (for checkpoint storage)

### For Local Training (Limited)
- MacBook with M-series chip (MPS), or
- Linux machine with NVIDIA GPU

> **Note:** Full training requires significant GPU resources (8×H100 or similar). Local machines can only run small experiments or inference.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/gkobilansky/lansky-chat.git
cd lansky-chat
```

### 2. Install Dependencies

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync --extra gpu

# Activate the environment
source .venv/bin/activate
```

### 3. Build the Rust Tokenizer

```bash
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Required for cloud training
SHADEFORM_API_KEY=your-shadeform-api-key
HF_TOKEN=your-huggingface-token

# Optional: WandB for training metrics
WANDB_API_KEY=your-wandb-key
```

Get your keys:
- Shadeform: https://platform.shadeform.ai/settings/api-keys
- HuggingFace: https://huggingface.co/settings/tokens (needs write access)
- WandB: https://wandb.ai/settings

## Verify Installation

### Check Python Environment

```bash
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}')"
uv run python -c "import nanochat; print('nanochat: OK')"
```

### Check Tokenizer

```bash
uv run python -c "from nanochat.tokenizer import Tokenizer; print('Tokenizer: OK')"
```

### Check API Keys

```bash
# Verify Shadeform connection
source .env
curl -s -X GET "https://api.shadeform.ai/v1/instances/types" \
    -H "X-API-KEY: $SHADEFORM_API_KEY" | \
    python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Shadeform: {len(d.get(\"instance_types\",[]))} GPU types available')"
```

## Project Structure

```
lansky-chat/
├── scripts/                    # Training and utility scripts
│   ├── launch_shadeform.sh     # Launch cloud training
│   ├── shadeform_train.sh      # Runs on cloud instance
│   ├── check_training_status.sh # Monitor training
│   ├── mid_train.py            # Midtraining script
│   ├── chat_sft.py             # SFT training script
│   ├── chat_rl.py              # RL training script
│   └── chat_cli.py             # Chat with your model
├── dev/
│   └── gen_synthetic_data.py   # Generate identity data
├── nanochat/                   # Core library
│   ├── gpt.py                  # Model architecture
│   ├── engine.py               # Inference engine
│   └── tokenizer.py            # Tokenizer
├── tasks/                      # Training tasks
├── docs/
│   ├── usage/                  # This documentation
│   └── research/               # Gameplan and notes
└── my-checkpoints/             # Local checkpoint storage
```

## Next Steps

1. **New to LLM training?** Start with [Training Pipeline](training-pipeline.md) to understand the phases.

2. **Want to customize personality?** See [Personality Customization](personality-customization.md).

3. **Ready to train?** Jump to [Cloud Training](cloud-training.md) to launch a training run.

4. **Just want to chat?** If you have checkpoints, run:
   ```bash
   uv run python -m scripts.chat_cli
   ```

## Troubleshooting

### "Module not found" errors
```bash
# Make sure you're in the virtual environment
source .venv/bin/activate

# Reinstall dependencies
uv sync --extra gpu
```

### Rust tokenizer build fails
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# Retry build
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

### CUDA out of memory (local training)
- Reduce `device_batch_size` in the training script
- Use a smaller model (`--depth=12` instead of `--depth=20`)
- Consider cloud training for full-size models
