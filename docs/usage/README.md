# LanBot Training Guide

This documentation covers how to use the lansky-chat fork of nanochat to train your own personalized LLM.

## What's Different from Nanochat

This fork adds:
- **Automated cloud training** via Shadeform scripts (hands-off GPU training)
- **Personality injection workflow** with LanBot identity
- **Multi-phase training support** (mid → SFT → RL in one command)
- **HuggingFace integration** for checkpoint storage

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](getting-started.md) | Prerequisites, setup, and first steps |
| [Cloud Training](cloud-training.md) | Using Shadeform scripts for GPU training |
| [Personality Customization](personality-customization.md) | Creating your model's identity |
| [Training Pipeline](training-pipeline.md) | Understanding mid → SFT → RL phases |

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/gkobilansky/lansky-chat.git
cd lansky-chat
uv sync --extra gpu

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 3. Launch SFT training on cloud GPU
./scripts/launch_shadeform.sh --phase sft

# 4. Monitor progress
./scripts/check_training_status.sh --watch
```

## Training Phases

```
Base Model (pretrained)
    ↓
Midtraining (personality injection)  ← You are here after Phase 2
    ↓
SFT (task completion)                ← Next step
    ↓
RL (math reinforcement)
    ↓
Agent Harness (tool use)
```

## Current Status

See [agentic-llm-gameplan.md](../research/agentic-llm-gameplan.md) for detailed progress tracking.

## Cost Estimates

| Phase | GPU | Time | Cost |
|-------|-----|------|------|
| Midtraining | 8×B200 | ~45 min | ~$25 |
| SFT | 8×H100 | ~1 hour | ~$24 |
| RL | 8×H100 | ~1 hour | ~$24 |
| **Full pipeline** | 8×H100 | ~4 hours | ~$100 |
