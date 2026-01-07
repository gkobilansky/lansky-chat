# Cloud Training with Shadeform

This guide covers the automated cloud training workflow using Shadeform GPU instances.

## Overview

The training workflow is designed for hands-off operation:

```
launch_shadeform.sh (local)
    ↓ creates instance with startup script
shadeform_train.sh (runs on cloud)
    ↓ clones repo, downloads checkpoints, trains
check_training_status.sh (local)
    ↓ monitors progress
Results uploaded to HuggingFace
    ↓
Instance self-destructs (saves money)
```

## Prerequisites

1. **Shadeform account** with API key
2. **HuggingFace account** with write token
3. **Checkpoints uploaded** to HuggingFace (see [Initial Setup](#initial-setup))

## Quick Start

```bash
# Run SFT training (default)
./scripts/launch_shadeform.sh

# Monitor progress
./scripts/check_training_status.sh --watch
```

## Training Phases

The `--phase` flag controls which training phase(s) to run:

| Phase | Description | Input Checkpoint | Output |
|-------|-------------|------------------|--------|
| `mid` | Personality injection | base_checkpoints | mid_checkpoints |
| `sft` | Task completion | mid_checkpoints | chatsft_checkpoints |
| `rl` | Math reinforcement | chatsft_checkpoints | chatrl_checkpoints |
| `mid+sft` | Mid then SFT | base_checkpoints | both |
| `sft+rl` | SFT then RL | mid_checkpoints | both |
| `all` | Full pipeline | base_checkpoints | all |

### Examples

```bash
# Run just SFT (most common after midtraining)
./scripts/launch_shadeform.sh --phase sft

# Run SFT followed by RL in one instance
./scripts/launch_shadeform.sh --phase sft+rl

# Run the full pipeline from base model
./scripts/launch_shadeform.sh --phase all

# Dry run to preview without launching
./scripts/launch_shadeform.sh --phase sft --dry-run
```

## Launch Script Options

```bash
./scripts/launch_shadeform.sh [OPTIONS]

Options:
  --phase PHASE       Training phase: mid, sft, rl, mid+sft, sft+rl, all
                      Default: sft

  --run-name NAME     Training run name (appears in WandB, logs)
                      Default: lanbot-v3

  --gpu TYPE          GPU type to search for
                      Default: B200

  --num-gpus N        Number of GPUs
                      Default: 8

  --spend-limit N     Auto-delete at this spend amount (USD)
                      Default: 150

  --all-configs       Show all GPU configurations (1x, 2x, 4x, 8x)

  --dry-run           Preview launch configuration without creating instance
```

## Monitoring Training

### One-time Status Check

```bash
./scripts/check_training_status.sh
```

Shows:
- Active instances and their status
- Current spend
- SSH connection command
- HuggingFace upload status

### Continuous Monitoring

```bash
# Poll every 60 seconds (default)
./scripts/check_training_status.sh --watch

# Poll every 30 seconds
./scripts/check_training_status.sh --watch 30
```

### SSH into Running Instance

The status script shows the SSH command. Example:

```bash
ssh -i .ssh_keys/lanbot-sft-xxx.pem ubuntu@203.0.113.42

# View training logs
tail -f /var/log/lanbot-training.log
```

## Cost Management

### Auto-Delete Safety Net

Instances automatically delete when spend reaches the threshold:

```bash
# Default: $150
./scripts/launch_shadeform.sh --spend-limit 100
```

### Self-Destruct on Completion

The training script automatically deletes the instance when training completes successfully. This happens before the spend threshold, saving money.

### Cost Estimates

| Phase | GPU Config | Time | Cost |
|-------|-----------|------|------|
| Midtraining | 8×B200 | ~45 min | ~$25 |
| Midtraining | 8×H100 | ~60 min | ~$24 |
| SFT | 8×H100 | ~60 min | ~$24 |
| RL | 8×H100 | ~60 min | ~$24 |
| **Full (all)** | 8×H100 | ~3-4 hours | ~$80-100 |

### GPU Selection

The launch script shows available GPUs sorted by price:

```
=== Available B200 Instances (sorted by price) ===

  #  | Provider     | Region          | GPUs   | VRAM     | Price/hr   | NVLink
  ---+--------------+-----------------+--------+----------+------------+--------
  1  | hyperstack   | canada-1        | 8x     | 192GB    | $33.60     | Yes
  2  | datacrunch   | FIN-01          | 8x     | 192GB    | $35.84     | Yes
  3  | tensordock   | us-east         | 8x     | 192GB    | $38.40     | No

Select instance [1-3] or 'q' to quit:
```

## Initial Setup

### One-Time: Upload Checkpoints to HuggingFace

Before your first cloud training run, upload your checkpoints:

```bash
# Create HuggingFace repo (if needed)
uv run hf repo create lanbot-checkpoints --type model

# Upload base checkpoints
uv run hf upload gkobilansky/lanbot-checkpoints \
    my-checkpoints/base_checkpoints/d20 \
    base_checkpoints/d20

# Upload tokenizer
uv run hf upload gkobilansky/lanbot-checkpoints \
    my-checkpoints/tokenizer \
    tokenizer

# Upload training data
uv run hf upload gkobilansky/lanbot-checkpoints \
    ~/.cache/nanochat/identity_conversations.jsonl \
    identity_conversations.jsonl
```

### Verify Upload

```bash
# Check repo contents
uv run hf repo files gkobilansky/lanbot-checkpoints
```

Expected structure:
```
lanbot-checkpoints/
├── base_checkpoints/d20/
├── mid_checkpoints/d20/      # After midtraining
├── chatsft_checkpoints/d20/  # After SFT
├── chatrl_checkpoints/d20/   # After RL
├── tokenizer/
├── identity_conversations.jsonl
└── logs/
```

## Troubleshooting

### Instance Not Starting

```bash
# Check Shadeform dashboard
open https://platform.shadeform.ai/instances

# Verify API key
curl -s -X GET "https://api.shadeform.ai/v1/instances" \
    -H "X-API-KEY: $SHADEFORM_API_KEY" | python3 -m json.tool
```

### Training Failed

1. Check error logs on HuggingFace:
   ```bash
   uv run hf download gkobilansky/lanbot-checkpoints \
       --include "logs/*.log" \
       --local-dir ./logs
   ```

2. Look for `error-*.log` files with stack traces

### Checkpoints Not Uploading

- Verify HF_TOKEN has write permissions
- Check if HuggingFace repo exists
- Look for upload errors in training log

### Spend Threshold Hit Before Completion

Increase the spend limit:
```bash
./scripts/launch_shadeform.sh --phase sft --spend-limit 200
```

## Advanced Usage

### Non-Interactive Launch

For scripting or CI/CD:

```bash
./scripts/launch_shadeform.sh \
    --phase sft \
    --cloud hyperstack \
    --region canada-1 \
    --instance-type "8x_b200_pcie_192gb"
```

### Custom Run Names

```bash
./scripts/launch_shadeform.sh \
    --phase sft \
    --run-name "lanbot-v3-experiment-1"
```

### Different GPU Types

```bash
# Use H100 instead of B200
./scripts/launch_shadeform.sh --gpu H100

# Use A100 (cheaper but slower)
./scripts/launch_shadeform.sh --gpu A100

# Show all available configurations
./scripts/launch_shadeform.sh --all-configs
```
