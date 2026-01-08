---
date: 2026-01-08T19:06:51Z
researcher: Claude
git_commit: ff3be26fc46ef49d439866b70d2ecef7c2dca4f0
branch: master
repository: lansky-chat
topic: "Codebase Simplification Analysis vs Plans"
tags: [research, codebase, simplification, nanochat, training-pipeline]
status: complete
last_updated: 2026-01-08
last_updated_by: Claude
---

# Research: Codebase Simplification Analysis vs Plans

**Date**: 2026-01-08T19:06:51Z
**Researcher**: Claude
**Git Commit**: ff3be26fc46ef49d439866b70d2ecef7c2dca4f0
**Branch**: master
**Repository**: lansky-chat

## Research Question

What can be removed and simplified in the lansky-chat codebase to more closely match the plans in `docs/research/`?

## Summary

The codebase contains **44 source files** across multiple directories. Based on analysis against the two plan documents (`agentic-llm-gameplan.md` and `llm-training-course-plan.md`), approximately **8-12 files** can potentially be removed or simplified without affecting the core training pipeline.

The plans explicitly identify what's essential for the LanBot training workflow and what could be simplified for educational purposes.

## Detailed Findings

### Files Explicitly Mentioned as Essential in Plans

From `docs/research/agentic-llm-gameplan.md` (lines 501-510):

| File | Purpose | Status |
|------|---------|--------|
| `nanochat/gpt.py` | Model architecture | Essential |
| `nanochat/engine.py` | Inference + tool use | Essential |
| `nanochat/dataset.py` | Data loading | Essential |
| `scripts/mid_train.py` | Personality training | Essential |
| `dev/gen_synthetic_data.py` | Synthetic data generation | Essential |
| `tasks/customjson.py` | Custom training data | Essential |

Training pipeline scripts (lines 258-264):
| File | Purpose | Status |
|------|---------|--------|
| `scripts/tok_train.py` | Tokenizer training | Essential |
| `scripts/base_train.py` | Pretraining | Essential |
| `scripts/chat_sft.py` | Supervised fine-tuning | Essential |
| `scripts/chat_rl.py` | Reinforcement learning | Essential |
| `scripts/chat_cli.py` | Inference/testing | Essential |

Shadeform automation scripts (lines 59-66):
| File | Purpose | Status |
|------|---------|--------|
| `scripts/launch_shadeform.sh` | Launch cloud GPU | Essential |
| `scripts/shadeform_train.sh` | Cloud training execution | Essential |
| `scripts/check_training_status.sh` | Monitor progress | Essential |

### Files Marked for Simplification in Plans

From `docs/research/llm-training-course-plan.md` (lines 108-127):

| File | Current State | Suggested Simplification |
|------|---------------|-------------------------|
| `nanochat/gpt.py` | 500+ lines | Create `gpt_simple.py` (~200 lines) for teaching |
| `dev/gen_synthetic_data.py` | 39KB (1,018 lines) | Extract `simple_synth.py` (~100 lines) |
| Training scripts | Include cloud/DDP code | Remove cloud-specific code for local lessons |
| Configuration | Many knobs | Create preset configs: `config_tiny.py`, `config_small.py` |

### Candidates for Removal

#### Safe to Remove (Not Used at Runtime)

| File | Reason |
|------|--------|
| `dev/generate_logo.html` | One-time logo generation utility |
| `dev/nanochat.png` | Static logo asset |
| `dev/repackage_data_reference.py` | Reference documentation only (line 13-14 states "not used during project runtime") |
| `nanochat/logo.svg` | Static logo asset |
| `dev/__pycache__/*.pyc` | Compiled bytecode (auto-generated) |

#### Potentially Removable (Feature-Dependent)

| File | Reason | Consideration |
|------|--------|---------------|
| `scripts/chat_web.py` | Web UI server | Only needed if serving web interface; `chat_cli.py` is sufficient for testing |
| `nanochat/ui.html` | Web chat interface | Paired with `chat_web.py` |
| `dev/runcpu.sh` | Local CPU/MPS demo | Replaced by Shadeform cloud training per gameplan line 100 |
| `run1000.sh` | $1000 tier training | Only needed for d32 model; `speedrun.sh` covers d20 |

#### Evaluation Scripts (Optional)

These scripts are used for metrics but not essential to the training pipeline:

| File | Purpose | Status |
|------|---------|--------|
| `scripts/base_eval.py` | CORE benchmark evaluation | Used but optional |
| `scripts/base_loss.py` | Validation loss evaluation | Used but optional |
| `scripts/chat_eval.py` | Chat model benchmarks | Used but optional |
| `scripts/tok_eval.py` | Tokenizer compression comparison | Used but optional |

### Infrastructure Files (Essential but Not Mentioned)

These files are not explicitly mentioned in plans but are **required infrastructure**:

| File | Why Essential |
|------|---------------|
| `nanochat/tokenizer.py` | Required for all tokenization |
| `nanochat/checkpoint_manager.py` | Required for model saving/loading |
| `nanochat/common.py` | Shared utilities used throughout |
| `nanochat/dataloader.py` | Required for data streaming |
| `nanochat/muon.py` | Custom optimizer used in training |
| `nanochat/adamw.py` | Distributed optimizer for DDP |
| `nanochat/configurator.py` | CLI config parsing |
| `nanochat/report.py` | Training report generation |
| `nanochat/loss_eval.py` | Bits-per-byte metric calculation |
| `nanochat/core_eval.py` | CORE benchmark logic |
| `nanochat/execution.py` | Sandboxed code execution for HumanEval |

### Task Files (Used in Training Mixtures)

All task files in `tasks/` are used by the training scripts even if not mentioned:

| File | Used In |
|------|---------|
| `tasks/common.py` | Base class for all tasks |
| `tasks/customjson.py` | mid_train.py, chat_sft.py (lines 102, 88) |
| `tasks/smoltalk.py` | mid_train.py (line 98) |
| `tasks/mmlu.py` | mid_train.py (line 99) |
| `tasks/gsm8k.py` | mid_train.py, chat_rl.py (line 100) |
| `tasks/spellingbee.py` | mid_train.py (lines 104-105) |
| `tasks/arc.py` | chat_sft.py (lines 84-85) |
| `tasks/humaneval.py` | chat_eval.py |

## Code References

### Plan Documents
- `docs/research/agentic-llm-gameplan.md:501-510` - Key files to study
- `docs/research/llm-training-course-plan.md:108-127` - What to simplify
- `docs/research/llm-training-course-plan.md:69-79` - What's already great

### Training Pipeline Entry Points
- `speedrun.sh` - Full pipeline orchestration
- `scripts/launch_shadeform.sh:357-426` - Cloud instance creation
- `scripts/shadeform_train.sh:461-486` - Multi-phase training execution

### Files Marked as "Reference Only"
- `dev/repackage_data_reference.py:13-14` - Explicit "not used during project runtime" comment

## Architecture Documentation

### Current Directory Structure

```
lansky-chat/
├── scripts/          # 14 files (10 essential, 4 evaluation)
├── nanochat/         # 17 files (all infrastructure, most essential)
├── tasks/            # 8 files (all used in training mixtures)
├── dev/              # 6 files (1 essential, 5 removable)
├── docs/             # Research plans and usage docs
├── tests/            # Test files
├── rustbpe/          # Rust tokenizer extension
└── Root files        # speedrun.sh, pyproject.toml, etc.
```

### Minimal Essential File Set

For the LanBot training workflow as described in the gameplan:

**Core Training (must keep):**
- `nanochat/gpt.py`, `engine.py`, `dataset.py`, `tokenizer.py`
- `nanochat/checkpoint_manager.py`, `common.py`, `dataloader.py`
- `nanochat/muon.py`, `adamw.py`, `configurator.py`
- `scripts/tok_train.py`, `base_train.py`, `mid_train.py`, `chat_sft.py`, `chat_rl.py`
- `scripts/chat_cli.py`
- `scripts/launch_shadeform.sh`, `shadeform_train.sh`, `check_training_status.sh`
- `dev/gen_synthetic_data.py`
- All `tasks/*.py` files
- `speedrun.sh`, `pyproject.toml`

**Evaluation (used but optional):**
- `scripts/base_eval.py`, `base_loss.py`, `chat_eval.py`, `tok_eval.py`
- `nanochat/core_eval.py`, `loss_eval.py`, `report.py`
- `nanochat/execution.py` (only for HumanEval)

**Web UI (can remove if CLI-only):**
- `scripts/chat_web.py`, `nanochat/ui.html`

**Safe to Remove:**
- `dev/generate_logo.html`, `dev/nanochat.png`, `dev/repackage_data_reference.py`
- `nanochat/logo.svg`
- `dev/runcpu.sh` (replaced by cloud training)
- `run1000.sh` (if only using d20 model)

## Related Research

No prior research documents found in `thoughts/shared/research/`.

## Open Questions

1. **Should evaluation scripts be kept?** They provide valuable metrics but add complexity. The gameplan uses them but they're not strictly required for training.

2. **Is the web UI needed?** The gameplan doesn't mention `chat_web.py` or `ui.html`, suggesting CLI is sufficient for the current workflow.

3. **Should `run1000.sh` be kept?** The gameplan focuses on d20 model training. The $1000 tier script may be premature until Phase 3+ is reached.

4. **What about the course plan suggestions?** The `llm-training-course-plan.md` suggests creating simplified versions of `gpt.py` and `gen_synthetic_data.py` for teaching - should these be created as separate files, or should the originals be simplified?
