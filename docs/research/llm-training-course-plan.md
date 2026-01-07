# Course Plan: Train Your Own LLM

## Overview

Build an interactive course (like [ccforpms.com](https://ccforpms.com/)) that teaches LLM training from scratch, fine-tuning, and building agent harnesses — all using the nanochat codebase.

---

## Why This Could Work Well

1. **Complete pipeline already exists** — tokenization → pretraining → midtraining → SFT → RL → inference → tool use
2. **ccforpms.com pattern fits perfectly** — interactive slash commands that guide through each stage
3. **Real hands-on experience** — students actually train models, not just read about it
4. **Clear progression** — from "what is a token" to "I built an agent"

---

## Proposed Course Structure

```
Module 0: Setup & Orientation
├── /start         → Install deps, understand the repo
├── /quick-tour    → 5-min overview of the pipeline
└── /concepts      → Tokens, embeddings, attention (visual)

Module 1: Tokenization (The Foundation)
├── /tok-explore   → What is BPE? Hands-on with tiktoken
├── /tok-train     → Train a custom tokenizer
└── /tok-eval      → Measure compression, compare to GPT-4

Module 2: Base Training (The Hard Part)
├── /pretrain-data → Understand FineWeb-Edu, data loading
├── /pretrain-arch → Walk through gpt.py architecture
├── /pretrain-run  → Train d12 (tiny) on CPU/MPS locally
└── /pretrain-eval → CORE metric, loss curves

Module 3: Personality Injection
├── /synth-data    → Generate identity conversations
├── /mid-train     → Mix identity into training
└── /mid-eval      → Test: "Who are you?"

Module 4: Task Training (SFT + RL)
├── /sft-intro     → What is supervised fine-tuning?
├── /sft-run       → Fine-tune on task mixtures
├── /rl-intro      → Reinforcement learning basics
└── /rl-run        → Train on math with rewards

Module 5: Open-Source Shortcut
├── /oss-models    → Qwen, Llama, Mistral comparison
├── /oss-adapt     → Skip base training, start from weights
└── /oss-train     → Apply your personality to a real model

Module 6: Building the Agent
├── /tools-intro   → How tool use works (special tokens)
├── /tools-add     → Add a new tool (web search example)
├── /agent-loop    → The generate → tool → result cycle
└── /agent-build   → Complete agent harness

Module 7: Production
├── /cloud-train   → Shadeform scripts walkthrough
├── /scale-up      → d20 → d26 → d32 cost/benefit
└── /deploy        → Serving your model
```

---

## Leveraging the Current Codebase

### What's Already Great (Keep As-Is)

| Component | Why It Works |
|-----------|--------------|
| `speedrun.sh` | Shows the full pipeline in one script |
| `scripts/tok_train.py` | Clean, well-structured tokenizer training |
| `nanochat/gpt.py` | Modern architecture with good comments |
| `dev/gen_synthetic_data.py` | Already supports identity/tool_use/reasoning |
| `tasks/*.py` | Clean task abstraction for evaluation |
| `scripts/chat_cli.py` | Immediate feedback loop for testing |

### What Needs Expansion

1. **Tiny training mode for local learning**
   ```python
   # Add to scripts/base_train.py
   # d4 or d8 model that trains in 5-10 minutes on CPU/MPS
   # Students need fast iteration loops
   ```

2. **Interactive lesson system** (new `/lessons` directory)
   ```
   lessons/
   ├── 01-tokenization/
   │   ├── lesson.md        # Lesson content
   │   ├── exercise.py      # Guided exercise
   │   └── solution.py      # Answer key
   ├── commands.md          # Slash command definitions
   └── progress.json        # Track completion
   ```

3. **Visual explanations**
   - ASCII diagrams of attention, embeddings, token flow
   - Could generate SVGs or use the web UI for visualization

4. **Step-by-step checkpoints**
   - Pre-trained checkpoints at each stage so students can skip ahead
   - "If tokenizer training failed, download this and continue"

### What to Simplify

1. **`nanochat/gpt.py`** — Currently 500+ lines
   - Create `gpt_simple.py` (200 lines) for teaching
   - Remove GQA, advanced optimizations for the learning version
   - Keep the production version separate

2. **`dev/gen_synthetic_data.py`** — 39KB is intimidating
   - Extract a simple `simple_synth.py` (~100 lines) for the lesson
   - Full version available for advanced users

3. **Training scripts** — Remove cloud-specific code for local lessons
   - `scripts/base_train.py` → `lessons/base_train_local.py`
   - No DDP, no wandb, just single-GPU/CPU training

4. **Configuration** — Too many knobs
   - Create preset configs: `config_tiny.py`, `config_small.py`, `config_full.py`
   - Students pick one, not 20 hyperparameters

---

## Implementation Approach

### Phase 1: Course Skeleton
1. Create `/lessons` directory structure
2. Write slash commands that read lesson content and present it
3. Build progress tracking (what lessons completed)

### Phase 2: Tiny Training Mode
1. Add `d4`/`d8` model configs for fast local training
2. Create 5-minute exercises that actually train something
3. Pre-generate checkpoints at each stage

### Phase 3: Lesson Content
1. Port gameplan phases into interactive lessons
2. Add inline exercises ("Now run this command and observe...")
3. Create "checkpoint" moments where students test understanding

### Phase 4: Open-Source Module
1. Add scripts to download/convert Qwen or SmolLM2
2. Show how to apply midtraining to existing weights
3. Compare scratch-trained vs. fine-tuned

### Phase 5: Agent Module
1. Expand `nanochat/engine.py` tool system
2. Create simple agent harness example
3. Let students add their own tools

---

## Key Differentiator from ccforpms.com

| ccforpms.com | This Course |
|--------------|-------------|
| Uses Claude (API) | Trains your own model |
| Consumer of AI | Builder of AI |
| Hours to complete | Days to weeks (deeper) |
| No GPU required | GPU optional but recommended |
| PM-focused skills | Engineering-focused skills |

---

## Open Questions

1. **Target audience?** ML engineers vs. curious developers vs. complete beginners?
2. **GPU requirement?** Can everything work on CPU/MPS, or is cloud mandatory?
3. **Paid vs. free?** Affects how much polish is needed
4. **Timeline?** This is a substantial project — prioritization matters

---

## Recommended First Steps

1. **Create a `d4` (tiny) model config** that trains in 5 min on M-series Mac
2. **Write 3 pilot lessons** (tokenization module) to test the format
3. **Build the slash command infrastructure** using Claude Code's custom commands
4. **Get one person through Module 1** and iterate based on feedback

---

## Reference

- [ccforpms.com](https://ccforpms.com/) — Interactive course format inspiration
- [agentic-llm-gameplan.md](./agentic-llm-gameplan.md) — Current training progress and phases
- [nanochat](https://github.com/karpathy/nanochat) — Original codebase
