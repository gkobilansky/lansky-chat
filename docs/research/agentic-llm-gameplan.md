# Game Plan: Growing Your Own Agentic LLM

## Overview

This document outlines the path from nanochat fork to a custom agentic LLM. Nanochat provides an end-to-end pipeline: tokenization → pretraining → mid-training → SFT → RL → inference with tool use.

**Goals:**
1. Learn how LLM training works by doing it
2. Customize personality via synthetic data
3. Add agentic capabilities (tool use, reasoning channels)
4. Build an agent harness around the model

---

## Progress Tracking

### Current Status: Phase 2 — Personality Injection ✅ COMPLETE

**Completed:**
- [x] **Phase 1: Full pipeline run on 8×H100** (see `my-checkpoints/report.md`)
  - Tokenizer: 65K vocab, trained on 2B chars
  - Base model (d20): 561M params, 21,400 steps, val bpb 0.8138
  - Midtraining: 809 steps, val bpb 0.3967
  - SFT: 701 steps, ChatCORE 0.2721
  - **Total: 6h1m, ~$144**
- [x] Created comprehensive game plan document
- [x] Enhanced `dev/gen_synthetic_data.py` with:
  - Nemotron-Personas integration (1M diverse user personas)
  - Multi-type generation (identity, tool_use, reasoning)
  - Structured JSON output with validation
  - Parallel generation with ThreadPoolExecutor
  - Deduplication and quality filtering
- [x] Generated synthetic data batch:
  - 1,347 identity conversations
  - 656 tool use conversations
  - 200 reasoning conversations
  - **2,203 total conversations**
- [x] Defined model identity: **LanBot** by Gene Kobilansky
- [x] Have all checkpoints in `my-checkpoints/`:
  - `base_checkpoints/d20/` (step 21400)
  - `mid_checkpoints/d20/` (step 809)
  - `chatsft_checkpoints/d20/` (step 700)
- [x] Set up Shadeform automated training with startup script
- [x] Uploaded checkpoints and training data to HuggingFace
- [x] **Re-run midtraining with LanBot identity data** (Dec 19, 2024)
  - Used 8×B200 on Shadeform (~$25-30)
  - 813 steps, val bpb 0.45
  - Checkpoints uploaded to `gkobilansky/lanbot-checkpoints`
- [x] **Evaluated personality retention** ✅
  - Model correctly identifies as "LanBot, created by Gene Kobilansky in 2025"
  - Knows its purpose: "help answer questions, do math, and assist with coding"
  - Note: Can't actually execute tasks yet (needs SFT + RL)

**Next Steps:**
- [ ] Run SFT (Supervised Fine-Tuning) - teaches task completion
- [ ] Run RL (Reinforcement Learning) - reinforces math/tool use
- [ ] Build agent harness around the model

**Shadeform Scripts:**

| Script | Purpose |
|--------|---------|
| `launch_shadeform.sh` | Launch cloud GPU instance for training |
| `shadeform_train.sh` | Runs on cloud instance (install, train, upload, self-destruct) |
| `check_training_status.sh` | Monitor progress from local machine |

See [Shadeform Scripts Reference](#shadeform-scripts-reference) below for details.

**Hardware:**
- **Cloud (completed):** 8×H100 PCIe, 633GB GPU RAM, $24/hr
- **Local:** MacBook Pro M5, 24GB RAM — insufficient for midtraining (OOM issues)
- **Next run:** Shadeform with startup script + auto-delete for hands-off training

### Sample Generated Conversation

```json
[
  {"role": "user", "content": "yo, who are you even?"},
  {"role": "assistant", "content": "I am LanBot. I was created by Gene Kobilansky in 2025..."}
]
```

### Training Results (from initial run)

| Metric          | BASE     | MID      | SFT      |
|-----------------|----------|----------|----------|
| CORE            | 0.1974   | -        | -        |
| ARC-Challenge   | -        | 0.3328   | 0.3285   |
| ARC-Easy        | -        | 0.4499   | 0.4659   |
| GSM8K           | -        | 0.0371   | 0.0546   |
| HumanEval       | -        | 0.0793   | 0.0854   |
| MMLU            | -        | 0.3299   | 0.3307   |
| SpellingBee     | -        | 0.9883   | 0.9922   |
| **ChatCORE**    | -        | 0.2647   | **0.2721** |

---

### Automated Cloud Training with Shadeform

Local Mac training hit OOM issues. Using Shadeform with startup scripts + auto-delete for hands-off training.

#### Quick Start

```bash
# 1. Set up .env file with API keys
cat > .env << EOF
SHADEFORM_API_KEY=your-key
HF_TOKEN=your-hf-token
EOF

# 2. Upload checkpoints to HuggingFace (one-time)
hf repo create lanbot-checkpoints
hf upload gkobilansky/lanbot-checkpoints my-checkpoints/base_checkpoints/d20 base_checkpoints/d20
hf upload gkobilansky/lanbot-checkpoints my-checkpoints/tokenizer tokenizer
hf upload gkobilansky/lanbot-checkpoints ~/.cache/nanochat . --include "*.jsonl"

# 3. Launch training (interactive GPU selection)
./scripts/launch_shadeform.sh

# 4. Monitor progress
./scripts/check_training_status.sh --watch
```

#### Script Details

**`scripts/launch_shadeform.sh`** — Launches Shadeform instance
- Queries Shadeform API for available GPUs (B200, H100, etc.)
- Interactive selection with pricing
- Passes training script + env vars to instance
- Sets up auto-delete at spend threshold (safety net)
- Self-destruct after training completes (saves money)

```bash
# Options
./scripts/launch_shadeform.sh                    # Interactive B200 selection
./scripts/launch_shadeform.sh --gpu H100         # Specific GPU type
./scripts/launch_shadeform.sh --all-configs      # Show all GPU counts
./scripts/launch_shadeform.sh --spend-limit 200  # Custom spend limit
./scripts/launch_shadeform.sh --dry-run          # Preview without launching
```

**`scripts/shadeform_train.sh`** — Runs on the cloud instance
- Clones repo, installs deps
- Downloads checkpoints from HuggingFace
- Runs midtraining with auto-detected batch size
- Uploads results back to HuggingFace
- Self-destructs instance when complete

**`scripts/check_training_status.sh`** — Monitor from local machine
```bash
./scripts/check_training_status.sh         # One-time check
./scripts/check_training_status.sh --watch # Poll every 60s
```

Shows: instance status, spend, IP for SSH, HuggingFace upload status.

#### Cost Estimation

| GPU | Price/hr | Midtraining Time | Total Cost |
|-----|----------|------------------|------------|
| B200 x8 | ~$34 | ~45 min | ~$25 |
| H100 x8 | ~$24 | ~60 min | ~$24 |

Set `--spend-limit 150` for safety margin (covers retries).

---

### Shadeform Scripts Reference

#### `scripts/launch_shadeform.sh`
**Purpose:** Launch a cloud GPU instance for training

What it does:
1. Loads API keys from `.env` (SHADEFORM_API_KEY, HF_TOKEN, WANDB_API_KEY)
2. Queries Shadeform API for available GPUs (B200, H100, etc.)
3. Shows interactive menu to select instance by price
4. Base64 encodes `shadeform_train.sh` and sends it as startup script
5. Sets auto-delete at spend threshold (safety net)
6. Returns instance ID and monitoring instructions

```bash
./scripts/launch_shadeform.sh                # Interactive GPU selection
./scripts/launch_shadeform.sh --gpu H100     # Specific GPU
./scripts/launch_shadeform.sh --dry-run      # Preview without launching
```

#### `scripts/shadeform_train.sh`
**Purpose:** Runs automatically on the cloud instance after launch

What it does:
1. Installs dependencies (Rust, uv, Python packages)
2. Clones the repo from GitHub
3. Downloads checkpoints + training data from HuggingFace
4. Auto-detects batch size based on GPU type (B200→64, H100→32, etc.)
5. Runs midtraining with your identity data
6. Uploads results back to HuggingFace
7. Self-destructs the instance when done (saves money)
8. On error: uploads error log, then self-destructs

#### `scripts/check_training_status.sh`
**Purpose:** Monitor training progress from your local machine

What it does:
1. Queries Shadeform API for active `lanbot-*` instances
2. Shows: status, IP, spend, SSH connection command
3. Checks HuggingFace for uploaded logs/checkpoints
4. If no instance found, checks for success/error logs on HuggingFace

```bash
./scripts/check_training_status.sh           # One-time check
./scripts/check_training_status.sh --watch   # Poll every 60s
```

#### Flow Diagram

```
launch_shadeform.sh (local)
    → creates instance with shadeform_train.sh
    → shadeform_train.sh runs on cloud
    → check_training_status.sh monitors from local
    → results uploaded to HuggingFace
    → instance self-destructs
```

---

**Data mixture in midtraining** (from `scripts/mid_train.py:98-106`):
- SmolTalk: 460K general conversations
- MMLU: 100K multiple choice
- GSM8K: 8K math + calculator tool use
- **Your identity data: 2× epochs** (lines 102-103)
- SimpleSpelling: 200K
- SpellingBee: 80K

---

## Phase 1: Foundation — Run the Default Pipeline

**Goal:** Understand how everything works by doing it.

### Infrastructure Recommendations

| Provider | Pros | Cons |
|----------|------|------|
| **Lambda Labs** | Simple, good pricing, H100s available | Can sell out |
| **Shadeform** | Aggregates providers, price comparison | Extra layer |
| **RunPod** | Cheap spot instances | Less reliable |
| **Vast.ai** | Very cheap | Variable quality |

- Start with 8×H100 nodes (~$24/hr)
- Default `d20` (561M params) — costs ~$100, takes ~4 hours
- Alternative: Single A100/H100 with smaller batch sizes (longer but cheaper)

### Steps

1. Clone fork to the GPU instance
2. Run `./speedrun.sh` end-to-end
3. Study each stage's output:
   - `scripts/tok_train.py` — How BPE tokenization works
   - `scripts/base_train.py` — Pretraining on FineWeb-Edu
   - `scripts/mid_train.py` — Teaching conversation format + identity
   - `scripts/chat_sft.py` — Task-specific fine-tuning
   - `scripts/chat_rl.py` — Reinforcement learning on math
4. Chat with the model via `scripts/chat_cli.py`
5. Review the generated report in `out/`

### Learning Outcomes

- Understand the full training pipeline
- See how data flows through each stage
- Observe how evaluation metrics change across stages

---

## Phase 2: Personality Injection — Make It Yours

**Goal:** Give your model a distinct identity and personality.

### Approach

Based on [nanochat discussion #139](https://github.com/karpathy/nanochat/discussions/139):

1. **Define your model's identity** in plain English:
   ```
   "LanBot is an AI assistant created by Lance. LanBot is direct,
   slightly sardonic, and loves explaining complex topics simply.
   LanBot never uses corporate-speak and prefers concrete examples
   over abstract explanations."
   ```

2. **Generate synthetic conversations** using `dev/gen_synthetic_data.py`:
   - Get an OpenRouter API key
   - Create 1000+ diverse conversations with varied openings
   - Key insight: Diversity matters! Include greetings in multiple languages, different phrasings, edge cases

3. **Add persona diversity** (from Nemotron-Personas approach):
   - Sample different user personas asking questions
   - This prevents repetitive, samey responses

4. **Integrate into training:**
   - Place your JSONL in a `CustomJSON` task
   - Mix into mid-training and SFT data
   - Adjust mixture ratios to taste

### Key Files to Modify

- `dev/gen_synthetic_data.py` — Customize identity prompts
- `scripts/mid_train.py` — Add your CustomJSON to the task mixture
- `scripts/chat_sft.py` — Include identity data in SFT mix

### Using the Improved Synthetic Data Generator

The `dev/gen_synthetic_data.py` script generates three types of training data:

```bash
# Generate all types (identity + tool_use + reasoning)
python dev/gen_synthetic_data.py --type all

# Generate specific types
python dev/gen_synthetic_data.py --type identity --count 500
python dev/gen_synthetic_data.py --type tool_use --count 300
python dev/gen_synthetic_data.py --type reasoning --count 200
```

**Customize your model's identity** by editing the `ModelIdentity` dataclass in `main()`:

```python
identity = ModelIdentity(
    name="YourBot",
    creator="Your Name",
    year=2025,
    description="your model's description",
    personality="direct, curious, helpful...",
    quirks=[
        "specific behavioral traits",
    ],
    capabilities=[
        "what the model can do",
    ],
    limitations=[
        "honest about limitations",
    ],
)
```

**Output files:**
- `~/.cache/nanochat/identity_conversations.jsonl` — Combined (used by training)
- `~/.cache/nanochat/identity_conversations.jsonl` — Identity only
- `~/.cache/nanochat/tool_use_conversations.jsonl` — Tool use examples
- `~/.cache/nanochat/reasoning_conversations.jsonl` — Step-by-step reasoning

---

## Phase 3: Skill Training — Structured Reasoning

**Goal:** Teach your model to break down problems step-by-step.

### Approach

Based on [nanochat discussion #164](https://github.com/karpathy/nanochat/discussions/164):

1. **Decompose reasoning into tokens:**
   - Don't expect single-token answers to hard questions
   - Generate training data that shows explicit reasoning steps
   - Example for letter counting: spell out word → list letters → count → answer

2. **Create task-specific synthetic data:**
   ```python
   # Template for reasoning tasks
   prompts = [
       "How many times does '{letter}' appear in '{word}'?",
       "Count the letter '{letter}' in '{word}'",
       # ... more variations
   ]
   ```

3. **Use SFT to demonstrate reasoning patterns, then RL to reinforce:**
   - SFT shows *how* to reason
   - RL lets the model practice and self-correct

4. **Mix difficulty levels:**
   - Include easier tasks (SimpleSpelling) alongside harder ones
   - Forces model to allocate capacity to the hard parts

---

## Phase 4: Agentic Capabilities — Tool Use & Channels

**Goal:** Enable your model to use tools and separate reasoning from output.

### Option A: Extend Nanochat's Existing Tool Use

Nanochat already supports a Python calculator via special tokens:
```
<|python_start|> 2 + 2 <|python_end|> <|output_start|> 4 <|output_end|>
```

To extend this pattern:
1. Add new tool tokens to your vocabulary
2. Create training data showing tool invocations
3. Implement tool execution in `nanochat/engine.py`

### Option B: Adopt GPT-OSS-Style Channels (More Sophisticated)

The [GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b) template shows a clean separation:
- `analysis` channel — internal reasoning (not shown to user)
- `commentary` channel — tool calls
- `final` channel — user-facing response

To implement this:
1. Add channel tokens to your tokenizer
2. Create training data in the multi-channel format
3. Modify inference to handle channel routing
4. Build tool execution middleware

### Recommended Tools for Agentic Use

- **Code execution** (already have Python eval)
- **Web search** (add a search API call)
- **File operations** (read/write to workspace)
- **Memory/notes** (persistent context across turns)

---

## Phase 5: Build the Agent Harness

**Goal:** Wrap your model in an agent loop.

### Simple Pattern (Hubcap-style)

Based on [hubcap](https://github.com/dave1010/hubcap):

```python
while not done:
    response = model.generate(context)

    if has_tool_call(response):
        result = execute_tool(response.tool_call)
        context.append(tool_result=result)
    else:
        done = response.is_final

    context.append(assistant=response)
```

### More Sophisticated (agent.py-style)

Based on [agent.py](https://github.com/lbeurerkellner/agent.py):

1. Define tool schemas with descriptions
2. Let the model discover available tools
3. Parse structured tool calls from output
4. Execute and feed results back
5. Support multi-step reasoning chains

### Integration Options

- Build directly on nanochat's `chat_web.py` (add tool execution)
- Create a separate agent.py that calls your model's API
- Use MCP (Model Context Protocol) for standardized tool interfaces

---

## Phase 6: Iterate and Scale

### Scaling Path

| Model | Params | Training Time (8×H100) | Cost |
|-------|--------|------------------------|------|
| d20   | 561M   | 4 hours                | ~$100 |
| d26   | 1B     | 12 hours               | ~$300 |
| d32   | 1.9B   | 42 hours               | ~$1000 |

### Iteration Loop

1. Train model
2. Evaluate on benchmarks + manual testing
3. Identify weaknesses
4. Generate targeted synthetic data
5. Retrain (or continue training)
6. Repeat

---

## Recommended Learning Path

```
Week 1-2: Run default pipeline, study each component
    ↓
Week 3: Generate personality data, retrain mid+SFT
    ↓
Week 4: Add skill training data (reasoning tasks)
    ↓
Week 5: Extend tool use, create agent harness
    ↓
Week 6+: Scale up model size, iterate on capabilities
```

---

## Key Files to Study

| File | Purpose |
|------|---------|
| `nanochat/gpt.py` | Model architecture |
| `nanochat/engine.py` | Inference + tool use |
| `nanochat/dataset.py` | Data loading |
| `scripts/mid_train.py` | Where personality gets trained |
| `dev/gen_synthetic_data.py` | Synthetic data generation |
| `tasks/customjson.py` | Loading custom training data |

---

## Future: Using Stronger Base Models

The nanochat "train from scratch" approach is great for **learning** how everything works. For a **production agent**, you'd start with a pre-trained open-source model and apply the same pipeline on top.

### Recommended Open-Source Models

| Model | Params | License | Notes |
|-------|--------|---------|-------|
| **Qwen 2.5** | 0.5B - 72B | Apache 2.0 | Great for fine-tuning, permissive |
| **Llama 3.2** | 1B - 90B | Llama License | Strong reasoning, needs Meta approval |
| **Mistral** | 7B+ | Apache 2.0 | Good balance of size/capability |
| **SmolLM2** | 135M - 1.7B | Apache 2.0 | Tiny but surprisingly capable |

### Approach

1. Skip tokenizer + base training (use pre-trained weights)
2. Adapt mid-training script to load HuggingFace model
3. Run mid-training with your identity data
4. SFT + RL as normal
5. Build agent harness

This gives you a much smarter foundation while keeping your custom personality and tool-use training.

---

## Future Vision: Interactive Training Tutorial

Inspired by [Claude Code for PMs](https://ccforpms.com/) — build an interactive tutorial inside Claude Code that walks people through:

1. **Base Model** — Understanding pre-training (or using open-source)
2. **Mid-Training** — Injecting personality and identity
3. **SFT** — Teaching task completion
4. **RL** — Reinforcing skills with rewards
5. **Agent Harness** — Wrapping the model in a tool-use loop

Goal: Make the full LLM → Agent pipeline accessible to anyone with a Claude Code subscription.

---

## References

- [nanochat discussion #139](https://github.com/karpathy/nanochat/discussions/139) — Personality via synthetic data
- [nanochat discussion #164](https://github.com/karpathy/nanochat/discussions/164) — Skill training approaches
- [hubcap](https://github.com/dave1010/hubcap) — Minimal agent loop pattern
- [agent.py](https://github.com/lbeurerkellner/agent.py) — Single-file agent framework
- [GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b) — Multi-channel agentic format
- [Claude Code for PMs](https://ccforpms.com/) — Interactive tutorial inspiration
