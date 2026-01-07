# Training Pipeline

This document explains the full LLM training pipeline: what each phase does, why it matters, and what you get at the end.

## The Full Pipeline

```
┌─────────────────┐
│   Tokenizer     │  Convert text → numbers
└────────┬────────┘
         ↓
┌─────────────────┐
│  Base Training  │  Learn language patterns from raw text
└────────┬────────┘
         ↓
┌─────────────────┐
│  Midtraining    │  Teach conversation format + personality
└────────┬────────┘
         ↓
┌─────────────────┐
│      SFT        │  Learn to complete specific tasks
└────────┬────────┘
         ↓
┌─────────────────┐
│       RL        │  Improve through practice (math, reasoning)
└────────┬────────┘
         ↓
┌─────────────────┐
│  Agent Harness  │  Add tool use and agentic behavior
└─────────────────┘
```

## Phase 1: Tokenization

**Script:** `scripts/tok_train.py`

### What It Does

Converts text into numbers (tokens) that the model can process:

```
"Hello world" → [15496, 995]
```

### Why It Matters

- Determines vocabulary size (affects model size)
- Compression efficiency affects training cost
- Special tokens enable tool use (`<|python_start|>`, etc.)

### LanBot Status

Using pre-trained tokenizer with 65K vocabulary, trained on 2B characters.

## Phase 2: Base Training (Pretraining)

**Script:** `scripts/base_train.py`

### What It Does

Trains the model to predict the next token on massive amounts of raw text (FineWeb-Edu dataset).

### What It Learns

- Grammar and syntax
- Facts about the world
- Reasoning patterns
- Common knowledge

### What It Can't Do Yet

- Follow instructions
- Have conversations
- Use tools
- Know who it is

### LanBot Status

Using pre-trained d20 (561M parameters) base model:
- 21,400 training steps
- Validation bpb: 0.8138
- CORE score: 0.1974

## Phase 3: Midtraining

**Script:** `scripts/mid_train.py`

### What It Does

Teaches the base model to:
1. Follow the conversation format (user/assistant turns)
2. Adopt a specific identity (LanBot)
3. Handle basic tasks

### Training Data Mix

| Dataset | Size | Purpose |
|---------|------|---------|
| SmolTalk | 460K | General conversations |
| MMLU | 100K | Multiple choice Q&A |
| GSM8K | 8K | Math with calculator |
| Identity data | 2K (2× epochs) | Personality |
| SimpleSpelling | 200K | Letter spelling |
| SpellingBee | 80K | Letter counting |

### What Changes

Before midtraining:
```
Input: "What is 2+2?"
Output: "What is 3+3? What is 4+4? What is..."  # Just predicts more text
```

After midtraining:
```
Input: "What is 2+2?"
Output: "The answer is 4."  # Actually responds
```

### LanBot Status: COMPLETE

- 813 steps on 8×B200
- Validation bpb: 0.45
- Model correctly identifies as "LanBot, created by Gene Kobilansky"

## Phase 4: Supervised Fine-Tuning (SFT)

**Script:** `scripts/chat_sft.py`

### What It Does

Fine-tunes on high-quality task examples to improve:
- Instruction following
- Task completion
- Response quality

### Training Data Mix

| Dataset | Size | Purpose |
|---------|------|---------|
| ARC-Easy | 2.3K | Science questions |
| ARC-Challenge | 1.1K | Harder science |
| GSM8K | 8K | Math problems |
| SmolTalk | 10K | Conversations |
| Identity | 1K | Personality reinforcement |
| SimpleSpelling | 300 | Spelling |
| SpellingBee | 300 | Letter counting |

### What Changes

Before SFT:
```
User: "Write a haiku about programming"
Assistant: "Sure, here's a haiku... [rambles, doesn't complete]"
```

After SFT:
```
User: "Write a haiku about programming"
Assistant: "Code flows like water
  Bugs hide in the shadows deep
  Debug, test, repeat"
```

### Key Difference from Midtraining

- **Midtraining**: Teaches the format ("what does a conversation look like?")
- **SFT**: Teaches quality ("what does a GOOD response look like?")

### LanBot Status: NEXT STEP

Run with:
```bash
./scripts/launch_shadeform.sh --phase sft
```

## Phase 5: Reinforcement Learning (RL)

**Script:** `scripts/chat_rl.py`

### What It Does

Improves specific skills through practice and rewards:
1. Model generates multiple responses
2. Responses are scored (correct/incorrect for math)
3. Model learns from successful responses

### Current Focus

GSM8K math problems with calculator tool use:

```
Problem: "If John has 3 apples and buys 5 more, how many does he have?"

Model generates:
  Response A: "8 apples" ✓ (reward: +1)
  Response B: "7 apples" ✗ (reward: 0)
  Response C: "3+5=8, so 8 apples" ✓ (reward: +1)

Model learns to produce more responses like A and C.
```

### What Changes

Before RL:
- Math accuracy: ~5%
- Often skips calculator

After RL:
- Math accuracy: ~8-10%
- More consistent tool use

### LanBot Status: PENDING

Run after SFT completes:
```bash
./scripts/launch_shadeform.sh --phase rl
```

## Phase 6: Agent Harness (Future)

### What It Is

A wrapper around the model that enables:
- Multi-turn tool use
- Memory across conversations
- External API calls
- Autonomous task completion

### How It Works

```python
while not done:
    response = model.generate(context)

    if has_tool_call(response):
        result = execute_tool(response.tool_call)
        context.append(tool_result=result)
    else:
        done = response.is_final
```

### Current Tool Support

The model already supports a Python calculator:
```
<|python_start|> 2 + 2 <|python_end|> <|output_start|> 4 <|output_end|>
```

### Future Additions

- Web search
- File operations
- Code execution sandbox
- Memory/notes system

## Metrics Reference

### CORE Score

Measures base model capability on held-out text prediction.
- Higher is better
- GPT-2 level: ~0.28

### Bits Per Byte (bpb)

Compression efficiency on validation data.
- Lower is better
- Measures how well the model predicts text

### ChatCORE

Evaluation of conversational ability across multiple benchmarks.
- Higher is better

### Benchmark Scores

| Metric | What It Measures |
|--------|------------------|
| ARC-Challenge | Science reasoning |
| ARC-Easy | Basic science knowledge |
| GSM8K | Grade school math |
| HumanEval | Python coding |
| MMLU | General knowledge |
| SpellingBee | Letter counting |

## Cost Summary

| Phase | Time (8×H100) | Cost |
|-------|---------------|------|
| Tokenizer | Minutes | ~$1 |
| Base (d20) | 4 hours | ~$100 |
| Midtraining | 1 hour | ~$24 |
| SFT | 1 hour | ~$24 |
| RL | 1 hour | ~$24 |
| **Total** | ~7 hours | ~$175 |

## Skipping Base Training

For production models, you can skip base training entirely by starting from open-source weights:

| Model | Params | License | Use Case |
|-------|--------|---------|----------|
| Qwen 2.5 | 0.5B-72B | Apache 2.0 | General purpose |
| Llama 3.2 | 1B-90B | Meta License | Strong reasoning |
| SmolLM2 | 135M-1.7B | Apache 2.0 | Tiny but capable |

Apply midtraining + SFT + RL to these and get a much smarter base with your personality.

## Next Steps

Based on current LanBot status (midtraining complete):

1. **Run SFT** → Teaches task completion
   ```bash
   ./scripts/launch_shadeform.sh --phase sft
   ```

2. **Run RL** → Improves math accuracy
   ```bash
   ./scripts/launch_shadeform.sh --phase rl
   ```

3. **Build Agent Harness** → Adds tool orchestration
   (See gameplan Phase 5)
