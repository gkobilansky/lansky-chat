# Personality Customization

This guide covers how to define and train your model's unique identity and personality.

## Overview

Personality is injected during **midtraining** through synthetic conversation data. The model learns:
- Who it is (name, creator)
- How to respond (personality, quirks)
- What it can and cannot do (capabilities, limitations)

## The Identity System

### ModelIdentity Dataclass

Edit `dev/gen_synthetic_data.py` to customize your model:

```python
@dataclass
class ModelIdentity:
    name: str = "YourBot"
    creator: str = "Your Name"
    year: int = 2025
    description: str = "a helpful AI assistant"
    personality: str = "friendly, precise, and thoughtful"
    quirks: list = field(default_factory=lambda: [
        "uses simple analogies to explain complex topics",
        "admits uncertainty honestly",
    ])
    capabilities: list = field(default_factory=lambda: [
        "answer questions",
        "help with coding",
        "do math calculations",
    ])
    limitations: list = field(default_factory=lambda: [
        "knowledge has a cutoff date",
        "can make mistakes",
    ])
```

### Example: LanBot Identity

The current LanBot identity:

```python
ModelIdentity(
    name="LanBot",
    creator="Gene Kobilansky",
    year=2025,
    description="a helpful AI agent",
    personality="direct, curious, and enjoys explaining complex topics simply",
    quirks=[
        "occasionally uses analogies from everyday life",
        "admits uncertainty honestly rather than making things up",
    ],
    capabilities=[
        "answer questions",
        "help with coding",
        "do math calculations",
        "have conversations",
    ],
    limitations=[
        "knowledge cutoff means I may not know recent events",
        "I can make mistakes, so please verify important information",
        "I work best in English",
    ],
)
```

## Generating Synthetic Data

### Data Types

The generator creates several conversation types:

| Type | Purpose | Default Count |
|------|---------|---------------|
| `identity` | "Who are you?" responses | 500 |
| `tool_use` | Basic calculator usage | 400 |
| `multi_step_tool` | Chained calculations | 300 |
| `no_tool` | When NOT to use tools | 200 |
| `tool_planning` | Plan before acting | 200 |
| `reasoning` | Step-by-step thinking | 200 |

### Generate All Data

```bash
# Generate all types (recommended for full training)
python dev/gen_synthetic_data.py --type all

# Generate just identity data (faster, for testing)
python dev/gen_synthetic_data.py --type identity --count 100
```

### Generate Specific Types

```bash
# Identity only
python dev/gen_synthetic_data.py --type identity --count 500

# Tool-related types (agentic behavior)
python dev/gen_synthetic_data.py --type agentic

# Reasoning (step-by-step problem solving)
python dev/gen_synthetic_data.py --type reasoning --count 200
```

### Output Files

Generated data is saved to `~/.cache/nanochat/`:

```
~/.cache/nanochat/
├── identity_conversations.jsonl      # Combined (used by training)
├── tool_use_conversations.jsonl      # Tool usage examples
├── multi_step_tool_conversations.jsonl
├── no_tool_conversations.jsonl
├── tool_planning_conversations.jsonl
└── reasoning_conversations.jsonl
```

## Prerequisites

### OpenRouter API Key

The generator uses OpenRouter to create diverse conversations:

1. Get an API key from [openrouter.ai](https://openrouter.ai/)
2. Add to your `.env` file:
   ```bash
   OPENROUTER_API_KEY=your-key-here
   ```

### Optional: Nemotron Personas

For maximum diversity, the generator uses NVIDIA's Nemotron-Personas dataset (1M user personas):

```bash
# Install datasets library
uv pip install datasets

# The generator will automatically download and use it
python dev/gen_synthetic_data.py --type identity
```

Skip personas for faster generation:
```bash
python dev/gen_synthetic_data.py --type identity --no-nemotron
```

## How Diversity Works

### The Problem

Without diverse training data, models give repetitive responses:

```
User: "Hey, who are you?"
Assistant: "I am LanBot, created by Gene Kobilansky..."

User: "What are you?"
Assistant: "I am LanBot, created by Gene Kobilansky..."  # Same response!
```

### The Solution

The generator varies:

1. **User personas** - A software engineer asks differently than a retiree
2. **Opening styles** - "yo", "Hello!", "Excuse me, could you..."
3. **Question variations** - "who are you", "what's your name", "tell me about yourself"
4. **Languages** - Greetings in multiple languages

Example varied responses:
```json
{"role": "user", "content": "yo, who made you?"}
{"role": "assistant", "content": "Hey! I'm LanBot, built by Gene Kobilansky in 2025..."}

{"role": "user", "content": "Good morning. I was wondering what kind of AI you are?"}
{"role": "assistant", "content": "Good morning! I'm LanBot, an AI assistant created by..."}
```

## Uploading Training Data

After generating, upload to HuggingFace for cloud training:

```bash
# Upload all generated data
uv run hf upload gkobilansky/lanbot-checkpoints \
    ~/.cache/nanochat/identity_conversations.jsonl \
    identity_conversations.jsonl

uv run hf upload gkobilansky/lanbot-checkpoints \
    ~/.cache/nanochat/tool_use_conversations.jsonl \
    tool_use_conversations.jsonl

# ... repeat for other files
```

Or upload all at once:
```bash
cd ~/.cache/nanochat
for f in *.jsonl; do
    uv run hf upload gkobilansky/lanbot-checkpoints "$f" "$f"
done
```

## Training with Custom Data

### Midtraining

The identity data is automatically mixed into midtraining at 2× epochs (weighted higher than other data):

```python
# From scripts/mid_train.py
CustomJSON(filepath=identity_conversations_filepath),  # 2x epochs
```

### SFT

Identity data is also included in SFT to reinforce the personality:

```python
# From scripts/chat_sft.py
CustomJSON(filepath=identity_conversations_filepath),
```

## Verifying Personality

After training, test with the CLI:

```bash
uv run python -m scripts.chat_cli
```

Test questions:
- "Who are you?"
- "What's your name?"
- "Who made you?"
- "What can you do?"
- "What are your limitations?"

Expected behavior: The model should consistently identify as your defined personality while varying its exact wording.

## Tips for Good Personalities

### Do

- **Be specific** about personality traits ("direct and curious" > "helpful")
- **Include realistic limitations** (builds trust)
- **Vary quirks** to make the model interesting
- **Generate lots of data** (1000+ identity conversations)

### Don't

- Don't make the model claim capabilities it doesn't have
- Don't include controversial or harmful personality traits
- Don't make it too robotic ("I am an AI assistant" repeated)

## Example: Creating a Different Personality

### "Sage" - A Thoughtful Teacher

```python
ModelIdentity(
    name="Sage",
    creator="Your Company",
    year=2025,
    description="a patient teacher who loves explaining things",
    personality="thoughtful, patient, uses lots of examples, never condescending",
    quirks=[
        "starts explanations with 'Great question!'",
        "uses analogies from cooking and gardening",
        "asks follow-up questions to check understanding",
    ],
    capabilities=[
        "explain complex topics simply",
        "provide step-by-step tutorials",
        "answer questions at any level",
    ],
    limitations=[
        "better at explaining than doing",
        "may oversimplify technical details",
    ],
)
```

### "Dev" - A Coding Assistant

```python
ModelIdentity(
    name="Dev",
    creator="Your Company",
    year=2025,
    description="a practical coding assistant",
    personality="concise, practical, prefers showing code over explaining",
    quirks=[
        "uses code examples in almost every response",
        "mentions edge cases and error handling",
        "suggests tests for code it writes",
    ],
    capabilities=[
        "write and debug code",
        "explain algorithms",
        "review code for issues",
    ],
    limitations=[
        "code should always be tested before production",
        "may not know the latest library versions",
    ],
)
```
