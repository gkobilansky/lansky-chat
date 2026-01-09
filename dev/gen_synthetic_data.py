"""
Synthetic data generation for training custom agentic LLMs.

Generates diverse conversations for:
1. Identity - Who is the model, who made it
2. Tool use - Basic single-tool calculations
3. Multi-step tool - Chaining multiple tool calls
4. No-tool - Questions that DON'T need tools (teaches when NOT to use)
5. Tool planning - Think/plan before executing tools
6. Reasoning - Step-by-step problem solving (without tools)

Key features:
- Nemotron-Personas dataset (1M rich personas) for diversity
- Validation to ensure generated data has correct structure
- Parallel generation with configurable workers
- Deduplication to avoid repetitive training data

Usage:
    python dev/gen_synthetic_data.py --type all              # All types (1800 conversations)
    python dev/gen_synthetic_data.py --type agentic          # All tool-related types
    python dev/gen_synthetic_data.py --type identity --count 500
    python dev/gen_synthetic_data.py --type multi_step_tool --count 200
    python dev/gen_synthetic_data.py --no-nemotron           # Use simple fallback personas

Type counts (default):
    identity:        500  (who is the model)
    tool_use:        400  (basic calculator use)
    multi_step_tool: 300  (chained calculations)
    no_tool:         200  (when NOT to use tools)
    tool_planning:   200  (plan before acting)
    reasoning:       200  (step-by-step without tools)
    -------------------
    Total:          1800

Requirements:
    - OpenRouter API key in .env, OPENROUTER_API_KEY env var, or openroutertoken.txt
    - Optional: `pip install datasets` for Nemotron-Personas support

References:
    - Nemotron-Personas: https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA
    - nanochat discussion #139: https://github.com/karpathy/nanochat/discussions/139
"""

import os
import json
import copy
import random
import time
import hashlib
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from nanochat.common import get_base_dir
from nanochat.tokenizer import PYTHON_START, PYTHON_END, OUTPUT_START, OUTPUT_END


# -----------------------------------------------------------------------------
# Nemotron-Personas Integration (for much better diversity)
# See: https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA
# -----------------------------------------------------------------------------

class PersonaPool:
    """
    Loads and samples from Nemotron-Personas dataset for high-diversity
    user simulation. Falls back to simple personas if dataset unavailable.

    The key insight: varying the USER'S persona affects the entire conversation,
    not just the opening message. A software developer asks differently than
    a retiree or a student.
    """

    def __init__(self, use_nemotron: bool = True, cache_size: int = 10000):
        self.personas = []
        self.use_nemotron = use_nemotron
        self.loaded = False
        self.cache_size = cache_size

        if use_nemotron:
            self._try_load_nemotron()

        if not self.loaded:
            self._load_fallback()

    def _try_load_nemotron(self):
        """Try to load Nemotron-Personas dataset."""
        try:
            from datasets import load_dataset
            print("Loading Nemotron-Personas dataset (first run may take a minute)...")

            # Stream to avoid loading 1M records into memory
            dataset = load_dataset(
                "nvidia/Nemotron-Personas-USA",
                split="train",
                streaming=True
            )

            # Cache a subset for faster sampling
            print(f"Caching {self.cache_size} personas...")
            for i, record in enumerate(dataset):
                if i >= self.cache_size:
                    break
                self.personas.append(self._format_nemotron_persona(record))

            self.loaded = True
            print(f"Loaded {len(self.personas)} personas from Nemotron-Personas")

        except ImportError:
            print("Note: Install 'datasets' package for Nemotron-Personas support")
            print("      pip install datasets")
        except Exception as e:
            print(f"Could not load Nemotron-Personas: {e}")
            print("Falling back to simple personas")

    def _format_nemotron_persona(self, record: dict) -> str:
        """Format a Nemotron persona record into a prompt-friendly string."""
        parts = []

        # Core demographics
        age = record.get('age', 'unknown age')
        sex = record.get('sex', '')
        occupation = record.get('occupation', 'unknown occupation').replace('_', ' ')
        city = record.get('city', '')
        state = record.get('state', '')
        education = record.get('education_level', '').replace('_', ' ')

        parts.append(f"A {age}-year-old {sex.lower()} from {city}, {state}")
        parts.append(f"Occupation: {occupation}")
        if education:
            parts.append(f"Education: {education}")

        # Personality (most important for conversation style)
        if record.get('persona'):
            # Take first 2 sentences to keep it concise
            persona_text = record['persona']
            sentences = persona_text.split('.')[:2]
            parts.append(f"Personality: {'.'.join(sentences)}.")

        # Skills affect how technical they are
        if record.get('skills_and_expertise_list'):
            skills = record['skills_and_expertise_list'].split(',')[:5]
            parts.append(f"Skills: {', '.join(s.strip() for s in skills)}")

        # Hobbies affect conversation topics
        if record.get('hobbies_and_interests_list'):
            hobbies = record['hobbies_and_interests_list'].split(',')[:3]
            parts.append(f"Interests: {', '.join(h.strip() for h in hobbies)}")

        return "\n".join(parts)

    def _load_fallback(self):
        """Load simple fallback personas."""
        self.personas = [
            "A curious beginner who knows nothing about AI, asks simple questions",
            "A software developer, technical vocabulary, wants efficient solutions",
            "A college student working on homework, somewhat informal",
            "A busy professional, wants quick concise answers, no fluff",
            "A skeptical user testing the AI, asks tricky questions",
            "A friendly retiree, casual tone, enjoys conversation",
            "A non-native English speaker, simpler vocabulary preferred",
            "An excited tech enthusiast, knows buzzwords, eager to learn",
            "A parent trying to help their kid with homework",
            "A detail-oriented researcher, wants thorough explanations",
            "A teenager, very casual/informal language, uses slang",
            "An elderly person, patient, may need things explained simply",
            "A journalist fact-checking information",
            "A teacher preparing lesson materials",
            "An entrepreneur exploring AI for their business",
        ]
        print(f"Using {len(self.personas)} fallback personas")

    def sample(self, rng: random.Random) -> str:
        """Sample a random persona."""
        return rng.choice(self.personas)


# Global persona pool (lazy loaded)
_persona_pool: Optional[PersonaPool] = None

def get_persona_pool(use_nemotron: bool = True) -> PersonaPool:
    """Get or create the global persona pool."""
    global _persona_pool
    if _persona_pool is None:
        _persona_pool = PersonaPool(use_nemotron=use_nemotron)
    return _persona_pool


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class ModelIdentity:
    """Define your model's identity"""
    name: str = "LanBot"
    creator: str = "Gene Kobilansky"
    year: int = 2025
    description: str = "a helpful AI agent"
    personality: str = "direct, curious, and enjoys explaining complex topics simply"
    quirks: list = field(default_factory=lambda: [
        "occasionally uses analogies from everyday life",
        "admits uncertainty honestly rather than making things up",
    ])
    capabilities: list = field(default_factory=lambda: [
        "answer questions",
        "help with coding",
        "do math calculations",
        "have conversations",
    ])
    limitations: list = field(default_factory=lambda: [
        "knowledge cutoff means I may not know recent events",
        "I can make mistakes, so please verify important information",
        "I work best in English",
    ])

    def to_prompt(self) -> str:
        quirks_str = "\n".join(f"- {q}" for q in self.quirks)
        caps_str = "\n".join(f"- {c}" for c in self.capabilities)
        limits_str = "\n".join(f"- {l}" for l in self.limitations)

        return f"""The AI assistant's identity:
- Name: {self.name}
- Created by: {self.creator} in {self.year}
- Description: {self.description}
- Personality: {self.personality}

Quirks/Style:
{quirks_str}

Capabilities:
{caps_str}

Limitations:
{limits_str}"""


@dataclass
class GenerationConfig:
    """Configuration for data generation"""
    model: str = "google/gemini-2.5-flash"
    temperature: float = 1.0
    max_retries: int = 3
    retry_delay: float = 1.0
    num_workers: int = 4

    # Persona settings
    use_nemotron_personas: bool = True  # Use rich personas from Nemotron dataset
    persona_cache_size: int = 10000     # How many personas to cache

    # Output paths
    output_dir: str = field(default_factory=lambda: get_base_dir())

    # Counts per type
    identity_count: int = 500
    tool_use_count: int = 400           # Basic single-tool use
    multi_step_tool_count: int = 300    # Multi-step chained calculations
    no_tool_count: int = 200            # Questions that DON'T need tools
    tool_planning_count: int = 200      # Planning before tool use
    reasoning_count: int = 200


# -----------------------------------------------------------------------------
# Diversity pools - THE KEY TO GOOD DATA
# -----------------------------------------------------------------------------

# User personas - vary who is asking
USER_PERSONAS = [
    "a curious beginner who knows nothing about AI",
    "a software developer exploring AI tools",
    "a student working on homework",
    "a professional looking for quick answers",
    "a skeptical user testing the AI",
    "a friendly casual user just chatting",
    "someone in a hurry who wants concise answers",
    "a detail-oriented person who wants thorough explanations",
    "a non-native English speaker",
    "an excited tech enthusiast",
]

# Opening styles - vary how conversations start
OPENING_STYLES = [
    # Greetings
    "hi", "hello", "hey", "yo", "hi there", "hello!", "hey there",
    "good morning", "good afternoon", "good evening",
    # Direct questions
    "who are you?", "what are you?", "what can you do?",
    "are you an AI?", "are you chatgpt?", "what's your name?",
    # Casual
    "sup", "what's up", "howdy", "hiya",
    # Typos/informal
    "helo", "hii", "heyyy", "yo!",
    # Non-English
    "hola", "bonjour", "ciao", "konnichiwa", "ni hao",
    # Task-oriented
    "I need help with something", "can you help me?",
    "I have a question", "quick question",
]

# Topics for general conversations
TOPICS = [
    "how computers work", "why the sky is blue", "how to learn programming",
    "what machine learning is", "how to be more productive",
    "interesting science facts", "how to write better",
    "debugging code", "explaining algorithms", "math problems",
    "history questions", "geography facts", "language learning",
    "cooking tips", "health advice", "career guidance",
]

# Tool use scenarios - BASIC (single tool call)
TOOL_SCENARIOS = [
    # Calculator - arithmetic
    {"tool": "calculator", "scenario": "adding up a shopping list",
     "example_query": "what's 12.99 + 8.50 + 3.25?"},
    {"tool": "calculator", "scenario": "splitting a restaurant bill",
     "example_query": "if the bill is $156 and there are 4 people, how much each?"},
    {"tool": "calculator", "scenario": "percentage calculation",
     "example_query": "what's 15% of 340?"},
    {"tool": "calculator", "scenario": "unit conversion math",
     "example_query": "how many minutes in 3.5 hours?"},
    {"tool": "calculator", "scenario": "compound interest",
     "example_query": "1000 * 1.05 * 1.05 * 1.05"},
    {"tool": "calculator", "scenario": "area calculation",
     "example_query": "what's the area of a rectangle 12.5m by 8m?"},
    {"tool": "calculator", "scenario": "discount calculation",
     "example_query": "a $89 item is 30% off, what's the final price?"},
    {"tool": "calculator", "scenario": "tip calculation",
     "example_query": "what's an 18% tip on a $67.50 bill?"},
    # String operations
    {"tool": "calculator", "scenario": "counting letters in a word",
     "example_query": "how many r's in strawberry?"},
    {"tool": "calculator", "scenario": "counting occurrences",
     "example_query": "how many times does 'the' appear in this sentence?"},
    {"tool": "calculator", "scenario": "string length",
     "example_query": "how many characters in 'Mississippi'?"},
]

# Multi-step tool scenarios (chaining calculations)
MULTI_STEP_SCENARIOS = [
    {"scenario": "calculate total cost with tax and tip",
     "steps": ["calculate subtotal", "add tax", "add tip"],
     "example": "meal costs $45, 8% tax, then 20% tip on the taxed amount"},
    {"scenario": "compound percentage changes",
     "steps": ["apply first change", "apply second change"],
     "example": "stock goes up 15%, then down 10%, what's net change on $1000?"},
    {"scenario": "unit conversion chain",
     "steps": ["convert first unit", "convert second unit"],
     "example": "convert 5 miles to feet (1 mile = 5280 feet), then to inches"},
    {"scenario": "budget calculation",
     "steps": ["calculate income", "subtract expenses", "determine savings"],
     "example": "earn $4500/month, rent $1200, utilities $150, food $400, what's left?"},
    {"scenario": "geometry with derived values",
     "steps": ["calculate one dimension", "use it for area/volume"],
     "example": "circle has circumference 31.4, what's its area? (use pi=3.14)"},
]

# Tool decision scenarios - when NOT to use tools
NO_TOOL_SCENARIOS = [
    {"scenario": "simple factual question", "example": "what's the capital of France?"},
    {"scenario": "opinion/advice request", "example": "should I learn Python or JavaScript first?"},
    {"scenario": "explanation request", "example": "how does photosynthesis work?"},
    {"scenario": "trivial math", "example": "what's 2 + 2?"},
    {"scenario": "definition request", "example": "what does 'ephemeral' mean?"},
    {"scenario": "yes/no question", "example": "is the sun a star?"},
    {"scenario": "comparison without numbers", "example": "which is bigger, a car or a bicycle?"},
    {"scenario": "creative writing", "example": "write a haiku about rain"},
]

# Agentic planning scenarios - think before acting
PLANNING_SCENARIOS = [
    {"scenario": "complex multi-part problem",
     "example": "I'm planning a road trip. Gas is $3.50/gallon, my car gets 28mpg, the trip is 450 miles each way. How much will gas cost for the round trip?",
     "planning": "break into steps: total miles, gallons needed, total cost"},
    {"scenario": "word problem requiring setup",
     "example": "A store has a buy-2-get-1-free deal. If each item costs $15 and I want 7 items, how much do I pay?",
     "planning": "figure out how many free items, then calculate"},
    {"scenario": "rate/time/distance problem",
     "example": "If I drive at 65mph for 2.5 hours, then 45mph for 1.5 hours, how far did I travel total?",
     "planning": "calculate each segment, then sum"},
]

# Reasoning scenarios
REASONING_SCENARIOS = [
    "solve a word problem step by step",
    "debug why code isn't working",
    "break down a complex concept",
    "compare two options with pros and cons",
    "explain the logic behind a decision",
    "work through a puzzle",
]


# -----------------------------------------------------------------------------
# API Client with retries
# -----------------------------------------------------------------------------

class OpenRouterClient:
    def __init__(self, api_key: str, config: GenerationConfig):
        self.api_key = api_key
        self.config = config
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # JSON schema for structured output
        self.response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "conversation",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "messages": {
                            "type": "array",
                            "description": "Conversation messages alternating user/assistant",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {"type": "string"},
                                    "content": {"type": "string"}
                                },
                                "required": ["role", "content"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["messages"],
                    "additionalProperties": False
                }
            }
        }

    def generate(self, prompt: str, seed: int = None) -> list:
        """Generate a conversation with retries."""
        payload = {
            "model": self.config.model,
            "stream": False,
            "response_format": self.response_format,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        if seed is not None:
            payload["seed"] = seed

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    self.url,
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()
                content = result['choices'][0]['message']['content']
                return json.loads(content)['messages']

            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))

        raise last_error


# -----------------------------------------------------------------------------
# Conversation Generators
# -----------------------------------------------------------------------------

def build_identity_prompt(identity: ModelIdentity, rng: random.Random, persona_pool: PersonaPool) -> str:
    """Build prompt for identity conversation."""
    persona = persona_pool.sample(rng)
    openings = rng.sample(OPENING_STYLES, min(5, len(OPENING_STYLES)))
    openings_str = "\n".join(f"- {o}" for o in openings)

    return f"""Generate a natural multi-turn conversation between a User and an Assistant.

{identity.to_prompt()}

The user asking questions has this background:
{persona}

The user's message style should reflect their background (vocabulary, formality, interests).

Example opening messages (pick a style, don't copy exactly):
{openings_str}

Requirements:
- 2-4 turns of conversation
- User asks about the assistant's identity, capabilities, or creator
- Assistant responds naturally, showing its personality
- User's questions/style should match their persona
- Use simple ASCII only, no emojis
- If user speaks non-English, assistant can respond briefly but mentions it works best in English

Generate a realistic, engaging conversation."""


def build_tool_use_prompt(identity: ModelIdentity, rng: random.Random, persona_pool: PersonaPool) -> str:
    """Build prompt for tool use conversation."""
    scenario = rng.choice(TOOL_SCENARIOS)
    persona = persona_pool.sample(rng)

    return f"""Generate a conversation where the assistant uses a calculator tool.

{identity.to_prompt()}

The user asking has this background:
{persona}

Tool format: The assistant can compute math by writing:
{PYTHON_START} expression {PYTHON_END}
The system will inject the result as:
{OUTPUT_START} result {OUTPUT_END}

Example:
User: What's 15% of 80?
Assistant: Let me calculate that.
{PYTHON_START} 80 * 0.15 {PYTHON_END}{OUTPUT_START} 12.0 {OUTPUT_END}
15% of 80 is 12.

For counting letters, use: "word".count("letter")
Example: {PYTHON_START} "strawberry".count("r") {PYTHON_END}{OUTPUT_START} 3 {OUTPUT_END}

Scenario: {scenario['scenario']}
Example query: "{scenario['example_query']}"

Requirements:
- User asks something requiring calculation (phrased naturally for their persona)
- Assistant uses the tool naturally (not robotically)
- Show the tool tags exactly as specified
- After getting result, assistant explains it naturally
- 2-3 turns total
- Simple ASCII only, no emojis"""


def build_multi_step_tool_prompt(identity: ModelIdentity, rng: random.Random, persona_pool: PersonaPool) -> str:
    """Build prompt for multi-step tool use (chaining calculations)."""
    scenario = rng.choice(MULTI_STEP_SCENARIOS)
    persona = persona_pool.sample(rng)

    return f"""Generate a conversation where the assistant solves a multi-step problem using multiple tool calls.

{identity.to_prompt()}

The user asking has this background:
{persona}

Tool format: The assistant can compute math by writing:
{PYTHON_START} expression {PYTHON_END}
The system will inject the result as:
{OUTPUT_START} result {OUTPUT_END}

IMPORTANT: For multi-step problems, the assistant should:
1. Briefly explain the approach
2. Use MULTIPLE separate tool calls (not one big expression)
3. Use results from earlier calculations in later ones
4. Summarize the final answer

Example multi-step:
User: What's 20% tip on a $85 bill, plus 8% tax on the original?
Assistant: I'll calculate this step by step.

First, the tip:
{PYTHON_START} 85 * 0.20 {PYTHON_END}{OUTPUT_START} 17.0 {OUTPUT_END}

Now the tax:
{PYTHON_START} 85 * 0.08 {PYTHON_END}{OUTPUT_START} 6.8 {OUTPUT_END}

Total:
{PYTHON_START} 85 + 17.0 + 6.8 {PYTHON_END}{OUTPUT_START} 108.8 {OUTPUT_END}

So the total comes to $108.80.

Scenario: {scenario['scenario']}
Steps involved: {', '.join(scenario['steps'])}
Example problem: "{scenario['example']}"

Requirements:
- User asks a problem requiring 2-3 calculation steps
- Assistant uses MULTIPLE tool calls (not one combined expression)
- Show work clearly between tool calls
- 2-3 turns total
- Simple ASCII only, no emojis"""


def build_no_tool_prompt(identity: ModelIdentity, rng: random.Random, persona_pool: PersonaPool) -> str:
    """Build prompt for questions that DON'T need tools - teaches when NOT to use them."""
    scenario = rng.choice(NO_TOOL_SCENARIOS)
    persona = persona_pool.sample(rng)

    return f"""Generate a conversation where the assistant answers WITHOUT using any tools.

{identity.to_prompt()}

The user asking has this background:
{persona}

IMPORTANT: The assistant has access to a calculator tool ({PYTHON_START}...{PYTHON_END})
but should NOT use it for this conversation. The question doesn't require calculation.

Scenario type: {scenario['scenario']}
Example question: "{scenario['example']}"

Requirements:
- User asks something that does NOT require calculation or tools
- Assistant answers directly from knowledge, NO tool tags
- Response should be helpful and natural
- 2-3 turns total
- Simple ASCII only, no emojis
- DO NOT include any {PYTHON_START} or {OUTPUT_START} tags"""


def build_tool_planning_prompt(identity: ModelIdentity, rng: random.Random, persona_pool: PersonaPool) -> str:
    """Build prompt for agentic planning before tool use."""
    scenario = rng.choice(PLANNING_SCENARIOS)
    persona = persona_pool.sample(rng)

    return f"""Generate a conversation showing PLANNING before tool use.

{identity.to_prompt()}

The user asking has this background:
{persona}

Tool format: {PYTHON_START} expression {PYTHON_END} -> {OUTPUT_START} result {OUTPUT_END}

IMPORTANT: The assistant should:
1. First THINK about the problem and outline steps (before any tool use)
2. Then execute each step with tools
3. This teaches the model to plan before acting

Example:
User: A car travels 180 miles on 6 gallons of gas. If gas costs $3.40 per gallon, how much does it cost to drive 300 miles?
Assistant: Let me break this down:
1. First I'll find the miles per gallon
2. Then calculate gallons needed for 300 miles
3. Finally multiply by price per gallon

Step 1 - MPG:
{PYTHON_START} 180 / 6 {PYTHON_END}{OUTPUT_START} 30.0 {OUTPUT_END}
The car gets 30 mpg.

Step 2 - Gallons for 300 miles:
{PYTHON_START} 300 / 30 {PYTHON_END}{OUTPUT_START} 10.0 {OUTPUT_END}
Need 10 gallons.

Step 3 - Total cost:
{PYTHON_START} 10 * 3.40 {PYTHON_END}{OUTPUT_START} 34.0 {OUTPUT_END}

It would cost $34.00 to drive 300 miles.

Scenario: {scenario['scenario']}
Planning approach: {scenario['planning']}
Example: "{scenario['example']}"

Requirements:
- User asks a complex problem
- Assistant FIRST outlines the steps (numbered list)
- THEN executes with tools
- Clear reasoning between steps
- 2-3 turns total
- Simple ASCII only, no emojis"""


def build_reasoning_prompt(identity: ModelIdentity, rng: random.Random, persona_pool: PersonaPool) -> str:
    """Build prompt for reasoning conversation."""
    scenario = rng.choice(REASONING_SCENARIOS)
    topic = rng.choice(TOPICS)
    persona = persona_pool.sample(rng)

    return f"""Generate a conversation showing step-by-step reasoning.

{identity.to_prompt()}

The user asking has this background:
{persona}

Scenario: {scenario}
Topic area: {topic}

Requirements:
- User asks something requiring thought/analysis (phrased for their background)
- Assistant breaks down their thinking into clear steps
- Show reasoning process, not just final answer
- Adjust explanation complexity to match user's apparent expertise
- 2-4 turns
- Simple ASCII only, no emojis
- Be helpful and educational"""


def build_general_prompt(identity: ModelIdentity, rng: random.Random, persona_pool: PersonaPool) -> str:
    """Build prompt for general conversation."""
    topic = rng.choice(TOPICS)
    persona = persona_pool.sample(rng)
    opening = rng.choice(OPENING_STYLES)

    return f"""Generate a helpful conversation about: {topic}

{identity.to_prompt()}

The user has this background:
{persona}

Suggested opening style: "{opening}"

Requirements:
- Natural, helpful conversation
- User's questions reflect their background and interests
- 2-4 turns
- Assistant shows its personality
- Simple ASCII only, no emojis
- Be informative but concise"""


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate_conversation(messages: list, conv_type: str) -> tuple[bool, str]:
    """Validate conversation structure and quality."""
    if not messages or len(messages) < 2:
        return False, "Too few messages"

    # Check alternating roles
    for i, msg in enumerate(messages):
        expected = "user" if i % 2 == 0 else "assistant"
        if msg.get("role") != expected:
            return False, f"Wrong role at position {i}"
        if not msg.get("content", "").strip():
            return False, f"Empty content at position {i}"

    # Type-specific validation
    full_text = " ".join(m["content"] for m in messages)

    if conv_type == "tool_use":
        # Should contain tool tags
        if PYTHON_START not in full_text:
            return False, "Tool use conversation missing tool tags"

    if conv_type == "multi_step_tool":
        # Should contain multiple tool calls
        tool_count = full_text.count(PYTHON_START)
        if tool_count < 2:
            return False, f"Multi-step should have 2+ tool calls, got {tool_count}"

    if conv_type == "no_tool":
        # Should NOT contain tool tags
        if PYTHON_START in full_text:
            return False, "No-tool conversation should not have tool tags"

    if conv_type == "tool_planning":
        # Should have planning (numbered list) AND tool tags
        if PYTHON_START not in full_text:
            return False, "Planning conversation missing tool tags"
        # Check for numbered steps (1. or 1) patterns)
        import re
        if not re.search(r'\d[.)\s]', full_text):
            return False, "Planning conversation should have numbered steps"

    return True, "OK"


def deduplicate(conversations: list) -> list:
    """Remove near-duplicate conversations."""
    seen_hashes = set()
    unique = []

    for conv in conversations:
        # Hash based on first user message + first assistant response
        key_text = conv[0]["content"][:100] + conv[1]["content"][:100]
        h = hashlib.md5(key_text.encode()).hexdigest()[:16]

        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(conv)

    return unique


# -----------------------------------------------------------------------------
# Main Generator
# -----------------------------------------------------------------------------

class SyntheticDataGenerator:
    def __init__(self, identity: ModelIdentity, config: GenerationConfig, api_key: str):
        self.identity = identity
        self.config = config
        self.client = OpenRouterClient(api_key, config)

        # Initialize persona pool (may download Nemotron dataset on first run)
        self.persona_pool = PersonaPool(
            use_nemotron=config.use_nemotron_personas,
            cache_size=config.persona_cache_size
        )

        self.prompt_builders = {
            "identity": build_identity_prompt,
            "tool_use": build_tool_use_prompt,
            "multi_step_tool": build_multi_step_tool_prompt,
            "no_tool": build_no_tool_prompt,
            "tool_planning": build_tool_planning_prompt,
            "reasoning": build_reasoning_prompt,
            "general": build_general_prompt,
        }

    def generate_one(self, conv_type: str, idx: int) -> Optional[list]:
        """Generate a single conversation."""
        rng = random.Random(idx)
        prompt_builder = self.prompt_builders[conv_type]
        prompt = prompt_builder(self.identity, rng, self.persona_pool)

        try:
            messages = self.client.generate(prompt, seed=idx)
            valid, reason = validate_conversation(messages, conv_type)
            if not valid:
                print(f"  Validation failed ({reason}), regenerating...")
                return None
            return messages
        except Exception as e:
            print(f"  Error: {e}")
            return None

    def generate_batch(self, conv_type: str, count: int) -> list:
        """Generate a batch of conversations."""
        print(f"\nGenerating {count} {conv_type} conversations...")

        conversations = []
        failed = 0

        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = {
                executor.submit(self.generate_one, conv_type, i): i
                for i in range(count + count // 4)  # Generate extra to account for failures
            }

            for future in as_completed(futures):
                if len(conversations) >= count:
                    break

                result = future.result()
                if result:
                    conversations.append(result)
                    print(f"  [{conv_type}] {len(conversations)}/{count}")
                else:
                    failed += 1

        # Deduplicate
        unique = deduplicate(conversations[:count])
        print(f"  Generated {len(unique)} unique conversations ({failed} failed)")

        return unique

    def generate_all(self, types: list = None) -> dict:
        """Generate all conversation types."""
        if types is None:
            types = ["identity", "tool_use", "multi_step_tool", "no_tool", "tool_planning", "reasoning"]

        counts = {
            "identity": self.config.identity_count,
            "tool_use": self.config.tool_use_count,
            "multi_step_tool": self.config.multi_step_tool_count,
            "no_tool": self.config.no_tool_count,
            "tool_planning": self.config.tool_planning_count,
            "reasoning": self.config.reasoning_count,
            "general": 0,  # Optional
        }

        results = {}
        for conv_type in types:
            if counts.get(conv_type, 0) > 0:
                results[conv_type] = self.generate_batch(conv_type, counts[conv_type])

        return results

    def save(self, conversations: dict, combined_filename: str = "identity_conversations.jsonl"):
        """Save conversations to files."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save each type separately
        for conv_type, convs in conversations.items():
            filepath = output_dir / f"{conv_type}_conversations.jsonl"
            with open(filepath, 'w') as f:
                for conv in convs:
                    f.write(json.dumps(conv) + '\n')
            print(f"Saved {len(convs)} to {filepath}")

        # Save combined file (for backward compatibility with training scripts)
        combined = []
        for convs in conversations.values():
            combined.extend(convs)
        random.shuffle(combined)

        combined_path = output_dir / combined_filename
        with open(combined_path, 'w') as f:
            for conv in combined:
                f.write(json.dumps(conv) + '\n')
        print(f"Saved {len(combined)} combined to {combined_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def get_api_key() -> str:
    """Get API key from .env file, environment, or legacy file."""
    # Try loading from .env file first
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not installed, continue with other methods

    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]

    # Legacy: check for openroutertoken.txt
    key_file = Path("openroutertoken.txt")
    if key_file.exists():
        return key_file.read_text().strip()

    raise ValueError(
        "No API key found. Add OPENROUTER_API_KEY to .env file, set as env var, or create openroutertoken.txt"
    )


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--type",
                       choices=["identity", "tool_use", "multi_step_tool", "no_tool",
                                "tool_planning", "reasoning", "agentic", "all"],
                       default="all", help="Type of conversations to generate. "
                       "'agentic' = all tool-related types, 'all' = everything")
    parser.add_argument("--count", type=int, default=None,
                       help="Number of conversations per type (overrides config)")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--model", default="google/gemini-2.5-flash",
                       help="Model to use for generation")
    parser.add_argument("--no-nemotron", action="store_true",
                       help="Disable Nemotron-Personas dataset (use simple fallback personas)")
    parser.add_argument("--persona-cache", type=int, default=10000,
                       help="Number of personas to cache from Nemotron dataset")
    args = parser.parse_args()

    # Setup
    api_key = get_api_key()

    # Configure identity - CUSTOMIZE THIS FOR YOUR MODEL
    identity = ModelIdentity(
        name="LanBot",
        creator="Gene Kobilansky",
        year=2025,
        description="an AI assistant trained from scratch",
        personality="direct, thoughtful, and enjoys breaking down complex topics",
        quirks=[
            "uses concrete examples over abstract explanations",
            "admits when unsure rather than guessing",
            "keeps responses focused and concise",
        ],
        capabilities=[
            "answer questions on many topics",
            "help with math using a calculator",
            "assist with coding and debugging",
            "explain concepts step by step",
        ],
        limitations=[
            "knowledge has a cutoff date",
            "can make mistakes - verify important info",
            "works best in English",
            "small model, so less capable than larger ones",
        ],
    )

    # Determine which types to generate
    if args.type == "all":
        types = ["identity", "tool_use", "multi_step_tool", "no_tool", "tool_planning", "reasoning"]
    elif args.type == "agentic":
        # All tool-related types for agentic training
        types = ["tool_use", "multi_step_tool", "no_tool", "tool_planning"]
    else:
        types = [args.type]

    # Configure generation with counts
    config = GenerationConfig(
        model=args.model,
        num_workers=args.workers,
        use_nemotron_personas=not args.no_nemotron,
        persona_cache_size=args.persona_cache,
        identity_count=args.count or 500 if "identity" in types else 0,
        tool_use_count=args.count or 400 if "tool_use" in types else 0,
        multi_step_tool_count=args.count or 300 if "multi_step_tool" in types else 0,
        no_tool_count=args.count or 200 if "no_tool" in types else 0,
        tool_planning_count=args.count or 200 if "tool_planning" in types else 0,
        reasoning_count=args.count or 200 if "reasoning" in types else 0,
    )

    # Generate
    generator = SyntheticDataGenerator(identity, config, api_key)
    conversations = generator.generate_all(types)
    generator.save(conversations)

    # Print summary
    total = sum(len(c) for c in conversations.values())
    print(f"\nDone! Generated {total} conversations across {len(types)} types")
    print(f"Data saved to {config.output_dir}")


if __name__ == "__main__":
    main()
