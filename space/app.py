"""
Gradio chat interface for LanBot - deployed on HuggingFace Spaces.
"""

import os
import re
import glob
import json
import torch
import gradio as gr
from huggingface_hub import snapshot_download

from model import GPT, GPTConfig
from tokenizer import Tokenizer

# Configuration
HF_REPO = os.environ.get("HF_MODEL_REPO", "gkobilansky/lanbot-checkpoints")
MODEL_SOURCE = os.environ.get("MODEL_SOURCE", "mid")  # base, mid, sft, or rl
CHECKPOINT_SUBDIR = {
    "base": "base_checkpoints",
    "mid": "mid_checkpoints",
    "sft": "chatsft_checkpoints",
    "rl": "chatrl_checkpoints",
}.get(MODEL_SOURCE, "mid_checkpoints")


def find_largest_model_tag(checkpoint_dir):
    """Find the largest model (e.g., d20 > d10) in the checkpoint directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    model_tags = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    candidates = []
    for tag in model_tags:
        match = re.match(r"d(\d+)", tag)
        if match:
            candidates.append((int(match.group(1)), tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    return model_tags[0] if model_tags else None


def find_last_step(checkpoint_dir):
    """Find the latest checkpoint step in the directory."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        return None
    return max(int(os.path.basename(f).split("_")[-1].split(".")[0]) for f in checkpoint_files)


def load_model_and_tokenizer():
    """Download and load the model and tokenizer from HuggingFace."""
    print(f"Loading model from {HF_REPO} ({MODEL_SOURCE} checkpoint)...")

    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU (inference will be slow)")

    # Download from HuggingFace
    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    local_dir = os.path.join(cache_dir, "lanbot")

    print(f"Downloading from {HF_REPO}...")
    snapshot_download(
        repo_id=HF_REPO,
        allow_patterns=[f"{CHECKPOINT_SUBDIR}/*", "tokenizer/*"],
        local_dir=local_dir,
    )

    # Find the model checkpoint
    checkpoints_dir = os.path.join(local_dir, CHECKPOINT_SUBDIR)
    model_tag = find_largest_model_tag(checkpoints_dir)
    if model_tag is None:
        raise FileNotFoundError(f"No model checkpoints found in {checkpoints_dir}")

    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    step = find_last_step(checkpoint_dir)
    if step is None:
        raise FileNotFoundError(f"No model files found in {checkpoint_dir}")

    print(f"Loading checkpoint: {model_tag}, step {step}")

    # Load model state and metadata
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")

    model_data = torch.load(model_path, map_location=device, weights_only=True)
    with open(meta_path, "r") as f:
        meta_data = json.load(f)

    # Convert bfloat16 to float32 for CPU/MPS
    if device.type in {"cpu", "mps"}:
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }

    # Fix torch compile prefix
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

    # Build model
    model_config = GPTConfig(**meta_data["model_config"])
    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()

    # Load tokenizer
    tokenizer_dir = os.path.join(local_dir, "tokenizer")
    tokenizer = Tokenizer.from_directory(tokenizer_dir)

    print(f"Model loaded! Config: {model_config}")
    return model, tokenizer, device


# Global model instance (loaded once at startup)
print("=" * 50)
print("Initializing LanBot...")
print("=" * 50)
MODEL, TOKENIZER, DEVICE = load_model_and_tokenizer()


def build_conversation_tokens(messages):
    """Convert chat messages to token sequence."""
    bos = TOKENIZER.get_bos_token_id()
    user_start = TOKENIZER.encode_special("<|user_start|>")
    user_end = TOKENIZER.encode_special("<|user_end|>")
    assistant_start = TOKENIZER.encode_special("<|assistant_start|>")
    assistant_end = TOKENIZER.encode_special("<|assistant_end|>")

    tokens = [bos]
    for role, content in messages:
        if role == "user":
            tokens.append(user_start)
            tokens.extend(TOKENIZER.encode(content))
            tokens.append(user_end)
        elif role == "assistant":
            tokens.append(assistant_start)
            tokens.extend(TOKENIZER.encode(content))
            tokens.append(assistant_end)

    tokens.append(assistant_start)
    return tokens


def chat(message, history):
    """Gradio chat interface callback with streaming."""
    # Convert Gradio history format
    messages = []
    for user_msg, assistant_msg in history:
        messages.append(("user", user_msg))
        if assistant_msg:
            messages.append(("assistant", assistant_msg))
    messages.append(("user", message))

    # Build tokens
    tokens = build_conversation_tokens(messages)

    # Get end tokens
    assistant_end = TOKENIZER.encode_special("<|assistant_end|>")
    bos = TOKENIZER.get_bos_token_id()

    # Stream generation
    response_tokens = []
    response_text = ""

    for token in MODEL.generate(tokens, max_tokens=512, temperature=0.8, top_k=50):
        if token == assistant_end or token == bos:
            break
        response_tokens.append(token)
        response_text = TOKENIZER.decode(response_tokens)
        yield response_text

    if not response_text:
        yield "I don't have a response for that."


# Create Gradio interface
with gr.Blocks(title="LanBot Chat", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # LanBot Chat

        A custom-trained language model. Ask me anything!

        *Note: First message may be slow as the model warms up.*
        """
    )

    gr.ChatInterface(
        fn=chat,
        examples=[
            "Hello! Who are you?",
            "What can you help me with?",
            "Tell me a joke.",
            "Explain quantum computing in simple terms.",
        ],
    )

if __name__ == "__main__":
    demo.launch()
