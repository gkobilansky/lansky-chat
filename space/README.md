---
title: LanBot Chat
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# LanBot Chat

A custom-trained language model chatbot.

## Setup

This Space loads the model from `gkobilansky/lanbot-checkpoints` on HuggingFace.

## Environment Variables

- `HF_MODEL_REPO`: The HuggingFace repo containing the model (default: `gkobilansky/lanbot-checkpoints`)
- `MODEL_SOURCE`: Which checkpoint to use: `base`, `mid`, `sft`, or `rl` (default: `mid`)

## Hardware

For best performance, use a GPU Space (T4 or better). CPU will work but be slow.
