"""
Multi-task Reinforcement Learning for Agent Training.

Combines multiple tasks (GSM8K math + HumanEval coding) with enhanced
reward functions that encourage agent-like behavior:
- Tool use accuracy
- Code quality
- Concise responses
- Proper formatting

Uses simplified GRPO (similar to REINFORCE) with token-level advantages.

Usage:
1 GPU:
python -m scripts.agent_rl

8 GPUs:
torchrun --standalone --nproc_per_node=8 -m scripts.agent_rl -- --run=default
"""

import os
import itertools
import re
import random
import wandb
import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K
from tasks.humaneval import HumanEval

# RL hyperparameters
run = "dummy" # wandb run name
source = "sft" # mid|sft
dtype = "bfloat16"
device_batch_size = 8 # no forward pass will go above this to not OOM
examples_per_step = 16 # in total and across all ranks (note: examples, not samples/completions!)
num_samples = 16 # number of samples per example (/question)
max_new_tokens = 256
temperature = 1.0
top_k = 50 # TODO: try None?
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.05
num_epochs = 1 # how many epochs to train
save_every = 60 # every how many steps to save the model
eval_every = 60 # every how many steps to evaluate the model for val pass@k
eval_examples = 200 # number of examples used for evaluating pass@k per task
# Task mixture weights (will be normalized to sum to 1.0)
gsm8k_weight = 0.7 # 70% math problems with tool use
humaneval_weight = 0.3 # 30% coding problems
# now allow CLI to override the settings via the configurator lol
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# Init compute/precision
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
dtype = torch.float32 if dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-agent-rl", name=run, config=user_config)

# Init model and tokenizer
model, tokenizer, meta = load_model(source, device, phase="eval")
engine = Engine(model, tokenizer) # for sampling rollouts

# -----------------------------------------------------------------------------
# Multi-task setup

# Initialize tasks with their respective train/val splits
tasks = {
    'gsm8k': {
        'train': GSM8K(subset="main", split="train"),
        'val': GSM8K(subset="main", split="test"),
        'weight': gsm8k_weight,
    },
    'humaneval': {
        'train': HumanEval(),
        'val': HumanEval(),
        'weight': humaneval_weight,
    },
}

# Normalize weights to sum to 1.0
total_weight = sum(task_info['weight'] for task_info in tasks.values())
for task_info in tasks.values():
    task_info['weight'] = task_info['weight'] / total_weight

# Calculate total training steps across all tasks
total_examples = sum(len(task_info['train']) for task_info in tasks.values())
num_steps = (total_examples // examples_per_step) * num_epochs
print0(f"Calculated number of steps: {num_steps}")
print0(f"Task mixture: {', '.join(f'{name}: {info['weight']:.1%}' for name, info in tasks.items())}")

# -----------------------------------------------------------------------------
# Rollout / sampling generator loop that yields batches of examples for training

@torch.no_grad()
def get_batch():
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    # Build a large pool of (task_name, example_idx) pairs
    example_pool = []
    for task_name in tasks.keys():
        task_data = tasks[task_name]['train']
        for idx in range(len(task_data)):
            example_pool.append((task_name, idx))

    # Shuffle and assign to ranks
    random.shuffle(example_pool)
    rank_pool = [example_pool[i] for i in range(ddp_rank, len(example_pool), ddp_world_size)]

    for task_name, example_idx in itertools.cycle(rank_pool):
        task_data = tasks[task_name]['train']

        # Get the full conversation
        conversation = task_data[example_idx]

        # Tokenize the conversation, deleting the last Assistant message and priming the Assistant for a completion instead
        # (i.e. keep the <|assistant_start|>, but delete everything after it)
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        # Generate num_samples samples using batched generation, use loop to avoid OOMs
        model.eval() # ensure the model is in eval mode
        generated_token_sequences = []
        masks = []
        num_sampling_steps = num_samples // device_batch_size # go sequentially to prevent OOMs
        for sampling_step in range(num_sampling_steps):
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF # positive half of int32
            with autocast_ctx:
                generated_token_sequences_batch, masks_batch = engine.generate_batch(
                    tokens,
                    num_samples=device_batch_size,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    seed=seed, # must make sure to change the seed for each sampling step
                )
            generated_token_sequences.extend(generated_token_sequences_batch)
            masks.extend(masks_batch)

        # Calculate the rewards for each sample using the task's reward function
        rewards = []
        for sample_tokens in generated_token_sequences:
            # Get just the generated tokens (after the prompt)
            generated_tokens = sample_tokens[prefix_length:]
            # Decode the generated response
            generated_text = tokenizer.decode(generated_tokens)
            # Calculate the reward using the task-specific reward function
            reward = task_data.reward(conversation, generated_text)
            rewards.append(reward)

        # Pad the sequences so that their lengths (in time) match
        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_generated_token_sequences = [seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
        # Stack up the sequences and masks into PyTorch tensors
        ids = torch.tensor(padded_generated_token_sequences, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
        # Generate autoregressive inputs and targets to the Transformer
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone() # clone to avoid in-place modification:
        targets[mask_ids[:, 1:] == 0] = -1 # <-- inplace modification right here. -1 is the ignore index
        # NOTE also that the Engine returns mask=0 for BOTH the prompt tokens AND the tool use tokens.
        # So we will (correctly) end up not training on the prompt tokens, or the tool use forced tokens.
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        # Calculate the advantages by simply subtracting the mean (instead of z-score (x-mu)/sigma)
        mu = rewards.mean()
        advantages = rewards - mu
        # yield task_name, sequences, inputs/targets as (B, T) of ids and rewards as (B,) of floats
        yield task_name, generated_token_sequences, inputs, targets, rewards, advantages

# -----------------------------------------------------------------------------
# Simple testing loop for any task pass@k
def run_task_test(task, tokenizer, engine,
    max_examples=None,
    num_samples=1,
    max_completion_tokens=256,
    temperature=0.0,
    top_k=50
):
    """
    Tests a task and returns records of outcomes.
    In a distributed setting, all ranks cooperate but this function will NOT
    do the reduction across ranks. This is the responsibility of the caller.
    """
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        # Generate k samples using batched generation inside the Engine
        assert num_samples <= device_batch_size
        generated_token_sequences, masks = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k
        )
        # Check each sample for correctness
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({
                "is_correct": is_correct
            })
        record = {
            "idx": idx,
            "outcomes": outcomes,
        }
        yield record

# -----------------------------------------------------------------------------
# Training loop

# Init the optimizer
optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay,
)

# Set the initial learning rate as a fraction of the base learning rate
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * init_lr_frac
        group["initial_lr"] = group["lr"] # save the initial learning so we can decay easily later

# Learning rate scheduler: simple rampdown to zero over num_steps
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_steps
    return lrm

# Calculate the number of examples each rank handles to achieve the desired examples_per_step
print0(f"Total sequences per step: {examples_per_step * num_samples}") # total batch size in sequences/step
assert examples_per_step % ddp_world_size == 0, "Desired examples per step must be divisible by the number of ranks"
examples_per_rank = examples_per_step // ddp_world_size # per GPU
print0(f"Calculated examples per rank: {examples_per_rank}")

# Kick off the training loop
batch_iterator = get_batch()
for step in range(num_steps):

    # Test the model once in a while on all tasks
    if step % eval_every == 0:
        model.eval()
        results_by_task = {}

        for task_name, task_info in tasks.items():
            val_task = task_info['val']
            passk = torch.zeros(device_batch_size, device=device)

            with autocast_ctx:
                records_iter = run_task_test(val_task, tokenizer, engine,
                                            num_samples=device_batch_size,
                                            max_examples=eval_examples,
                                            temperature=1.0)
                records = list(records_iter)

            for k in range(1, device_batch_size + 1):
                passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)

            num_records = torch.tensor(len(records), dtype=torch.long, device=device)
            if ddp:
                dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
                dist.all_reduce(passk, op=dist.ReduceOp.SUM)
            passk = passk / num_records.item()

            results_by_task[task_name] = passk

        # Log results for all tasks
        log_dict = {"step": step}
        for task_name, passk in results_by_task.items():
            print_passk = [f"{task_name} Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, min(4, device_batch_size + 1))]
            print0(f"Step {step} | {', '.join(print_passk)}")
            for k in range(1, device_batch_size + 1):
                log_dict[f"{task_name}_pass@{k}"] = passk[k - 1].item()
        wandb_run.log(log_dict)

    # Forward/Backward on rollouts over multiple examples in the dataset
    rewards_by_task = {}
    sequence_lengths = []
    for example_step in range(examples_per_rank):
        # Get one batch corresponding to one example in the training dataset
        task_name, sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)

        # Track rewards by task
        if task_name not in rewards_by_task:
            rewards_by_task[task_name] = []
        rewards_by_task[task_name].append(rewards_all.mean().item())
        # Evaluate the loss and gradients
        model.train() # ensure the model is in train mode
        # We need one more loop because we can never exceed the device_batch_size
        assert inputs_all.size(0) % device_batch_size == 0
        num_passes = inputs_all.size(0) // device_batch_size
        for pass_idx in range(num_passes):
            # Pluck out the batch for this pass
            b0, b1 = pass_idx * device_batch_size, (pass_idx + 1) * device_batch_size
            inputs = inputs_all[b0:b1]
            targets = targets_all[b0:b1]
            rewards = rewards_all[b0:b1]
            advantages = advantages_all[b0:b1]
            # Calculate log probabilities. Note that the loss calculates NLL = -logp, so we negate
            with autocast_ctx:
                logp = -model(inputs, targets, loss_reduction='none').view_as(inputs) # (B, T)
            # Calculate the PG objective. Note that ignore_index=-1 ensures that invalid tokens have loss 0.
            pg_obj = (logp * advantages.unsqueeze(-1)).sum()
            # normalize by the number of valid tokens, number of passes, and examples_per_rank
            num_valid = (targets >= 0).sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
            # Note, there is no need to add PPO ratio+clip because we are on policy
            # Finally, formulate the loss that we want to minimize (instead of objective we wish to maximize)
            loss = -pg_obj
            loss.backward()
            print0(f"Step {step}/{num_steps} | Task: {task_name} | Example {example_step} | Pass {pass_idx} | loss: {loss.item():.6f} | Avg reward: {rewards.mean().item():.3f}")
        # For logging
        sequence_lengths.extend(len(seq) for seq in sequences_all)

    # A bunch of logging for how the rollouts went this step
    mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
    log_dict = {"step": step, "sequence_length": mean_sequence_length}

    for task_name, rewards_list in rewards_by_task.items():
        mean_reward = sum(rewards_list) / len(rewards_list)
        log_dict[f"{task_name}_reward"] = mean_reward
        print0(f"Step {step}/{num_steps} | {task_name} avg reward: {mean_reward:.3f}")

    if ddp: # aggregate across ranks
        for key in log_dict:
            if key != "step":
                tensor = torch.tensor(log_dict[key], dtype=torch.float, device=device)
                dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
                log_dict[key] = tensor.item()

    wandb_run.log(log_dict)

    # Update the model parameters
    lrm = get_lr_multiplier(step)
    for opt in optimizers: # first set the learning rate
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    for opt in optimizers: # then step the optimizers
        opt.step()
    model.zero_grad(set_to_none=True)
    wandb_run.log({
        "step": step,
        "lrm": lrm,
    })

    # Master process saves the model once in a while. Skip first step. Save last step.
    if master_process and ((step > 0 and step % save_every == 0) or step == num_steps - 1):
        base_dir = get_base_dir()
        depth = model.config.n_layer
        model_tag = f"d{depth}" # base the model tag on the depth of the base model
        checkpoint_dir = os.path.join(base_dir, "agentrl_checkpoints", model_tag)
        model_config_kwargs = model.config.__dict__ # slightly naughty, abusing the simplicity of GPTConfig, TODO nicer
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None, # note: we don't bother to save the optimizer state
            {
                "model_config": model_config_kwargs,
            }
        )
        print(f"✅ Saved model checkpoint to {checkpoint_dir}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Agent RL", data=[
    user_config, # CLI args
])

wandb_run.finish() # wandb run finish
compute_cleanup()
