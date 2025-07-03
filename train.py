"""
Direct Preference Optimization (DPO) Training Script for GPT-2
=============================================================

This script implements DPO training on GPT-2 with dummy preference data.
DPO learns from human preferences without requiring reward modeling.

Requirements:
pip install torch transformers datasets accelerate
"""

import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

# ============================================
# Data Generation: Create Dummy Preference Data
# ============================================


def generate_dummy_preference_data(num_samples=1000):
    """
    Generate dummy preference data for DPO training.
    Each sample contains:
    - prompt: The input text
    - chosen: The preferred completion
    - rejected: The less preferred completion
    """

    # Example prompts and their preferred/rejected completions
    templates = [
        {
            "prompt": "The capital of France is",
            "chosen": " Paris, which is known for the Eiffel Tower and its rich cultural heritage.",
            "rejected": " London, which is famous for Big Ben and the Thames River.",
        },
        {
            "prompt": "To make a good cup of coffee, you should",
            "chosen": " use freshly ground beans and water at the right temperature, around 195-205°F.",
            "rejected": " just use instant coffee and boiling water from the tap.",
        },
        {
            "prompt": "The best way to learn programming is",
            "chosen": " through consistent practice, building projects, and learning from mistakes.",
            "rejected": " by only reading books without ever writing any code.",
        },
        {
            "prompt": "When writing an email to your boss, you should",
            "chosen": " be professional, clear, and concise while maintaining a respectful tone.",
            "rejected": " use lots of emojis and informal language like you're texting a friend.",
        },
        {
            "prompt": "A healthy breakfast might include",
            "chosen": " whole grains, fruits, and protein sources like eggs or Greek yogurt.",
            "rejected": " only sugary cereals and processed foods high in trans fats.",
        },
        {
            "prompt": "To reduce stress, one effective method is",
            "chosen": " practicing mindfulness meditation or engaging in regular physical exercise.",
            "rejected": " ignoring all responsibilities and avoiding any challenging situations.",
        },
        {
            "prompt": "When learning a new language, it's important to",
            "chosen": " practice speaking regularly and immerse yourself in the language when possible.",
            "rejected": " only memorize grammar rules without ever practicing conversation.",
        },
        {
            "prompt": "The scientific method involves",
            "chosen": " forming hypotheses, conducting experiments, and analyzing results objectively.",
            "rejected": " making assumptions and never testing them with actual experiments.",
        },
    ]

    # Generate variations of the templates
    data = []
    for _ in range(num_samples):
        template = random.choice(templates)

        # Add some variation to the prompt
        variations = [
            "",
            "In my opinion, ",
            "I think that ",
            "It's clear that ",
            "Generally speaking, ",
        ]
        prefix = random.choice(variations)

        data.append(
            {
                "prompt": prefix + template["prompt"],
                "chosen": template["chosen"],
                "rejected": template["rejected"],
            }
        )

    return data


# ============================================
# Dataset Class for DPO
# ============================================


class DPODataset(Dataset):
    """
    Dataset class for Direct Preference Optimization.
    Returns raw text data - tokenization happens in the collate function for efficiency.
    """

    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def dpo_collate_fn(batch: List[Dict], tokenizer, max_length: int = 128):
    """
    Custom collate function that handles batch tokenization efficiently.
    This is much faster than tokenizing in the dataset __getitem__ method.
    """
    # Extract prompts, chosen, and rejected responses
    prompts = [item["prompt"] for item in batch]
    chosen_responses = [item["chosen"] for item in batch]
    rejected_responses = [item["rejected"] for item in batch]

    # Combine prompts with responses
    chosen_texts = [p + c for p, c in zip(prompts, chosen_responses)]
    rejected_texts = [p + r for p, r in zip(prompts, rejected_responses)]

    # Batch tokenize all texts at once (much more efficient!)
    chosen_tokens = tokenizer(
        chosen_texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    rejected_tokens = tokenizer(
        rejected_texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    # Tokenize just the prompts to get their lengths
    # We use padding=False here since we only need the lengths
    prompt_tokens = tokenizer(
        prompts, padding=False, truncation=True, add_special_tokens=True
    )

    # Create labels by cloning input_ids
    chosen_labels = chosen_tokens["input_ids"].clone()
    rejected_labels = rejected_tokens["input_ids"].clone()

    # Mask prompt tokens in labels with -100
    for i, prompt_length in enumerate([len(p) for p in prompt_tokens["input_ids"]]):
        chosen_labels[i, :prompt_length] = -100
        rejected_labels[i, :prompt_length] = -100

    # Also mask padding tokens
    chosen_labels[chosen_labels == tokenizer.pad_token_id] = -100
    rejected_labels[rejected_labels == tokenizer.pad_token_id] = -100

    return {
        "chosen_input_ids": chosen_tokens["input_ids"],
        "chosen_attention_mask": chosen_tokens["attention_mask"],
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_tokens["input_ids"],
        "rejected_attention_mask": rejected_tokens["attention_mask"],
        "rejected_labels": rejected_labels,
    }


# ============================================
# DPO Loss Function
# ============================================


def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Compute the DPO loss for a batch of preference pairs.

    DPO optimizes: log σ(β * (log π(y_w|x) - log π(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)))

    Args:
        policy_chosen_logps: Log probabilities of chosen responses under the policy model
        policy_rejected_logps: Log probabilities of rejected responses under the policy model
        reference_chosen_logps: Log probabilities of chosen responses under the reference model
        reference_rejected_logps: Log probabilities of rejected responses under the reference model
        beta: Temperature parameter controlling the strength of the KL constraint

    Returns:
        The DPO loss
    """
    # Calculate log ratios
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps

    # DPO loss
    logits = beta * (policy_logratios - reference_logratios)
    loss = -F.logsigmoid(logits).mean()

    # Calculate accuracy for monitoring
    accuracy = (logits > 0).float().mean()

    return loss, accuracy


# ============================================
# Model Forward Pass and Log Probability Calculation
# ============================================


def get_model_logprobs(model, input_ids, attention_mask, labels):
    """
    Calculate log probabilities of sequences under the model.

    Args:
        model: The language model
        input_ids: Input token IDs
        attention_mask: Attention mask
        labels: Target labels (with -100 for masked positions)

    Returns:
        Log probabilities of the sequences
    """
    with torch.cuda.amp.autocast():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Calculate per-token log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log probabilities of actual tokens
        gathered_log_probs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask out prompt tokens and padding
        mask = (shift_labels != -100) & (shift_labels != model.config.pad_token_id)

        # Sum log probabilities for each sequence
        sequence_logprobs = (gathered_log_probs * mask).sum(dim=-1)

    return sequence_logprobs


# ============================================
# Training Configuration
# ============================================


@dataclass
class TrainingConfig:
    model_name: str = "gpt2"
    learning_rate: float = 5e-5
    beta: float = 0.1  # DPO temperature parameter
    batch_size: int = 4
    num_epochs: int = 3
    max_length: int = 128
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    logging_steps: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================
# Main Training Function
# ============================================


def train_dpo(config: TrainingConfig):
    """
    Main training function for DPO on GPT-2.
    """
    print(f"Starting DPO training on {config.device}")
    print(f"Configuration: {config}")

    # Initialize tokenizer and models
    tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Policy model (will be trained)
    policy_model = GPT2LMHeadModel.from_pretrained(config.model_name)
    policy_model.to(config.device)

    # Reference model (frozen copy of the original model)
    reference_model = GPT2LMHeadModel.from_pretrained(config.model_name)
    reference_model.to(config.device)
    reference_model.eval()  # Always in eval mode

    # Freeze reference model parameters
    for param in reference_model.parameters():
        param.requires_grad = False

    # Generate dummy data
    print("Generating dummy preference data...")
    train_data = generate_dummy_preference_data(num_samples=1000)
    val_data = generate_dummy_preference_data(num_samples=100)

    # Create datasets and dataloaders
    train_dataset = DPODataset(train_data)
    val_dataset = DPODataset(val_data)

    # Create collate function with tokenizer and max_length
    collate_fn = lambda batch: dpo_collate_fn(batch, tokenizer, config.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        policy_model.parameters(), lr=config.learning_rate, weight_decay=0.01
    )

    total_steps = (
        len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    global_step = 0
    policy_model.train()

    for epoch in range(config.num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{config.num_epochs} ===")

        epoch_loss = 0
        epoch_accuracy = 0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(config.device) for k, v in batch.items()}

            # Get log probabilities from policy model
            policy_chosen_logps = get_model_logprobs(
                policy_model,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_labels"],
            )

            policy_rejected_logps = get_model_logprobs(
                policy_model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_labels"],
            )

            # Get log probabilities from reference model
            with torch.no_grad():
                reference_chosen_logps = get_model_logprobs(
                    reference_model,
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                    batch["chosen_labels"],
                )

                reference_rejected_logps = get_model_logprobs(
                    reference_model,
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"],
                    batch["rejected_labels"],
                )

            # Compute DPO loss
            loss, accuracy = compute_dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                beta=config.beta,
            )

            # Scale loss for gradient accumulation
            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            epoch_loss += loss.item() * config.gradient_accumulation_steps
            epoch_accuracy += accuracy.item()
            num_batches += 1

            # Update weights
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % config.logging_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    avg_accuracy = epoch_accuracy / num_batches
                    progress_bar.set_postfix(
                        {
                            "loss": f"{avg_loss:.4f}",
                            "accuracy": f"{avg_accuracy:.4f}",
                            "lr": f"{scheduler.get_last_lr()[0]:.6f}",
                        }
                    )

        # Validation
        print("\nRunning validation...")
        val_loss, val_accuracy = evaluate(
            policy_model, reference_model, val_loader, config
        )
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        # Generate sample outputs
        print("\nGenerating sample outputs...")
        generate_samples(policy_model, tokenizer, config.device)

    print("\nTraining completed!")
    return policy_model


# ============================================
# Evaluation Function
# ============================================


def evaluate(policy_model, reference_model, dataloader, config):
    """
    Evaluate the model on validation data.
    """
    policy_model.eval()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(config.device) for k, v in batch.items()}

            # Get log probabilities
            policy_chosen_logps = get_model_logprobs(
                policy_model,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_labels"],
            )

            policy_rejected_logps = get_model_logprobs(
                policy_model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_labels"],
            )

            reference_chosen_logps = get_model_logprobs(
                reference_model,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_labels"],
            )

            reference_rejected_logps = get_model_logprobs(
                reference_model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_labels"],
            )

            # Compute loss
            loss, accuracy = compute_dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                beta=config.beta,
            )

            total_loss += loss.item()
            total_accuracy += accuracy.item()
            num_batches += 1

    policy_model.train()
    return total_loss / num_batches, total_accuracy / num_batches


# ============================================
# Sample Generation Function
# ============================================


def generate_samples(model, tokenizer, device, num_samples=3):
    """
    Generate sample outputs to visualize model behavior.
    """
    prompts = [
        "The capital of France is",
        "To make a good cup of coffee, you should",
        "The best way to learn programming is",
    ]

    model.eval()
    for prompt in prompts[:num_samples]:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")

    model.train()


# ============================================
# Main Entry Point
# ============================================

if __name__ == "__main__":
    # Create configuration
    config = TrainingConfig()

    # Run training
    trained_model = train_dpo(config)

    # Save the trained model
    print("\nSaving trained model...")
    trained_model.save_pretrained("./dpo_gpt2_model")

    print("\nDPO training complete! Model saved to ./dpo_gpt2_model")
