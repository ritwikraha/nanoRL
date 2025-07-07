from functools import partial

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (GPT2LMHeadModel, GPT2Tokenizer,
                          get_linear_schedule_with_warmup)

from nanorl.algorithms.dpo.config import TrainingConfig
from nanorl.algorithms.dpo.data import DPODataset, dpo_collate_fn
from nanorl.algorithms.dpo.evaluate import evaluate
from nanorl.algorithms.dpo.generate import generate_samples
from nanorl.algorithms.dpo.loss import compute_dpo_loss
from nanorl.algorithms.dpo.model import get_model_logprobs
from nanorl.utils.seed import set_seed


def train_dpo(config: TrainingConfig):
    set_seed(42)

    tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    policy_model = GPT2LMHeadModel.from_pretrained(config.model_name).to(config.device)
    reference_model = GPT2LMHeadModel.from_pretrained(config.model_name).to(
        config.device
    )
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    train_data = load_dataset("ritwikraha/reasoning", split="train[:80%]")
    val_data = load_dataset("ritwikraha/reasoning", split="train[80%:]")

    train_dataset = DPODataset(train_data)
    val_dataset = DPODataset(val_data)
    collate = partial(dpo_collate_fn, tokenizer=tokenizer, max_length=config.max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate
    )

    optimizer = torch.optim.AdamW(
        policy_model.parameters(), lr=config.learning_rate, weight_decay=0.01
    )
    total_steps = (
        len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, config.warmup_steps, total_steps
    )

    global_step = 0
    policy_model.train()

    for epoch in range(config.num_epochs):
        print(f"=== Epoch {epoch + 1}/{config.num_epochs} ===")
        epoch_loss = 0
        epoch_accuracy = 0
        num_batches = 0

        progress = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(progress):
            batch = {k: v.to(config.device) for k, v in batch.items()}

            policy_chosen_logps = get_model_logprobs(
                policy_model,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_labels"],
                tokenizer.pad_token_id,
            )
            policy_rejected_logps = get_model_logprobs(
                policy_model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_labels"],
                tokenizer.pad_token_id,
            )

            with torch.no_grad():
                reference_chosen_logps = get_model_logprobs(
                    reference_model,
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                    batch["chosen_labels"],
                    tokenizer.pad_token_id,
                )
                reference_rejected_logps = get_model_logprobs(
                    reference_model,
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"],
                    batch["rejected_labels"],
                    tokenizer.pad_token_id,
                )

            loss, accuracy = compute_dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                config.beta,
            )

            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            epoch_loss += loss.item() * config.gradient_accumulation_steps
            epoch_accuracy += accuracy.item()
            num_batches += 1

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % config.logging_steps == 0:
                    progress.set_postfix(
                        {
                            "loss": f"{epoch_loss / num_batches:.4f}",
                            "accuracy": f"{epoch_accuracy / num_batches:.4f}",
                            "lr": f"{scheduler.get_last_lr()[0]:.6f}",
                        }
                    )

        print("Running validation...")
        val_loss, val_acc = evaluate(policy_model, reference_model, val_loader, config, tokenizer.pad_token_id)
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        print("Generating sample outputs...")
        generate_samples(policy_model, tokenizer, config.device)

    print("Training completed!")
    return policy_model
