import torch

from nanorl.algorithms.dpo.loss import compute_dpo_loss
from nanorl.algorithms.dpo.model import get_model_logprobs


def evaluate(policy_model, reference_model, dataloader, config):
    policy_model.eval()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(config.device) for k, v in batch.items()}

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
