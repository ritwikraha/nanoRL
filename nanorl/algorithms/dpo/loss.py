import torch.nn.functional as F


def compute_dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta=0.1,
):
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps

    logits = beta * (policy_logratios - reference_logratios)
    loss = -F.logsigmoid(logits).mean()
    accuracy = (logits > 0).float().mean()

    return loss, accuracy
