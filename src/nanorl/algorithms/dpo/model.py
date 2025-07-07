import torch
import torch.nn.functional as F


def get_model_logprobs(model, input_ids, attention_mask, labels, pad_token_id):
    with torch.cuda.amp.autocast():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, T, V)

        # shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
        shift_labels = labels[:, 1:].contiguous()  # (B, T-1)

        # compute mask first
        mask = (shift_labels != -100) & (shift_labels != model.config.pad_token_id)

        # replace all -100 with pad_token_id so gather index is in-bounds
        safe_labels = shift_labels.masked_fill(shift_labels == -100, pad_token_id)

        # log-probs over vocab
        log_probs = F.log_softmax(shift_logits, dim=-1)  # (B, T-1, V)

        # pick out each true-token log-prob
        gathered = torch.gather(
            log_probs, dim=-1, index=safe_labels.unsqueeze(-1)
        ).squeeze(-1)  # (B, T-1)

        # zero out padded/ignored positions, then sum
        sequence_logprobs = (gathered * mask).sum(dim=-1)  # (B,)

    return sequence_logprobs
