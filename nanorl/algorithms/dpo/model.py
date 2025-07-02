import torch
import torch.nn.functional as F


def get_model_logprobs(model, input_ids, attention_mask, labels):
    with torch.cuda.amp.autocast():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        gathered_log_probs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        mask = (shift_labels != -100) & (shift_labels != model.config.pad_token_id)
        sequence_logprobs = (gathered_log_probs * mask).sum(dim=-1)

    return sequence_logprobs
