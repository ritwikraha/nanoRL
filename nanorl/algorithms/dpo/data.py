from typing import Dict, List

from torch.utils.data import Dataset


class DPODataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def dpo_collate_fn(batch: List[Dict], tokenizer, max_length: int = 128):
    prompts = [item["prompt"] for item in batch]
    chosen_responses = [item["chosen"] for item in batch]
    rejected_responses = [item["rejected"] for item in batch]

    chosen_texts = [p + c for p, c in zip(prompts, chosen_responses)]
    rejected_texts = [p + r for p, r in zip(prompts, rejected_responses)]

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

    prompt_tokens = tokenizer(
        prompts, padding=False, truncation=True, add_special_tokens=True
    )
    chosen_labels = chosen_tokens["input_ids"].clone()
    rejected_labels = rejected_tokens["input_ids"].clone()

    for i, prompt_length in enumerate([len(p) for p in prompt_tokens["input_ids"]]):
        chosen_labels[i, :prompt_length] = -100
        rejected_labels[i, :prompt_length] = -100

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
