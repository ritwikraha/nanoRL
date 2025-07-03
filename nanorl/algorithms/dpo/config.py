from dataclasses import dataclass

import torch


@dataclass
class TrainingConfig:
    model_name: str = "gpt2"
    learning_rate: float = 5e-5
    beta: float = 0.1
    batch_size: int = 4
    num_epochs: int = 3
    max_length: int = 128
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    logging_steps: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
