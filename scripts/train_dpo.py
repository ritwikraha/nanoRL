from nanorl.algorithms.dpo.config import TrainingConfig
from nanorl.algorithms.dpo.train import train_dpo

if __name__ == "__main__":
    config = TrainingConfig()
    model = train_dpo(config)
    model.push_to_hub("dpo_gpt2_model")
