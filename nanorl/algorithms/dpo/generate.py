import torch


def generate_samples(model, tokenizer, device, num_samples=3):
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
