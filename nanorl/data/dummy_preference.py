import random


def generate_dummy_preference_data(num_samples=1000):
    templates = [
        {
            "prompt": "The capital of France is",
            "chosen": " Paris, which is known for the Eiffel Tower and its rich cultural heritage.", # factualy correct
            "rejected": " London, which is famous for Big Ben and the Thames River.", # factually incorrect
        },
    ]
    data = []
    variations = [
        "",
        "In my opinion, ",
        "I think that ",
        "It's clear that ",
        "Generally speaking, ",
    ]

    for _ in range(num_samples):
        template = random.choice(templates)
        prefix = random.choice(variations)
        data.append(
            {
                "prompt": prefix + template["prompt"],
                "chosen": template["chosen"],
                "rejected": template["rejected"],
            }
        )

    return data
