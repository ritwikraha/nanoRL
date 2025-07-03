import csv
import io
import json
import os
import random
import time

import pandas as pd
import google.generativeai as genai



# To run this script, you need to set up the Google Generative AI API.
# Do this by setting the environment variable `GEMINI_API_KEY` with your
# Google Generative AI API key, or you can hardcode it for testing purposes.
# Uncomment the following lines to hardcode the API key for testing.

# GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
# if not GEMINI_API_KEY:
#     raise ValueError("GEMINI_API_KEY environment variable not set. Please set it or hardcode for testing.")


# genai.configure(api_key=GEMINI_API_KEY)
# model = genai.GenerativeModel('gemini-2.5-flash')

def generate_preference_data(num_samples=2000):
    """
    Generates a dataset of verbal reasoning prompts, then creates "chosen" (correct)
    and "rejected" (incorrect) responses for each, with both answers and reasoning.

    Args:
        num_samples (int): The total number of samples to generate.

    Returns:
        str: A CSV formatted string containing the generated data.
             Returns an empty string if no data is generated.
    """
    print(f"INFO: Starting data generation for {num_samples} samples...")

    # topics designed to create verbal reasoning questions
    topics = [
        "a logical analogy problem (e.g., 'Leaf is to Tree as Page is to X')",
        "a short scenario requiring a logical deduction to find the outcome",
        "identifying the unstated, underlying assumption in a given argument",
        "determining the most logical cause for a specifically described effect",
        "finding the word that doesn't belong in a group based on a shared category",
        "evaluating a simple argument to pinpoint its primary logical flaw",
        "drawing a single, valid conclusion from a short passage of text",
        "completing a logical sequence of related words or concepts",
        "identifying the specific relationship between two words (like cause/effect)",
        "solving a simple syllogism to determine if the conclusion is valid"
    ]

    generated_data = []
    while len(generated_data) < num_samples:
        try:
            topic = pd.Series(topics).sample(1).iloc[0]

            # refined prompt to get ONLY the question text from the model
            prompt_instruction = f"Generate a single, clear question that is an example of the following verbal reasoning task: {topic}. Output nothing but the question itself, without any labels, formatting, or introductory text."
            
            question_response = model.generate_content(prompt_instruction)
            prompt = question_response.text.strip()

            # prompt to generate the chosen/rejected pair with 'answer' and 'reasoning'
            generation_prompt = f"""
            For the verbal reasoning question: "{prompt}"

            Please generate two distinct answers, each with an 'answer' and 'reasoning' field:
            1.  A logically sound and well-explained correct answer.
            2.  A plausible-sounding but logically flawed or incorrect answer.

            Format your response as a single, clean JSON object with two keys: "chosen" and "rejected".
            Each value (for "chosen" and "rejected") should itself be a JSON object with two keys: "answer" and "reasoning".

            Example:
            {{
                "chosen": {{
                    "answer": "Correct answer text.",
                    "reasoning": "Detailed explanation of why this answer is correct."
                }},
                "rejected": {{
                    "answer": "Incorrect answer text.",
                    "reasoning": "Detailed explanation of why this answer is flawed or incorrect."
                }}
            }}
            """

            response = model.generate_content(generation_prompt)
            # the model sometimes wraps its output in ```json ... ```, this removes it.
            cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
            
            response_json = json.loads(cleaned_response)
            if "chosen" in response_json and "rejected" in response_json:
                # Ensure the 'chosen' and 'rejected' values are dictionaries with 'answer' and 'reasoning'
                if isinstance(response_json["chosen"], dict) and \
                   "answer" in response_json["chosen"] and "reasoning" in response_json["chosen"] and \
                   isinstance(response_json["rejected"], dict) and \
                   "answer" in response_json["rejected"] and "reasoning" in response_json["rejected"]:
                    
                    generated_data.append({
                        "prompt": prompt,
                        "chosen": response_json["chosen"],
                        "rejected": response_json["rejected"],
                    })
                    print(f"SUCCESS: Generated sample {len(generated_data)}/{num_samples}")
                else:
                    print("WARNING: Skipping sample due to 'chosen' or 'rejected' not having expected 'answer'/'reasoning' format.")
            else:
                print("WARNING: Skipping sample due to missing 'chosen' or 'rejected' key.")

        except json.JSONDecodeError as e:
            print(f"WARNING: Skipping sample due to a JSON decoding error: {e}. Raw response: {cleaned_response}")
        except Exception as e:
            print(f"ERROR: An unexpected error occurred: {e}. Retrying in 5 seconds...")
            time.sleep(5)

    if not generated_data:
        print("INFO: No data was generated.")
        return ""

    # convert the list of dictionaries to a pandas DataFrame and then to a CSV string
    df = pd.DataFrame(generated_data)
    print("\nINFO: Data generation complete. Converting to CSV format.")
    return df.to_csv(index=False)

# usage example:
# generated_data_csv = generate_preference_data(num_samples=5) # reduced for quick testing
