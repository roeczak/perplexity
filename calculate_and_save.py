import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

class PerplexityEvaluator:
    def __init__(self, model_id, device="cuda", precision="float16"):
        self.device = device
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if precision == "float16" else torch.float32,
            device_map="auto"
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Set max_length based on model
        self.max_length = 1024 if model_id in ["gpt2", "gpt2-xl"] else 2048
        self.stride = 512

    def get_ppl(self, sentence):
        encodings = self.tokenizer(sentence, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)

        if seq_len < 10:  # Ensure minimum length of 10 characters
            return None

        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, self.stride):
            end_loc = min(begin_loc + self.max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        return int(torch.exp(torch.stack(nlls).sum() / end_loc))

# List of models to evaluate
models = ["bigscience/bloom-7b1", "tiiuae/falcon-7b"]

# Load the dataset
csv_path = "/home/ubuntu/perplexity/multitude.csv"
df = pd.read_csv(csv_path)

# Base output directory
output_folder = "/home/ubuntu/perplexity/perplexity_data/"

# Define languages that should be limited to 5000 texts
limited_languages = {"en", "es", "ru"}  # English, Spanish, Russian

# Iterate over models and languages
for model_name in models:
    print(f"Processing model: {model_name}")
    
    # Initialize the evaluator for the current model
    evaluator = PerplexityEvaluator(model_name)

    for lang in df["language"].unique():
        print(f"Processing language: {lang}")

        # Ensure directory exists
        model_lang_folder = os.path.join(output_folder, f"{model_name}")
        os.makedirs(model_lang_folder, exist_ok=True)

        # Filter dataset for the language
        df_lang = df[df["language"] == lang]

 # Apply limit if the language is in the predefined list
        if lang in limited_languages:
            df_lang = df_lang.head(5000)  # Limit to 5000 texts

        # Calculate perplexities
        results = []
        for i, (_, row) in enumerate(df_lang.iterrows(), start=1):
            ppl = evaluator.get_ppl(row["text"])
            if ppl is not None:
                results.append({"text": row["text"], "label": row["label"], "perplexity": ppl})

 # Print progress update every 500 texts
            if i % 200 == 0:
                print(f"Processed {i} texts for {lang} using {model_name}")

        # Convert to DataFrame and save
        if results:
            output_file = os.path.join(model_lang_folder, f"perplexity_{lang}.csv")
            pd.DataFrame(results).to_csv(output_file, index=False)
            print(f"Saved {output_file}")

print("All CSV files generated successfully.")
