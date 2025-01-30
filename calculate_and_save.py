import torch
import pandas as pd
import numpy as np
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

class BloomPPL:
    def __init__(self, device="cuda", model_id="gpt2", precision="float32"):
        self.device = device
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if precision == "float16" else torch.float32,
            device_map="auto",
            offload_folder="./offload",
            low_cpu_mem_usage=True
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.max_length = 1024
        self.stride = 512

    def __call__(self, sentence):
        return self.getPPL(sentence)

    def getPPL(self, sentence):
        encodings = self.tokenizer(sentence, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)

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

        ppl = int(torch.exp(torch.stack(nlls).sum() / end_loc))
        return ppl

# Initialize the model
model = BloomPPL()

# Load the dataset
csv_path = "/home/ubuntu/perplexity/multitude.csv"
df = pd.read_csv(csv_path)

# Create output folder
output_folder = "/home/ubuntu/perplexity/perplexity_data/"
os.makedirs(output_folder, exist_ok=True)

# Function to calculate perplexities
def calculate_perplexities_limited(df, model, label, lang, limit=3000, min_length=10):
    filtered_data = df[(df["label"] == label) & (df["language"] == lang) & (df["text"].str.len() >= min_length)]
    limited_data = filtered_data.head(min(limit, len(filtered_data)))

    results = []
    for count, (_, row) in enumerate(limited_data.iterrows(), start=1):
        sentence = row["text"]
        perplexity = model(sentence)
        results.append({"text": sentence, "label": label, "perplexity": perplexity})

        if count % 500 == 0:
            print(f"Processed {count} texts for language: {lang}, label: {label}")

    return results

# Process all languages
languages = df["language"].unique()

for lang in languages:
    print(f"Processing language: {lang}")

    human_data = calculate_perplexities_limited(df, model, label=0, lang=lang)
    ai_data = calculate_perplexities_limited(df, model, label=1, lang=lang)

    # Combine and save data
    combined_data = pd.DataFrame(human_data + ai_data)
    output_file = os.path.join(output_folder, f"perplexity_{lang}.csv")
    combined_data.to_csv(output_file, index=False)

    print(f"Saved perplexities for {lang} to {output_file}")

print("All perplexity calculations completed.")
