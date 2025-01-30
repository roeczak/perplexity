import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import ttest_ind
import numpy as np
from datasets import load_dataset

class GPT2PPL:
    def __init__(self, device="cuda", model_id="tiiuae/falcon-7b", precision="float16"):
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

        self.max_length = 2048
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

        ppl = round(torch.exp(torch.stack(nlls).sum() / end_loc).item(), 2)
        return ppl

model = GPT2PPL()

# Load the dataset from CSV
csv_path = "/home/ubuntu/perplexity/multitude.csv"
df = pd.read_csv(csv_path)

# Function to calculate perplexities for a specific label and language
def calculate_perplexities_limited(df, model, label, lang, limit=2000):
    # Filter the dataset for the specific label and language
    filtered_data = df[(df["label"] == label) & (df["language"] == lang)]

    # Limit the number of rows to the specified limit
    limited_data = filtered_data.head(limit)

    results = []
    for count, (_, row) in enumerate(limited_data.iterrows(), start=1):
        sentence = row["text"]
        perplexity = model(sentence)
        results.append(perplexity)

        if count % 500 == 0:  # Progress update
            print(f"Processed {count} texts for language: {lang}, label: {label}")

    return results

# List of languages in the dataset
languages = ["ar","zh","cs"] 
#df["language"].unique()

# Dictionary to store results for all languages
language_results = {}

# Iterate over each language
for lang in languages:
    print(f"Processing language: {lang}")

    # Calculate perplexities for human and AI texts
    human_perplexities = calculate_perplexities_limited(df, model, label=0, lang=lang)
    ai_perplexities = calculate_perplexities_limited(df, model, label=1, lang=lang)

    # Perform t-test
    t_stat, p_value = ttest_ind(human_perplexities, ai_perplexities, equal_var=False)

    # Calculate Cohen's d
    mean_diff = np.mean(human_perplexities) - np.mean(ai_perplexities)
    pooled_std = np.sqrt((np.std(human_perplexities, ddof=1)**2 + np.std(ai_perplexities, ddof=1)**2) / 2)
    cohen_d = mean_diff / pooled_std

    # Store results
    language_results[lang] = {
        "t_stat": t_stat,
        "p_value": p_value,
        "cohen_d": cohen_d,
        "human_perplexities": human_perplexities,
        "ai_perplexities": ai_perplexities,
    }

    print(f"Language: {lang}")
    print(f"  T-statistic: {t_stat}")
    print(f"  P-value: {p_value}")
    print(f"  Cohen's d: {cohen_d}\n")

# Summarize results
summary = pd.DataFrame([
    {
        "Language": lang,
        "T-statistic": res["t_stat"],
        "P-value": res["p_value"],
        "Cohen's d": res["cohen_d"]
    }
    for lang, res in language_results.items()
])

print("Summary of Results:")
print(summary)

# Optionally save the summary to a CSV file
summary.to_csv("/home/ubuntu/perplexity/perplexity_ttest_summary_2.csv", index=False)
