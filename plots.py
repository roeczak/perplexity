import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import ttest_ind
import numpy as np
import matplotlib.pyplot as plt

class BloomPPL:
    def __init__(self, device="cuda", model_id="gpt2-xl", precision="float32"):
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

# Load the dataset from Hugging Face
from datasets import load_dataset
dataset = load_dataset("Jinyan1/COLING_2025_MGT_multingual")
df = pd.DataFrame(dataset["train"])

# Function to calculate perplexities for a specific label
def calculate_perplexities(df, model, label, limit=2000, min_length=20):
    filtered_data = df[(df["label"] == label) & (df["text"].str.len() >= min_length) & (df["lang"] == "ur")]
    limited_data = filtered_data.head(limit)

    perplexities = []
    for count, text in enumerate(limited_data["text"], start=1):
        perplexity = model(text)
        perplexities.append(perplexity)

        if count % 200 == 0:  # Progress update
            print(f"Processed {count} texts for label {label}")

    return perplexities

# Calculate perplexities for human and AI texts
print("Calculating perplexities for human texts...")
human_perplexities = calculate_perplexities(df, model, label=0)

print("Calculating perplexities for generated texts...")
ai_perplexities = calculate_perplexities(df, model, label=1)

# Perform t-test
t_stat, p_value = ttest_ind(human_perplexities, ai_perplexities, equal_var=False)

# Calculate Cohen's d
mean_diff = np.mean(human_perplexities) - np.mean(ai_perplexities)
pooled_std = np.sqrt((np.std(human_perplexities, ddof=1)**2 + np.std(ai_perplexities, ddof=1)**2) / 2)
cohen_d = mean_diff / pooled_std

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")
print(f"Cohen's d: {cohen_d}")

# Plot perplexities
plt.figure(figsize=(10, 6))
plt.hist(human_perplexities, bins=np.linspace(0, 100, 50), alpha=0.5, color='green', label='Human Texts', density=True)
plt.hist(ai_perplexities, bins=np.linspace(0, 100, 50), alpha=0.5, color='red', label='AI-Generated Texts', density=True)
plt.xlabel('Perplexity')
plt.ylabel('Percentage of Texts')
plt.title('Perplexity Distribution of Human vs AI Texts')
plt.xlim(0, 100)  # Focus on 0-100 range
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("perplexity_plot2.png")
print("Plot saved as 'perplexity_plot2.png'")
