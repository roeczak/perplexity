import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np, pandas as pd 
from scipy.stats import ttest_ind
from datasets import load_dataset

class GPT2PPL:
    def __init__(self, device="cuda", model_id="gpt2"):
        self.device = device
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.max_length = self.model.config.max_position_embeddings
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

model = GPT2PPL()

# Load the dataset from Hugging Face
dataset = load_dataset("akoukas/HC3_v2")

# Function to calculate perplexities for a dataset
def calculate_perplexities(dataset, model, label, limit=1000, min_length=10):
    # Filter data by label and minimum text length
    filtered_data = dataset.filter(lambda x: x["label"] == label and len(x["text"]) >= min_length )
    
    # Shuffle and limit the number of samples
    limited_data = filtered_data.shuffle(seed=42).select(range(min(limit, len(filtered_data))))
    
    results = []
    for count, entry in enumerate(limited_data, start=1):
        sentence = entry["text"]
        perplexity = model(sentence)
        results.append(perplexity)

        if count % 200 == 0:  # Progress update
            print(f"Processed {count} texts for label {label}")

    return results

# Calculate perplexities for human and AI texts
print("Calculating perplexities for human texts...")
human_perplexities = calculate_perplexities(dataset["train"], model, label=0)

print("Calculating perplexities for AI texts...")
ai_perplexities = calculate_perplexities(dataset["train"], model, label=1)

# Perform t-test
t_stat, p_value = ttest_ind(human_perplexities, ai_perplexities, equal_var=False)

# Calculate Cohen's d
mean_diff = np.mean(human_perplexities) - np.mean(ai_perplexities)
pooled_std = np.sqrt((np.std(human_perplexities, ddof=1)**2 + np.std(ai_perplexities, ddof=1)**2) / 2)
cohen_d = mean_diff / pooled_std

# Output results
print("\nT-test Results:")
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")
print(f"Cohen's d: {cohen_d}")

# Save results
results = pd.DataFrame({
    "T-statistic": [t_stat],
    "P-value": [p_value],
    "Cohen's d": [cohen_d]
})
results.to_csv("/home/ec2-user/monolingual_ttest_results.csv", index=False)
