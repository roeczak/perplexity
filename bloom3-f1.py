import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np, pandas as pd
from datasets import load_dataset
from sklearn.metrics import f1_score

class BloomPPL:
    def __init__(self, device="cuda", model_id="gpt2", precision="float32"):
        self.device = device
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if precision == "float16" else torch.float32,  # Using FP16 for memory efficiency
            device_map="auto",  # Automatically offload layers to CPU if needed
            offload_folder="./offload",  # Optionally store layers offload data on disk
            low_cpu_mem_usage=True  # To handle models without requiring too much CPU RAM
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

model = BloomPPL()

dataset = load_dataset("Jinyan1/COLING_2025_MGT_multingual")

# Function to calculate perplexities for the dataset with a limit and text length filtering
def calculate_perplexities_limited(dataset, model, label, lang, limit=5000, min_length=10):
    filtered_data = dataset["train"].filter(lambda x: x["lang"] == lang and x["label"] == label and len(x["text"]) >= min_length)
    filtered_data = filtered_data.select(range(min(limit, len(filtered_data))))

    results = []
    for i, entry in enumerate(filtered_data):
        sentence = entry["text"]
        perplexity = model(sentence)
        results.append({"Label": label, "Sentence": sentence, "Bloom_Perplexity": perplexity})

        if (i + 1) % 200 == 0:
            print(f"Processed {i + 1} texts for label {label}...")

    return results

# Function to find the optimal threshold and accuracy
# Function to find the optimal threshold using macro-F1 score
def find_optimal_threshold_macro_f1(results_df):
    sorted_ppls = results_df["Bloom_Perplexity"].sort_values().unique()

    best_threshold = None
    best_macro_f1 = 0

    for threshold in sorted_ppls:
        results_df["Predicted"] = results_df["Bloom_Perplexity"].apply(lambda x: 0 if x > threshold else 1)

        # Calculate F1 scores for each label
        f1_human = f1_score(results_df["Label"], results_df["Predicted"], pos_label=0)
        f1_generated = f1_score(results_df["Label"], results_df["Predicted"], pos_label=1)

        # Compute macro-F1 score
        macro_f1 = (f1_human + f1_generated) / 2

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold = threshold

    return best_threshold, best_macro_f1

# Main code execution
labels = [0, 1]  # 0 for human, 1 for machine-generated
all_results = []

for label in labels:
    label_name = "Human" if label == 0 else "Generated"
    print(f"\nProcessing category: {label_name} (label={label})")
    results = calculate_perplexities_limited(dataset, model, label, lang="ar", limit=5000, min_length=20)
    all_results.extend(results)

# Convert all results to a DataFrame
results_df = pd.DataFrame(all_results)

# Find the optimal threshold using macro-F1 score
optimal_threshold, macro_f1 = find_optimal_threshold_macro_f1(results_df)

# Print the results
print("\nPerplexities for each text:")
print(results_df)
print("\nAverage Perplexity by Label:")
print(results_df.groupby("Label")["Bloom_Perplexity"].mean())
print(f"\nOptimal Threshold: {optimal_threshold}")
print(f"Macro-F1 Score at Optimal Threshold: {macro_f1:.4f}")
