import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np, pandas as pd
from datasets import load_dataset
from sklearn.metrics import f1_score

class BloomPPL:
    def __init__(self, device="cuda", model_id="bigscience/bloom-7b1", precision="float16"):
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

        ppl = int(torch.exp(torch.stack(nlls).sum() / end_loc))
        return ppl

model = BloomPPL()

# Load the dataset from CSV
csv_path = "/home/ubuntu/perplexity/multitude.csv"
df = pd.read_csv(csv_path)

# Function to calculate perplexities for a given label and language
def calculate_perplexities_limited(df, model, label, lang, limit=2000):
    # Filter the dataset for the specific label and language
    filtered_data = df[(df["label"] == label) & (df["language"] == lang)]

    # Limit the number of rows to the specified limit
    limited_data = filtered_data.head(limit)

    results = []
    for count, (_, row) in enumerate(limited_data.iterrows(), start=1):  # `count` starts from 1
        sentence = row["text"]
        perplexity = model(sentence)
        results.append({"Label": label, "Sentence": sentence, "Bloom_Perplexity": perplexity})
        
        if count % 500 == 0:  # Sequential count-based progress
            print(f"Processed {count} texts...")
    
    return results

# Function to find the optimal thresholds for accuracy and macro-F1
def find_optimal_thresholds(results_df):
    sorted_ppls = results_df["Bloom_Perplexity"].sort_values().unique()

    # Initialize best metrics for accuracy
    best_accuracy = 0
    best_accuracy_threshold = None

    # Initialize best metrics for macro-F1
    best_macro_f1 = 0
    best_macro_f1_threshold = None

    for threshold in sorted_ppls:
        results_df["Predicted"] = results_df["Bloom_Perplexity"].apply(lambda x: 0 if x < threshold else 1)

        # Calculate accuracy
        accuracy = (results_df["Predicted"] == results_df["Label"]).mean()

        # Calculate F1 scores
        f1_human = f1_score(results_df["Label"], results_df["Predicted"], pos_label=0)
        f1_generated = f1_score(results_df["Label"], results_df["Predicted"], pos_label=1)
        macro_f1 = (f1_human + f1_generated) / 2

        # Update best accuracy metrics
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_accuracy_threshold = threshold

        # Update best macro-F1 metrics
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_macro_f1_threshold = threshold

    return {
        "best_accuracy_threshold": best_accuracy_threshold,
        "best_accuracy": best_accuracy,
        "best_macro_f1_threshold": best_macro_f1_threshold,
        "best_macro_f1": best_macro_f1,
    }

# Main code execution
labels = [0, 1]  # 0 for human, 1 for machine-generated
all_results = []

for label in labels:
    label_name = "Human" if label == 0 else "Generated"
    print(f"\nProcessing category: {label_name} (label={label})")
    results = calculate_perplexities_limited(df, model, label, lang="ar", limit=2000)
    all_results.extend(results)

# Convert all results to a DataFrame
results_df = pd.DataFrame(all_results)

# Find the optimal thresholds and their corresponding metrics
thresholds_metrics = find_optimal_thresholds(results_df)

# Print the results
print("\nPerplexities for each text:")
print(results_df)
print("\nAverage Perplexity by Label:")
print(results_df.groupby("Label")["Bloom_Perplexity"].mean())

# Display thresholds and metrics
print(f"\nOptimal Threshold for Accuracy: {thresholds_metrics['best_accuracy_threshold']}")
print(f"Accuracy at Optimal Threshold: {thresholds_metrics['best_accuracy']:.2%}")

print(f"\nOptimal Threshold for Macro-F1: {thresholds_metrics['best_macro_f1_threshold']}")
print(f"Macro-F1 Score at Optimal Threshold: {thresholds_metrics['best_macro_f1']:.4f}")
