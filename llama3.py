import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define the LlamaPPL model class
class LlamaPPL:
    def __init__(self, device="cuda", model_id="meta-llama/Llama-3.2-3B"):
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

        ppl = round(torch.exp(torch.stack(nlls).sum() / end_loc).item(), 2)
        return ppl

# Load the dataset
dataset = load_dataset("Jinyan1/COLING_2025_MGT_multingual")

# Initialize the perplexity model
model = LlamaPPL()

# Function to calculate perplexities for the entire dataset
def calculate_perplexities_full(dataset, model, label, lang):
    filtered_data = dataset["train"].filter(lambda x: x["lang"] == lang and x["label"] == label)
    
    results = []
    for i, entry in enumerate(filtered_data):
        sentence = entry["text"]
        perplexity = model(sentence)
        results.append({"Label": label, "Sentence": sentence, "Llama_Perplexity": perplexity})
        
        if (i + 1) % 200 == 0:
            print(f"Processed {i + 1} texts...")
    
    return results

# Function to find the optimal threshold and accuracy
def find_optimal_threshold(results_df):
    sorted_ppls = results_df["Llama_Perplexity"].sort_values().unique()
    
    best_threshold = None
    best_accuracy = 0
    
    for threshold in sorted_ppls:
        results_df["Predicted"] = results_df["Llama_Perplexity"].apply(lambda x: 0 if x > threshold else 1)
        correct_predictions = (results_df["Predicted"] == results_df["Label"]).sum()
        accuracy = correct_predictions / len(results_df)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    return best_threshold, best_accuracy

# Main code execution
labels = [0, 1]  # 0 for human, 1 for machine-generated
all_results = []

for label in labels:
    label_name = "Human" if label == 0 else "Generated"
    print(f"\nProcessing category: {label_name} (label={label})")
    results = calculate_perplexities_full(dataset, model, label, lang="zh")
    all_results.extend(results)

# Convert all results to a DataFrame
results_df = pd.DataFrame(all_results)

# Find the optimal threshold
optimal_threshold, accuracy = find_optimal_threshold(results_df)

# Print the results
print("\nPerplexities for each text:")
print(results_df)
print("\nAverage Perplexity by Label:")
print(results_df.groupby("Label")["Llama_Perplexity"].mean())
print(f"\nOptimal Threshold: {optimal_threshold}")
print(f"Accuracy at Optimal Threshold: {accuracy:.2f}")
