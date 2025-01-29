import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import f1_score

# Define the LlamaPPL model class
class LlamaPPL:
    def __init__(self, device="cuda", model_id="facebook/opt-2.7b"):
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

# Load the dataset from CSV
csv_path = "/home/ec2-user/multitude2.csv"
df = pd.read_csv(csv_path)

# Initialize the perplexity model
model = LlamaPPL()

# Function to calculate perplexities for a given label and language
def calculate_perplexities_limited(df, model, label, lang="zh", limit=2000):
    # Filter the dataset for the specific label and language
    filtered_data = df[(df["label"] == label) & (df["language"] == lang)]

    # Limit the number of rows to the specified limit
    limited_data = filtered_data.head(limit)

    results = []
    for count, (_, row) in enumerate(limited_data.iterrows(), start=1):  # `count` starts from 1
        sentence = row["text"]
        perplexity = model(sentence)
        results.append({"Label": label, "Sentence": sentence, "Llama_Perplexity": perplexity})

        if count % 500 == 0:  # Sequential count-based progress
            print(f"Processed {count} texts...")

    return results

# Function to find the optimal threshold using macro-F1 score
def find_optimal_threshold_macro_f1(results_df):
    sorted_ppls = results_df["Llama_Perplexity"].sort_values().unique()

    best_threshold = None
    best_macro_f1 = 0

    for threshold in sorted_ppls:
        results_df["Predicted"] = results_df["Llama_Perplexity"].apply(lambda x: 0 if x > threshold else 1)

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
    results = calculate_perplexities_limited(df, model, label, lang="es", limit=2000)
    all_results.extend(results)

# Convert all results to a DataFrame
results_df = pd.DataFrame(all_results)

# Find the optimal threshold using macro-F1 score
optimal_threshold, macro_f1 = find_optimal_threshold_macro_f1(results_df)

# Print the results
print("\nPerplexities for each text:")
print(results_df)
print("\nAverage Perplexity by Label:")
print(results_df.groupby("Label")["Llama_Perplexity"].mean())
print(f"\nOptimal Threshold: {optimal_threshold}")
print(f"Macro-F1 Score at Optimal Threshold: {macro_f1:.4f}")
