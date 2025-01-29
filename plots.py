import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import ttest_ind
import numpy as np
import matplotlib.pyplot as plt

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

# Load the dataset from Hugging Face
from datasets import load_dataset
dataset = load_dataset("Jinyan1/COLING_2025_MGT_multingual")
df = pd.DataFrame(dataset["train"])

# Function to calculate perplexities for a specific label
def calculate_perplexities(df, model, label, limit=2000, min_length=10):
    filtered_data = df[(df["label"] == label) & (df["text"].str.len() >= min_length) & (df["lang"] == "en")]
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

# Function to plot perplexity distributions
def plot_perplexity_distributions(human_ppl, ai_ppl, bin_width=5, output_file="perplexity_plot.png"):
    max_ppl = max(max(human_ppl), max(ai_ppl))
    bins = np.arange(0, max_ppl + bin_width, bin_width)

    # Calculate percentage distribution
    human_hist, _ = np.histogram(human_ppl, bins=bins)
    ai_hist, _ = np.histogram(ai_ppl, bins=bins)

    human_percentages = 100 * human_hist / len(human_ppl)
    ai_percentages = 100 * ai_hist / len(ai_ppl)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(bins[:-1], human_percentages, width=bin_width, color="green", alpha=0.6, label="Human")
    plt.bar(bins[:-1], ai_percentages, width=bin_width, color="red", alpha=0.6, label="Generated", align="edge")
    plt.xlabel("Perplexity")
    plt.ylabel("Percentage of Texts (%)")
    plt.title("Perplexity Distribution of Human and Generated Texts")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save the plot as an image
    plt.savefig(output_file)
    print(f"Plot saved as {output_file}")


#Plot the distributions
plot_perplexity_distributions(human_perplexities, ai_perplexities)
