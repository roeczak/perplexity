import os
import pandas as pd
import glob
from transformers import AutoTokenizer

# Define paths
csv_folder = "/home/ubuntu/perplexity/perplexity_data_coling_test"  # Change this if needed
output_csv = "/home/ubuntu/perplexity/average_token_lengths_3.csv"
num_words = 100  # Number of words to take from each text

# List to store results
results = []

# Get all CSV files recursively
csv_files = glob.glob(os.path.join(csv_folder, "**", "perplexity_*.csv"), recursive=True)

for csv_path in csv_files:
    # Extract model name and language
    parts = csv_path.replace(csv_folder, "").strip(os.sep).split(os.sep)
    
    if len(parts) == 2:
        model, filename = parts
    else:
        model = "/".join(parts[:-1])  # Handle nested model names
        filename = parts[-1]
    
    lang = filename.replace("perplexity_", "").replace(".csv", "")

    # Load tokenizer (ensure correct model formatting)
    model_id = model.replace("_", "/")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load CSV
    df = pd.read_csv(csv_path)

    if df.empty:
        continue  # Skip empty files
    
    # Extract first non-null text
    first_text = df["text"].dropna().iloc[0]

    # Limit to the first few words
    first_words = " ".join(first_text.split()[:num_words])

    # Tokenize
    tokens = tokenizer.tokenize(first_words)
    
    # Compute average token length
    if tokens:
        avg_token_length = sum(len(token) for token in tokens) / len(tokens)
    else:
        avg_token_length = 0

    # Store result
    results.append({"model": model, "language": lang, "average_token_length": avg_token_length})

# Convert results to DataFrame and save as CSV
df_results = pd.DataFrame(results)
df_results.to_csv(output_csv, index=False)

print(f"Average token lengths saved to {output_csv}")
