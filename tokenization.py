import os
import pandas as pd
import torch
from transformers import AutoTokenizer
import glob

# Define the root folder containing all models
csv_folder = "/home/ubuntu/perplexity/perplexity_data"

# Output file
output_file = "/home/ubuntu/perplexity/tokenization_analysis.csv"

# Recursively find all CSV files
csv_files = glob.glob(os.path.join(csv_folder, "**", "perplexity_*.csv"), recursive=True)

# Store results in a list
results = []

for csv_file in csv_files:
    try:
        # Extract model name (parent directory of the CSV file)
        model_dir = os.path.dirname(csv_file).replace(csv_folder + "/", "")

        # Extract language from filename
        language = os.path.basename(csv_file).replace("perplexity_", "").replace(".csv", "")

        # Load CSV
        df = pd.read_csv(csv_file)

        # Get the first sentence (assuming it's the first row)
        if df.empty or "text" not in df.columns:
            print(f"Skipping {model_dir} - {language}: No valid text column.")
            continue
        first_sentence = df.iloc[0]["text"]

        # Load model tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
        except Exception as e:
            print(f"Skipping {model_dir} - {language} due to tokenizer error: {e}")
            continue

        # Tokenize the sentence
        tokenized = tokenizer.tokenize(first_sentence)

        # Store results
        results.append({"Model": model_dir, "Language": language, "Original Text": first_sentence, "Tokenized": " ".join(tokenized)})

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")

# Convert to DataFrame and save to CSV
df_results = pd.DataFrame(results)
df_results.to_csv(output_file, index=False)

print(f"Tokenization analysis saved to {output_file}")
