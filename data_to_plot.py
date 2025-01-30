import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Define languages with a 0-100 x-axis range
low_range_languages = {"en", "zh", "ar", "uk", "ru"}

# Folder containing the CSV files
input_folder = "/home/ubuntu/perplexity/perplexity_data/"
output_folder = "/home/ubuntu/perplexity/perplexity_plots/"
os.makedirs(output_folder, exist_ok=True)

# List all CSV files
csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

for file in csv_files:
    lang = file.replace("perplexity_", "").replace(".csv", "")
    file_path = os.path.join(input_folder, file)
    
    # Load data
    df = pd.read_csv(file_path)

    # Determine x-axis limits based on language
    x_limit = 100 if lang in low_range_languages else 300

    # Extract human and AI perplexities
    human_perplexities = df[df["label"] == 0]["perplexity"]
    ai_perplexities = df[df["label"] == 1]["perplexity"]

    # Determine min/max for binning
    ppl_min = 0
    ppl_max = min(max(human_perplexities.max(), ai_perplexities.max()), x_limit)
    
    # Plot
    plt.figure(figsize=(10, 6))
    bins = np.arange(ppl_min, ppl_max + 1, 1)
    plt.hist(human_perplexities, bins=bins, alpha=0.5, color='green', label='Human Texts', density=True)
    plt.hist(ai_perplexities, bins=bins, alpha=0.5, color='red', label='AI-Generated Texts', density=True)
    plt.xlabel('Perplexity')
    plt.ylabel('Percentage of Texts')
    plt.title(f'Perplexity Distribution for {lang}')
    plt.xlim(ppl_min, ppl_max)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save the plot
    output_file = os.path.join(output_folder, f"perplexity_plot_{lang}.png")
    plt.savefig(output_file)
    plt.close()

    print(f"Saved plot for {lang} to {output_file}")

print("All plots generated successfully.")
