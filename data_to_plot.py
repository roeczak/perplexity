import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the folder where the CSV files are stored
csv_folder = "/home/ubuntu/perplexity/perplexity_data/"  # Update this if needed
output_folder = "/home/ubuntu/perplexity/perplexity_plots/"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get all CSV files
csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

for file in csv_files:
    file_path = os.path.join(csv_folder, file)
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Extract language from filename (assuming filenames are like "perplexity_en.csv")
    language = file.replace("perplexity_", "").replace(".csv", "")

    # Separate human and AI-generated perplexities
    human_perplexities = df[df["label"] == 0]["perplexity"].tolist()
    ai_perplexities = df[df["label"] == 1]["perplexity"].tolist()

    if not human_perplexities or not ai_perplexities:
        print(f"Skipping {language} due to insufficient data.")
        continue

    # Determine x-axis limits dynamically
    ppl_min = min(min(human_perplexities), min(ai_perplexities))
    ppl_max = max(max(human_perplexities), max(ai_perplexities))
    margin = (ppl_max - ppl_min) * 0.05  # 5% margin

    # Plot perplexities
    plt.figure(figsize=(10, 6))
    bins = np.arange(ppl_min, ppl_max + 1, 1)  # Bins of size 1
    plt.hist(human_perplexities, bins=bins, alpha=0.5, color='green', label='Human Texts', density=True)
    plt.hist(ai_perplexities, bins=bins, alpha=0.5, color='red', label='AI-Generated Texts', density=True)
    plt.xlabel('Perplexity')
    plt.ylabel('Percentage of Texts')
    plt.title(f'Perplexity Distribution of Human vs AI Texts ({language})')
    plt.xlim(ppl_min - margin, ppl_max + margin)  # Adjust x-axis
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save plot
    plot_path = os.path.join(output_folder, f"perplexity_plot_{language}.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved plot for {language} as {plot_path}")

print("All plots generated successfully.")
