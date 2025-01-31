import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Base directories
csv_base_dir = "/home/ubuntu/perplexity/perplexity_data/"
plot_base_dir = "/home/ubuntu/perplexity/perplexity_plots/"

# Ensure the plot directory exists
os.makedirs(plot_base_dir, exist_ok=True)

# Define zoom levels for specific languages
zoom_100_languages = {"en", "zh", "ar", "uk", "ru"}  # English, Chinese, Arabic, Ukrainian, Russian

# Iterate over main directories in perplexity_data (these could be organizations or models)
for main_dir in os.listdir(csv_base_dir):
    main_path = os.path.join(csv_base_dir, main_dir)

    if not os.path.isdir(main_path):  # Skip non-directory files
        continue

    # Check if the main_path contains actual CSVs (single model case) or subdirectories (organization case)
    potential_csvs = [f for f in os.listdir(main_path) if f.endswith(".csv")]
    if potential_csvs:
        # This means main_dir is actually a model (e.g., "gpt2", "gpt2-xl") not an organization like "facebook"
        model_name = main_dir
        model_folder = main_path
        model_plot_folder = os.path.join(plot_base_dir, model_name.replace("/", "_"))
        os.makedirs(model_plot_folder, exist_ok=True)
        model_subdirs = [model_folder]  # Treat as a list to reuse logic
    else:
        # If no CSVs are directly in the folder, it's an organization containing models (e.g., "facebook/opt-125m")
        model_subdirs = [os.path.join(main_path, sub) for sub in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, sub))]
    
    # Iterate over all detected model directories
    for model_folder in model_subdirs:
        model_name = os.path.relpath(model_folder, csv_base_dir).replace("/", "_")  # Ensure unique model names
        model_plot_folder = os.path.join(plot_base_dir, model_name)
        os.makedirs(model_plot_folder, exist_ok=True)  # Ensure the plot directory exists

        # Iterate over language-specific CSV files in the model folder
        for filename in os.listdir(model_folder):
            if not filename.endswith(".csv"):
                continue

            # Extract language from filename
            lang = filename.split("_")[1].split(".")[0]
            csv_path = os.path.join(model_folder, filename)

            print(f"Generating plot for {lang} using {model_name}")

            # Load the dataset
            df = pd.read_csv(csv_path)

            # Separate human and AI-generated texts
            human_perplexities = df[df["label"] == 0]["perplexity"].values
            ai_perplexities = df[df["label"] == 1]["perplexity"].values

            # Determine plot limits based on language
            x_max = 100 if lang in zoom_100_languages else 300
            bins = np.arange(0, x_max + 1, 1)  # Bin size of 1

            # Plot perplexities
            plt.figure(figsize=(10, 6))
            plt.hist(human_perplexities, bins=bins, alpha=0.5, color='green', label='Human Texts', density=True)
            plt.hist(ai_perplexities, bins=bins, alpha=0.5, color='red', label='AI-Generated Texts', density=True)
            plt.xlabel('Perplexity')
            plt.ylabel('Percentage of Texts')
            plt.title(f'Perplexity Distribution ({model_name} - {lang})')
            plt.xlim(0, x_max)  # Set dynamic x-axis limits
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)

            # Save the plot with a safe filename
            plot_filename = os.path.join(model_plot_folder, f"perplexity_{lang}.png")
            plt.savefig(plot_filename)
            plt.close()

            print(f"Saved plot: {plot_filename}")

print("All plots generated successfully.")
