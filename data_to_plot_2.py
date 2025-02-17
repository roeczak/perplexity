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
def get_zoom_level(language, model):
    zoom_levels = {
        "ar": {"default": 20, "exceptions": [
            (100, ["utter-project/EuroLLM-1.7B", "ai-forever/mGPT", "meta-llama/Llama-3.2-3B", "bigscience/bloom-7b1"])
        ]},
        "ca": {"default": 300, "exceptions": [
            (50, ["utter-project/EuroLLM-1.7B", "ai-forever/mGPT", "facebook/opt-2.7b", "meta-llama/Llama-3.2-3B"]),
            (100, ["bigscience/bloom-7b1"])
        ]},
        "cs": {"default": 150, "exceptions": [
            (300, ["gpt2"]),
            (50, ["ai-forever/mGPT", "facebook/opt-2.7b"])
        ]},
        "de": {"default": 100, "exceptions": [
            (200, ["gpt2"]),
            (50, ["facebook/opt-2.7b", "meta-llama/Llama-3.2-3B"])
        ]},
        "en": {"default": 100},
        "es": {"default": 50, "exceptions": [
            (200, ["gpt2"]),
            (100, ["gpt2-xl"])
        ]},
        "nl": {"default": 150, "exceptions": [
            (300, ["gpt2"]),
            (50, ["facebook/opt-2.7b", "meta-llama/Llama-3.2-3B"])
        ]},
        "pt": {"default": 100, "exceptions": [
            (300, ["gpt2"]),
            (50, ["facebook/opt-2.7b", "meta-llama/Llama-3.2-3B"])
        ]},
        "ru": {"default": 20, "exceptions": [
            (40, ["utter-project/EuroLLM-1.7B", "ai-forever/mGPT", "meta-llama/Llama-3.2-3B"]),
            (40, ["bigscience/bloom-7b1"])
        ]},
        "uk": {"default": 20, "exceptions": [
            (100, ["utter-project/EuroLLM-1.7B", "ai-forever/mGPT", "meta-llama/Llama-3.2-3B"]),
            (60, ["bigscience/bloom-7b1"])
        ]},
        "zh": {"default": 60, "exceptions": [
            (100, ["ai-forever/mGPT", "meta-llama/Llama-3.2-3B", "bigscience/bloom-7b1"])
        ]},
	"id": {"default": 50, "exceptions": [
            (100, ["utter-project/EuroLLM-1.7B", "EleutherAI/gpt-neo-125m", "facebook/opt-125m"]),
	    (300, ["gpt2"]), (150, ["gpt2-xl"])
        ]},
	"bg": {"default": 20, "exceptions": [
            (50, ["utter-project/EuroLLM-1.7B", "ai-forever/mGPT", "meta-llama/Llama-3.2-3B"]),
            (50, ["bigscience/bloom-7b1"])
	]},
	"ur": {"default": 20, "exceptions": [
            (50, ["utter-project/EuroLLM-1.7B", "ai-forever/mGPT"]),
            (50, ["bigscience/bloom-7b1"])
	]}
    }
    
    if language in zoom_levels:
        for zoom, models in zoom_levels[language].get("exceptions", []):
            if model in models:
                return zoom
        return zoom_levels[language]["default"]
    
    return 300  # Default for unlisted languages

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
            x_max = get_zoom_level(lang,model_name.replace("_", "/"))
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
