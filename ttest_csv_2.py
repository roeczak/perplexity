import pandas as pd
from scipy.stats import ttest_ind
import os
import glob

# Define the root folder containing all models
csv_folder = "/home/ubuntu/perplexity/perplexity_data_coling_test"

# Recursively find all CSV files
csv_files = glob.glob(os.path.join(csv_folder, "**", "perplexity_*.csv"), recursive=True)

for csv_file in csv_files:
    try:
        # Extract model name (parent directory of the CSV file)
        model_dir = os.path.dirname(csv_file).replace(csv_folder + "/", "")

        # Extract language from filename
        language = os.path.basename(csv_file).replace("perplexity_", "").replace(".csv", "")

        # Load CSV
        df = pd.read_csv(csv_file)

        # Extract perplexities for human and AI-generated texts
        human_ppl = df[df["label"] == 0]["perplexity"].dropna()
        ai_ppl = df[df["label"] == 1]["perplexity"].dropna()

        # Ensure both groups have data
        if human_ppl.empty or ai_ppl.empty:
            print(f"Skipping {model_dir} - {language}: One or both distributions are empty.")
            continue

        # Run t-test
        t_stat, p_value = ttest_ind(human_ppl, ai_ppl, equal_var=False)

        # Print results
        print(f"T-test results for {model_dir} - {language}:")
        print(f"T-statistic: {t_stat}, P-value: {p_value}\n")

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
