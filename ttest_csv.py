import pandas as pd
from scipy.stats import ttest_ind
import os

# Define paths
csv_folder = "/home/ubuntu/perplexity/perplexity_data"  # Change to your actual folder
model = "meta-llama/Llama-3.2-3B"  # Change to the model you want to analyze
language = "de"  # Change to the language you want to test

csv_path = os.path.join(csv_folder, f"{model}/perplexity_{language}.csv")

# Load CSV
df = pd.read_csv(csv_path)

# Extract perplexities for human and AI-generated texts
human_ppl = df[df["label"] == 0]["perplexity"].dropna()
ai_ppl = df[df["label"] == 1]["perplexity"].dropna()

# Run t-test
t_stat, p_value = ttest_ind(human_ppl, ai_ppl, equal_var=False)

# Print results
print(f"T-test results for {model} - {language} (Human vs. AI):")
print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")

# Interpretation:
if p_value < 0.05:
    print("❌ Significant difference between human and AI perplexity distributions!")
else:
    print("✅ No significant difference between human and AI perplexity distributions.")
