import pandas as pd
import scipy.stats as stats
from scipy.stats import wasserstein_distance

def compare_perplexity_across_models(csv1, csv2):
    # Load data
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    # Debugging step: Check columns
    print(f"Columns in {csv1}: {df1.columns}")
    print(f"Columns in {csv2}: {df2.columns}")

    # Ensure required columns exist
    required_cols = {"perplexity", "label"}
    if not required_cols.issubset(df1.columns) or not required_cols.issubset(df2.columns):
        raise ValueError(f"Both CSVs must contain {required_cols} columns")

    # Print label categories for debugging
    print(f"Unique labels in {csv1}: {df1['label'].unique()}")
    print(f"Unique labels in {csv2}: {df2['label'].unique()}")

    # Filter by label (0 = Human, 1 = AI) and drop NaN values
    human_ppl_1 = df1[df1["label"] == 0]["perplexity"].dropna()
    ai_ppl_1 = df1[df1["label"] == 1]["perplexity"].dropna()
    human_ppl_2 = df2[df2["label"] == 0]["perplexity"].dropna()
    ai_ppl_2 = df2[df2["label"] == 1]["perplexity"].dropna()

    # Check if any distribution is empty
    for name, dist in zip(["Human Model 1", "AI Model 1", "Human Model 2", "AI Model 2"], 
                          [human_ppl_1, ai_ppl_1, human_ppl_2, ai_ppl_2]):
        print(f"{name} sample size: {len(dist)}")
        if dist.empty:
            raise ValueError(f"{name} distribution is empty!")

    # Run statistical tests
    ks_human_stat, ks_human_p = stats.ks_2samp(human_ppl_1, human_ppl_2)
    ks_ai_stat, ks_ai_p = stats.ks_2samp(ai_ppl_1, ai_ppl_2)

    emd_human = wasserstein_distance(human_ppl_1, human_ppl_2)
    emd_ai = wasserstein_distance(ai_ppl_1, ai_ppl_2)

    spearman_human, _ = stats.spearmanr(human_ppl_1, human_ppl_2)
    spearman_ai, _ = stats.spearmanr(ai_ppl_1, ai_ppl_2)

    # Print results
    print(f"KS Test (Human Texts): D={ks_human_stat:.4f}, p={ks_human_p:.4f}")
    print(f"KS Test (AI-Generated Texts): D={ks_ai_stat:.4f}, p={ks_ai_p:.4f}")
    print(f"Wasserstein Distance (Human): {emd_human:.4f}")
    print(f"Wasserstein Distance (AI): {emd_ai:.4f}")
    print(f"Spearman Correlation (Human): {spearman_human:.4f}")
    print(f"Spearman Correlation (AI): {spearman_ai:.4f}")

# Example usage
compare_perplexity_across_models("perplexity_data/utter-project/EuroLLM-1.7B/perplexity_nl.csv", "perplexity_data/meta-llama/Llama-3.2-3B/perplexity_nl.csv")  # Replace with actual filenames
