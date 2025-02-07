import pandas as pd
from scipy.stats import spearmanr

# Function to load perplexity values from CSV files
def load_perplexity_values(csv_file):
    # Assuming that the column containing perplexity is named 'perplexity'
    df = pd.read_csv(csv_file)
    return df['perplexity']  # Modify column name if necessary

# File paths for the CSVs
csv_file_1 = 'perplexity_data_coling/facebook/opt-125m/perplexity_id.csv'
csv_file_2 = 'perplexity_data_coling/gpt2/perplexity_id.csv'

# Load perplexity values from the two CSVs
perplexity_1 = load_perplexity_values(csv_file_1)
perplexity_2 = load_perplexity_values(csv_file_2)

# Ensure both lists have the same length
if len(perplexity_1) != len(perplexity_2):
    raise ValueError("The CSV files must have the same number of rows to compute correlation.")

# Compute the Spearman rank correlation
correlation, p_value = spearmanr(perplexity_1, perplexity_2)

# Output the results
print(f"Spearman Correlation: {correlation}")
print(f"P-value: {p_value}")
