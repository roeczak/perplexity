import pandas as pd

# Load CSV
csv_file = "coling_testset.csv"  # Change to your actual file
df = pd.read_csv(csv_file)

# Replace these with your actual column names
language_column = "language"
label_column = "label"

# Ensure the label column is treated as a string or numeric
df[label_column] = df[label_column].astype(str)

# Count occurrences of each label per language
result = df.groupby([language_column, label_column]).size().unstack(fill_value=0)

# Print result
print(result)
