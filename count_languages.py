import pandas as pd
from collections import Counter

# File path
csv_file = "coling_testset.csv"

# Load CSV
df = pd.read_csv(csv_file)

# Replace 'language' with the actual column name in your CSV
language_column = "language"  # Change this to match your column name

# Count occurrences of each language
if language_column in df.columns:
    language_counts = Counter(df[language_column].dropna())  # Remove NaN values
    for lang, count in language_counts.items():
        print(f"{lang}: {count}")
else:
    print(f"Column '{language_column}' not found in the CSV.")
