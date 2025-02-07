import pandas as pd

# Load CSV file
csv_file = "coling_testset.csv"
df = pd.read_csv(csv_file)

# Print column names
print("Columns in CSV:", df.columns.tolist())
