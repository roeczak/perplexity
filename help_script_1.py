import pandas as pd

# Load CSV
df = pd.read_csv("coling_testset.csv")

# Replace values in the 'language' column
df['language'] = df['language'].replace("russian", "Russian")

# Save the updated file
df.to_csv("data.csv", index=False)

print("Update complete!")
