import pandas as pd

df = pd.read_json("test_set_multilingual_with_label.jsonl", lines=True)
df.to_csv("coling_testset.csv", index=False)

print("Conversion complete!")
