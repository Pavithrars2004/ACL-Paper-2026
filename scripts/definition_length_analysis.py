import pandas as pd

df = pd.read_csv("data/results/gemini_evaluated.csv")

df["def_length"] = df["definition"].apply(
    lambda x: len(str(x).split())
)

df["bucket"] = pd.qcut(df["def_length"], 3)

result = df.groupby("bucket")["consistent"].mean()

print("Consistency by definition length:")
print(result)
