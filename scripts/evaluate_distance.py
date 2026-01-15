import pandas as pd

df = pd.read_csv("data/results/gemini_distance_raw.csv")

def normalize(x):
    x = str(x).lower()
    if x.startswith("yes"):
        return "yes"
    if x.startswith("no"):
        return "no"
    return "unknown"

df["norm"] = df["model_answer"].apply(normalize)
df["consistent"] = df["norm"] == df["expected_answer"]

print("Distance consistency:", df["consistent"].mean())
