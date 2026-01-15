import pandas as pd

df = pd.read_csv("data/results/gemini_evaluated.csv")

CONFIDENT_MARKERS = [
    "definitely", "clearly", "always", "certainly",
    "must", "is", "are"
]

def is_confident(text):
    text = str(text).lower()
    return any(m in text for m in CONFIDENT_MARKERS)

df["confident"] = df["model_answer"].apply(is_confident)

confident_wrong = df[(~df["consistent"]) & (df["confident"])]

print("Confident but wrong rate:",
      len(confident_wrong) / len(df))
