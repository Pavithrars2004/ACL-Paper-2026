import pandas as pd

# =========================
# Local copy (DO NOT import evaluate.py)
# =========================
def normalize_answer(text):
    if not isinstance(text, str):
        return "unknown"
    text = text.lower().strip()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    if "yes" in text and "no" not in text:
        return "yes"
    if "no" in text and "yes" not in text:
        return "no"
    return "unknown"


INPUT_FILE = "data/raw/definition_application_pairs.csv"
MODEL_OUTPUT = "data/results/llama3_raw.csv"
OUTPUT_FILE = "data/results/llama3_definition_free.csv"

# Load data
gold = pd.read_csv(INPUT_FILE)
model = pd.read_csv(MODEL_OUTPUT)

# Merge
df = gold.merge(model[["id", "model_answer"]], on="id")

# Remove definitions entirely
df["definition_query"] = ""

# Normalize + evaluate
df["normalized_answer"] = df["model_answer"].apply(normalize_answer)
df["consistent"] = df["normalized_answer"] == df["expected_answer"]

df.to_csv(OUTPUT_FILE, index=False)

print("✅ Definition-free baseline saved to:", OUTPUT_FILE)
print("📊 Consistency:", round(df["consistent"].mean(), 3))
