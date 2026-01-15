import pandas as pd

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
OUTPUT_FILE = "data/results/llama3_shuffled_definition.csv"

gold = pd.read_csv(INPUT_FILE)
model = pd.read_csv(MODEL_OUTPUT)

df = gold.merge(model[["id", "model_answer"]], on="id")

# Shuffle definitions
df["definition_query"] = df["definition_query"].sample(
    frac=1, random_state=42
).values

df["normalized_answer"] = df["model_answer"].apply(normalize_answer)
df["consistent"] = df["normalized_answer"] == df["expected_answer"]

df.to_csv(OUTPUT_FILE, index=False)

print("✅ Shuffled-definition baseline saved to:", OUTPUT_FILE)
print("📊 Consistency:", round(df["consistent"].mean(), 3))
