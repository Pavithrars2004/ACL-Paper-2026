import pandas as pd
import sys

# =========================
# 1. Input / Output paths
# =========================
INPUT_FILE = sys.argv[1]
OUTPUT_FILE = INPUT_FILE.replace("_raw.csv", "_evaluated.csv")

df = pd.read_csv(INPUT_FILE)
print(f"✅ Loaded {len(df)} rows from {INPUT_FILE}")

# =========================
# 2. Normalize model answers
# =========================
def normalize_answer(text):
    """
    Normalize model outputs to yes / no / unknown
    """
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

df["normalized_answer"] = df["model_answer"].apply(normalize_answer)

# =========================
# 3. Compute consistency
# =========================
df["consistent"] = df["normalized_answer"] == df["expected_answer"]

# =========================
# 4. Summary statistics
# =========================
overall_consistency = df["consistent"].mean()
unknown_rate = (df["normalized_answer"] == "unknown").mean()

print("\n📊 EVALUATION SUMMARY")
print("--------------------")
print(f"Overall consistency rate: {overall_consistency:.3f}")
print(f"Unknown / unclear answers: {unknown_rate:.3f}")

# =========================
# 5. Per-domain breakdown
# =========================
domain_stats = (
    df.groupby("domain")["consistent"]
    .mean()
    .sort_values(ascending=False)
)

print("\n📊 Consistency by domain")
print(domain_stats)
import pandas as pd
import sys
import re

# =========================
# 1. Input / Output paths
# =========================
INPUT_FILE = sys.argv[1]
OUTPUT_FILE = INPUT_FILE.replace("_raw.csv", "_evaluated.csv")

df = pd.read_csv(INPUT_FILE)
print(f"✅ Loaded {len(df)} rows from {INPUT_FILE}")

# =========================
# 2. Detect model output column
# =========================
# We assume the model output column is the one that is NOT metadata
excluded_cols = {
    "id", "domain", "concept",
    "definition_query", "application_query",
    "expected_answer"
}

candidate_cols = [c for c in df.columns if c not in excluded_cols]

if len(candidate_cols) != 1:
    raise ValueError(
        f"❌ Could not uniquely identify model output column. Found: {candidate_cols}"
    )

MODEL_COL = candidate_cols[0]
print(f"🧠 Detected model output column: '{MODEL_COL}'")

# =========================
# 3. Normalize model answers
# =========================
def normalize_answer(text):
    """
    Normalize model outputs to yes / no / unknown
    Robust to explanations and verbose outputs
    """
    if not isinstance(text, str):
        return "unknown"

    text = text.lower().strip()

    # Extract first standalone yes/no
    match = re.search(r"\b(yes|no)\b", text)
    if match:
        return match.group(1)

    return "unknown"

df["normalized_answer"] = df[MODEL_COL].apply(normalize_answer)

# =========================
# 4. Compute consistency
# =========================
df["consistent"] = df["normalized_answer"] == df["expected_answer"]

# =========================
# 5. Summary statistics
# =========================
overall_consistency = df["consistent"].mean()
unknown_rate = (df["normalized_answer"] == "unknown").mean()

print("\n📊 EVALUATION SUMMARY")
print("--------------------")
print(f"Overall consistency rate: {overall_consistency:.3f}")
print(f"Unknown / unclear answers: {unknown_rate:.3f}")

# =========================
# 6. Per-domain breakdown
# =========================
domain_stats = (
    df.groupby("domain")["consistent"]
    .mean()
    .sort_values(ascending=False)
)

print("\n📊 Consistency by domain")
print(domain_stats)

# =========================
# 7. Save evaluated file
# =========================
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Evaluated results saved to: {OUTPUT_FILE}")

# =========================
# 6. Save evaluated file
# =========================
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Evaluated results saved to: {OUTPUT_FILE}")
