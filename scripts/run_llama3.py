import pandas as pd
import requests
import time
from tqdm import tqdm

INPUT_FILE = "data/raw/definition_application_pairs.csv"
OUTPUT_FILE = "data/results/llama3_raw.csv"
MODEL = "llama3:8b"

def query(prompt):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }
    )
    return r.json()["response"].strip().lower()

df = pd.read_csv(INPUT_FILE)
rows = []

print(f"Loaded {len(df)} examples")

for _, row in tqdm(df.iterrows(), total=len(df)):
    # Step 1: Definition (use dataset prompt)
    definition = query(row["definition_query"])
    time.sleep(0.5)

    # Step 2: Application (use dataset prompt)
    application_prompt = (
        f"Definition:\n{definition}\n\n"
        f"Question: {row['application_query']}\n"
        f"Answer only yes or no."
    )

    answer = query(application_prompt)
    time.sleep(0.5)

    rows.append({
        "id": row["id"],
        "domain": row["domain"],
        "concept": row["concept"],
        "definition": definition,
        "model_answer": answer,
        "expected_answer": row["expected_answer"]
    })

pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)
print("Saved:", OUTPUT_FILE)
