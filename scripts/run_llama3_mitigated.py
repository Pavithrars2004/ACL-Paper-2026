import pandas as pd
from tqdm import tqdm
import subprocess

INPUT_FILE = "data/raw/definition_application_pairs.csv"
OUTPUT_FILE = "data/results/llama3_mitigated_raw.csv"

df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} examples")

def query_llama(prompt):
    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt,
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

rows = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    definition_prompt = row["definition_query"]
    application_query = row["application_query"]

    # Step 1: Get definition
    definition = query_llama(definition_prompt)

    # Step 2: Mitigated application prompt
    mitigated_prompt = f"""
You previously gave the following definition:

"{definition}"

Using ONLY this definition, reason carefully and answer yes or no.

Question: {application_query}
Answer:
"""

    answer = query_llama(mitigated_prompt)

    rows.append({
        "domain": row["domain"],
        "concept": row["concept"],
        "definition": definition,
        "application_query": application_query,
        "model_answer": answer,
        "expected_answer": row["expected_answer"]
    })

pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)
print(f"Saved: {OUTPUT_FILE}")
