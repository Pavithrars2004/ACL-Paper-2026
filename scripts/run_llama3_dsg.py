import pandas as pd
from tqdm import tqdm
import ollama

INPUT_FILE = "data/raw/definition_application_pairs.csv"
OUTPUT_FILE = "data/results/llama3_dsg_raw.csv"

df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} examples")

def query_llama(prompt):
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0}
    )
    return response["message"]["content"].strip()

results = []

for _, row in tqdm(df.iterrows(), total=len(df)):

    # ======================
    # 1. Get definition
    # ======================
    definition_prompt = f"Define the following concept clearly:\n\nConcept: {row['concept']}"
    definition = query_llama(definition_prompt)

    # ======================
    # 2. Extract constraints
    # ======================
    constraint_prompt = f"""
You previously gave the following definition:

{definition}

Extract the key defining constraints as 2–4 short bullet points.
Only include constraints explicitly stated in the definition.
"""
    constraints = query_llama(constraint_prompt)

    # ======================
    # 3. Application query
    # ======================
    application_prompt = f"""
Definition constraints:
{constraints}

Using ONLY the constraints above, answer the following question.
If the constraints are insufficient, answer "unknown".

Question: {row['application_query']}
Answer with yes / no / unknown.
"""
    answer = query_llama(application_prompt)

    results.append({
        "id": row["id"],
        "domain": row["domain"],
        "concept": row["concept"],
        "definition": definition,
        "constraints": constraints,
        "model_answer": answer,
        "expected_answer": row["expected_answer"],
    })

out_df = pd.DataFrame(results)
out_df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Saved DSG results to {OUTPUT_FILE}")
