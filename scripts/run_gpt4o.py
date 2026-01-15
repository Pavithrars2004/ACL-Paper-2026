import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import os

# ===== CONFIG =====
MODEL_NAME = "gpt-4o"
INPUT_FILE = "data/raw/definition_application_pairs.csv"

OUTPUT_FILE = "data/results/gpt4o_raw.csv"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def query_gpt(prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip().lower()

# ===== LOAD DATA =====
df = pd.read_csv(INPUT_FILE)
rows = []

print(f"Loaded {len(df)} examples")

for _, row in tqdm(df.iterrows(), total=len(df)):
    definition_prompt = f"What is {row['concept']}?"
    definition = query_gpt(definition_prompt)

    application_prompt = (
        f"Definition:\n{definition}\n\n"
        f"Question: {row['application_question']}\n"
        f"Answer yes or no."
    )

    answer = query_gpt(application_prompt)

    rows.append({
        "id": row["id"],
        "domain": row["domain"],
        "concept": row["concept"],
        "definition": definition,
        "model_answer": answer,
        "expected_answer": row["expected_answer"]
    })

pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)
print("GPT-4o run completed.")
