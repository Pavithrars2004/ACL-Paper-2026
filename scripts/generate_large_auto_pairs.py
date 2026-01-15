import pandas as pd
import random

# =========================
# Config
# =========================
OUTPUT_FILE = "data/raw/large_auto_pairs_10000.csv"
TARGET_SIZE = 10000

domains = [
    "Common Nouns",
    "Scientific Concepts",
    "Procedural Concepts",
    "Social & Ethical Concepts",
    "Abstract Properties",
    "Mathematics & Logic"
]

concept_templates = {
    "Common Nouns": [
        "chair", "vehicle", "tool", "animal", "fruit", "device"
    ],
    "Scientific Concepts": [
        "enzyme", "planet", "force", "cell", "acid", "algorithm"
    ],
    "Procedural Concepts": [
        "algorithm", "workflow", "protocol", "procedure"
    ],
    "Social & Ethical Concepts": [
        "fairness", "justice", "honesty", "responsibility"
    ],
    "Abstract Properties": [
        "symmetry", "continuity", "stability", "efficiency"
    ],
    "Mathematics & Logic": [
        "prime number", "function", "set", "relation"
    ]
}

rows = []
idx = 0

for _ in range(TARGET_SIZE):
    domain = random.choice(domains)
    concept = random.choice(concept_templates[domain])

    definition_query = f"Define the concept of {concept}."
    application_query = f"Based on this definition, does the following satisfy it?"

    rows.append({
        "id": idx,
        "domain": domain,
        "concept": concept,
        "definition_query": definition_query,
        "application_query": application_query,
        "expected_answer": random.choice(["yes", "no"])  # auto/noisy by design
    })
    idx += 1

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved {len(df)} rows to {OUTPUT_FILE}")
