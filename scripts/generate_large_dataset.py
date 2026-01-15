import random
import pandas as pd

# =========================
# 1. Config
# =========================
OUTPUT_PATH = "data/raw/large_auto_pairs.csv"
NUM_SAMPLES = 3000   # you can change to 2000 or 5000

DOMAINS = {
    "Common Nouns": [
        ("bird", "an animal with feathers and wings"),
        ("vehicle", "a machine used for transportation"),
        ("fruit", "the edible reproductive part of a plant"),
    ],
    "Procedural Concepts": [
        ("algorithm", "a finite sequence of well-defined steps"),
        ("protocol", "a set of rules governing communication"),
    ],
    "Scientific Concepts": [
        ("acid", "a substance that donates protons"),
        ("mammal", "a warm-blooded vertebrate with hair"),
    ],
}

YES_TEMPLATES = [
    "Is a {y} a {x}?",
    "Does a {y} qualify as a {x}?",
]

NO_TEMPLATES = [
    "Is a {y} a {x}?",
    "Does a {y} qualify as a {x}?",
]

POSITIVE_EXAMPLES = {
    "bird": ["sparrow", "eagle"],
    "vehicle": ["car", "bus"],
    "fruit": ["apple", "banana"],
    "algorithm": ["sorting algorithm", "search procedure"],
    "protocol": ["HTTP", "TCP"],
    "acid": ["hydrochloric acid", "sulfuric acid"],
    "mammal": ["dog", "whale"],
}

NEGATIVE_EXAMPLES = {
    "bird": ["cat", "lizard"],
    "vehicle": ["tree", "house"],
    "fruit": ["rock", "chair"],
    "algorithm": ["infinite loop", "random guessing"],
    "protocol": ["weather", "emotion"],
    "acid": ["salt", "water"],
    "mammal": ["snake", "frog"],
}

# =========================
# 2. Generate Dataset
# =========================
rows = []
uid = 0

for _ in range(NUM_SAMPLES):
    domain = random.choice(list(DOMAINS.keys()))
    concept, definition = random.choice(DOMAINS[domain])

    if random.random() < 0.5:
        y = random.choice(POSITIVE_EXAMPLES[concept])
        expected = "yes"
    else:
        y = random.choice(NEGATIVE_EXAMPLES[concept])
        expected = "no"

    question = random.choice(YES_TEMPLATES).format(x=concept, y=y)

    rows.append({
        "id": uid,
        "domain": domain,
        "concept": concept,
        "definition_query": f"Define a {concept}.",
        "application_query": question,
        "expected_answer": expected
    })
    uid += 1

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Generated {len(df)} large-scale examples at {OUTPUT_PATH}")
