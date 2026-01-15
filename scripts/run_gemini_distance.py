import os
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai

# =========================
# Load API key
# =========================
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "gemini-2.5-flash"

# =========================
# Load prompts
# =========================
with open("prompts/definition_prompt.txt") as f:
    DEF_PROMPT = f.read()

with open("prompts/application_prompt.txt") as f:
    APP_PROMPT = f.read()

# =========================
# Load dataset (sample 100)
# =========================
df = pd.read_csv("data/raw/definition_application_pairs.csv")
df = df.sample(100, random_state=42)

results = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    try:
        # Definition
        def_prompt = DEF_PROMPT.format(concept=row["concept"])
        definition = client.models.generate_content(
            model=MODEL_NAME,
            contents=def_prompt
        ).text.strip()

        time.sleep(0.25)

        # Insert neutral turn (DISTANCE)
        app_prompt = (
            f"{definition}\n\n"
            "Thanks for the explanation.\n\n"
            + APP_PROMPT.format(application_query=row["application_query"])
        )

        answer = client.models.generate_content(
            model=MODEL_NAME,
            contents=app_prompt
        ).text.strip().lower()

        time.sleep(0.25)

    except Exception:
        answer = "error"

    results.append({
        "domain": row["domain"],
        "model_answer": answer,
        "expected_answer": row["expected_answer"]
    })

os.makedirs("data/results", exist_ok=True)
pd.DataFrame(results).to_csv(
    "data/results/gemini_distance_raw.csv", index=False
)

print("✅ Distance sensitivity run completed")
