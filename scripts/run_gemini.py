import os
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai

# =========================
# 1. Load API key
# =========================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=API_KEY)

# ✅ USE A MODEL THAT EXISTS IN YOUR PROJECT
MODEL_NAME = "gemini-2.5-flash"

# =========================
# 2. Load prompts
# =========================
with open("prompts/definition_prompt.txt") as f:
    DEF_PROMPT = f.read()

with open("prompts/application_prompt.txt") as f:
    APP_PROMPT = f.read()

# =========================
# 3. Load dataset
# =========================
df = pd.read_csv("data/raw/definition_application_pairs.csv")
print(f"✅ Loaded dataset with {len(df)} items")

# =========================
# 4. Run Gemini
# =========================
results = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    try:
        # ---- Definition ----
        def_prompt = DEF_PROMPT.format(concept=row["concept"])
        def_resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=def_prompt,
        )
        definition = def_resp.text.strip()

        time.sleep(0.25)

        # ---- Application ----
        app_prompt = (
            f"{definition}\n\n"
            + APP_PROMPT.format(application_query=row["application_query"])
        )
        app_resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=app_prompt,
        )
        answer = app_resp.text.strip().lower()

        time.sleep(0.25)

    except Exception as e:
        print(f"⚠️ Error at id={row['id']}: {e}")
        definition = "ERROR"
        answer = "ERROR"

    results.append({
        "id": row["id"],
        "domain": row["domain"],
        "concept": row["concept"],
        "definition": definition,
        "model_answer": answer,
        "expected_answer": row["expected_answer"],
    })

# =========================
# 5. Save results
# =========================
os.makedirs("data/results", exist_ok=True)
out_path = "data/results/gemini_raw.csv"
pd.DataFrame(results).to_csv(out_path, index=False)

print("✅ Gemini run completed successfully")
print(f"📁 Results saved to {out_path}")
