import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Load evaluated results
# =========================
df = pd.read_csv("data/results/gemini_evaluated.csv")

# =========================
# 1. Overall consistency
# =========================
overall = df["consistent"].mean()
print(f"\nOverall consistency: {overall:.3f}")

# =========================
# 2. Per-domain consistency
# =========================
domain_scores = (
    df.groupby("domain")["consistent"]
    .mean()
    .sort_values(ascending=False)
)

print("\nConsistency by domain:")
print(domain_scores)

# =========================
# 3. Plot (for paper)
# =========================
plt.figure(figsize=(10, 4))
domain_scores.plot(kind="bar")
plt.ylabel("Consistency Rate")
plt.title("Definition–Application Consistency (Gemini)")
plt.ylim(0, 1)
plt.tight_layout()

plt.savefig("figures/gemini_domain_consistency.png")
print("\nFigure saved to figures/gemini_domain_consistency.png")

plt.show()
