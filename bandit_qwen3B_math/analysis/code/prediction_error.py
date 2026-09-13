import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

INPUT_FILE = "../src/bandit_history.json"

BETA = 0.1
DELTA = 0.1

OUTPUT_FILE = (
    f"../outputs/Prediction_error_beta_{BETA}_delta_{DELTA}.png"
)


# Load JSON
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract data
iterations = [entry["iter"] for entry in data]
prediction_errors = [entry["prediction_error"] for entry in data]

# Plot
fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(
    iterations,
    prediction_errors,
    marker="o",
    markersize=3,
    linewidth=1.5,
    label="Prediction error"
)

ax.set_xlabel("Iteration")
ax.set_ylabel("Prediction Error")

ax.set_title(
    "Prediction Error vs Iteration\n"
    rf"$\beta={BETA}$, $\delta={DELTA}$"
)

# Readable tick intervals
ax.xaxis.set_major_locator(
    MaxNLocator(nbins=10, integer=True)
)

ax.yaxis.set_major_locator(
    MaxNLocator(nbins=8)
)

ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()


# ─────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Saved plot to: {OUTPUT_FILE}")