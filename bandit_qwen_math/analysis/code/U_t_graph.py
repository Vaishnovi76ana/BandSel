import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

INPUT_FILE = "../src/bandit_history_beta_03_delta_005.json"

BETA = 0.3
DELTA = 0.05

OUTPUT_FILE = (
    f"../outputs/Ut_statistics_beta_{BETA}_delta_{DELTA}.png"
)


# ─────────────────────────────────────────────
# Load JSON
# ─────────────────────────────────────────────

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


# ─────────────────────────────────────────────
# Extract U_t statistics
# ─────────────────────────────────────────────

iterations = []
ut_min = []
ut_median = []
ut_max = []

for entry in data:
    iterations.append(entry["iter"])
    ut_min.append(entry["ut_min"])
    ut_median.append(entry["ut_median"])
    ut_max.append(entry["ut_max"])


# ─────────────────────────────────────────────
# Check data
# ─────────────────────────────────────────────

if not iterations:
    raise ValueError("No data found in JSON file.")

print(f"Found {len(iterations)} iterations.")

print(f"First U_t min:    {ut_min[0]:.6f}")
print(f"First U_t median: {ut_median[0]:.6f}")
print(f"First U_t max:    {ut_max[0]:.6f}")

print(f"Last U_t min:     {ut_min[-1]:.6f}")
print(f"Last U_t median:  {ut_median[-1]:.6f}")
print(f"Last U_t max:     {ut_max[-1]:.6f}")


# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(
    iterations,
    ut_min,
    marker="o",
    markersize=3,
    linewidth=1.5,
    label=r"$U_t$ minimum"
)

ax.plot(
    iterations,
    ut_median,
    marker="o",
    markersize=3,
    linewidth=1.5,
    label=r"$U_t$ median"
)

ax.plot(
    iterations,
    ut_max,
    marker="o",
    markersize=3,
    linewidth=1.5,
    label=r"$U_t$ maximum"
)


# ─────────────────────────────────────────────
# Labels
# ─────────────────────────────────────────────

ax.set_xlabel("Iteration")
ax.set_ylabel(r"Estimated reward $\hat{\rho}_t(a)$")

ax.set_title(
    r"$U_t$ Score Statistics vs Iteration"
    "\n"
    rf"$\beta={BETA}$, $\delta={DELTA}$"
)


# ─────────────────────────────────────────────
# Readable tick intervals
# ─────────────────────────────────────────────

ax.xaxis.set_major_locator(
    MaxNLocator(nbins=10, integer=True)
)

ax.yaxis.set_major_locator(
    MaxNLocator(nbins=8)
)


# ─────────────────────────────────────────────
# Grid / legend
# ─────────────────────────────────────────────

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