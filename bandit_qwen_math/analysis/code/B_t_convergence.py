import re
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

LOG_FILE = "../src/case_output_beta_03_delta_005_20260831_091202.txt"
OUTPUT_FILE = "../outputs/B_convergence_beta_03_delta_005.png"

BETA = 0.3
DELTA = 0.05
EPSILON = 0.05


# ─────────────────────────────────────────────
# Extract B_t from log
# ─────────────────────────────────────────────

iterations = []
B_values = []

current_iteration = None

with open(LOG_FILE, "r", encoding="utf-8") as f:
    for line in f:

        # Example:
        # # ITERATION 1/100
        match_iter = re.search(r"# ITERATION\s+(\d+)", line)

        if match_iter:
            current_iteration = int(match_iter.group(1))
            continue

        # Example:
        # B_t(s,b): 4.012562
        match_B = re.search(
            r"B_\w*\(s,b\):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            line
        )

        if match_B and current_iteration is not None:
            B = float(match_B.group(1))

            iterations.append(current_iteration)
            B_values.append(B)


# ─────────────────────────────────────────────
# Check extracted data
# ─────────────────────────────────────────────

print(f"Found {len(B_values)} B values.")

if B_values:
    print(f"First B: {B_values[0]:.6f}")
    print(f"Last B:  {B_values[-1]:.6f}")


# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(
    iterations,
    B_values,
    marker="o",
    markersize=3,
    linewidth=1.5,
    label=r"$B_t$"
)


# ─────────────────────────────────────────────
# Epsilon stopping line
# ─────────────────────────────────────────────

ax.axhline(
    y=EPSILON,
    linestyle="--",
    linewidth=1.5,
    label=rf"$\epsilon={EPSILON}$"
)


# ─────────────────────────────────────────────
# Axis labels
# ─────────────────────────────────────────────

ax.set_xlabel("Iteration")
ax.set_ylabel(r"$B_t$")

ax.set_title(
    "CASE Convergence: $B_t$ vs Iteration\n"
    rf"$\beta={BETA}$, $\delta={DELTA}$, $\epsilon={EPSILON}$"
)


# ─────────────────────────────────────────────
# Automatically choose readable tick intervals
# ─────────────────────────────────────────────

# X-axis: approximately 10 readable intervals
ax.xaxis.set_major_locator(
    MaxNLocator(nbins=10, integer=True)
)

# Y-axis: approximately 8 readable intervals
ax.yaxis.set_major_locator(
    MaxNLocator(nbins=8)
)


# ─────────────────────────────────────────────
# Grid / legend
# ─────────────────────────────────────────────

ax.grid(
    True,
    alpha=0.3
)

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