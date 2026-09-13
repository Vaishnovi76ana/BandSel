import re
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

LOG_FILE = "../src/case_output_20260907_094136.txt"
OUTPUT_FILE = "../outputs/reward_tracker_beta_01.png"


BETA = 0.1
ACCURACY_BASE = 0.626


# ─────────────────────────────────────────────
# Extract reward information
# ─────────────────────────────────────────────

iterations = []
accuracy_magnitudes = []
compression_magnitudes = []
total_reward_magnitudes = []

current_iteration = None
current_accuracy_part = None
current_compression_part = None

with open(LOG_FILE, "r", encoding="utf-8") as f:

    for line in f:

        # Iteration
        match_iter = re.search(
            r"# ITERATION\s+(\d+)",
            line
        )

        if match_iter:
            current_iteration = int(match_iter.group(1))
            continue

        # Accuracy Part
        match_acc = re.search(
            r"Accuracy\s+Part\s*:\s*"
            r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            line
        )

        if match_acc:
            current_accuracy_part = float(match_acc.group(1))
            continue

        # Avg CR Part
        match_cr = re.search(
            r"Avg\s+CR\s+Part\s*:\s*"
            r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            line
        )

        if match_cr:
            current_compression_part = float(match_cr.group(1))
            continue

        # Total reward
        match_reward = re.search(
            r"Reward\s+calculated\s*:\s*"
            r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            line
        )

        if (
            match_reward
            and current_iteration is not None
            and current_accuracy_part is not None
            and current_compression_part is not None
        ):
            total_reward = float(match_reward.group(1))

            iterations.append(current_iteration)

            accuracy_magnitudes.append(
                abs(current_accuracy_part)
            )

            compression_magnitudes.append(
                abs(current_compression_part)
            )

            total_reward_magnitudes.append(
                abs(total_reward)
            )

            # Reset for next iteration
            current_accuracy_part = None
            current_compression_part = None


# ─────────────────────────────────────────────
# Plot magnitude
# ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(
    iterations,
    accuracy_magnitudes,
    marker="o",
    markersize=3,
    linewidth=1.5,
    label=r"$|R_{\mathrm{accuracy}}|$"
)

ax.plot(
    iterations,
    compression_magnitudes,
    marker="o",
    markersize=3,
    linewidth=1.5,
    label=r"$|R_{\mathrm{compression}}|$"
)


ax.set_xlabel("Iteration")
ax.set_ylabel("Reward Magnitude")

ax.set_title("Magnitude of Reward Components vs Iteration")

ax.xaxis.set_major_locator(
    MaxNLocator(nbins=10, integer=True)
)

ax.yaxis.set_major_locator(
    MaxNLocator(nbins=8)
)

ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()