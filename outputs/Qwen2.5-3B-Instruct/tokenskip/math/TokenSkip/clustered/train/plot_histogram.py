
import matplotlib.pyplot as plt
import os
import json

# Data based on previous analysis
# Ratio 0.5: 1429
# Ratio 0.7: 342
# Ratio 0.9: 87
# Ratio 1.0: 67

ratios = [0.5, 0.7, 0.9, 1.0]
counts = [1429, 342, 87, 67]

output_dir = r"e:\MTP\tokenskip\TokenSkip\outputs\Qwen2.5-3B-Instruct\math\3b\TokenSkip\clustered"
output_file = os.path.join(output_dir, "optimal_ratio_histogram.png")

plt.figure(figsize=(8, 6))
bars = plt.bar([str(r) for r in ratios], counts, color='skyblue', edgecolor='black')

plt.xlabel('Optimal Compression Ratio')
plt.ylabel('Frequency (Number of Questions)')
plt.title('Distribution of Optimal Compression Ratios')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add text labels on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height}',
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig(output_file)
print(f"Histogram saved to: {output_file}")
