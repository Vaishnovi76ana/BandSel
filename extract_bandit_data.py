import json

model_name = "Qwen2.5-7B-Instruct"

files = [
    (0.5, f"outputs/{model_name}/tokenskip/math/TokenSkip/train/0.5/samples/predictions.jsonl"),
    (0.7, f"outputs/{model_name}/tokenskip/math/TokenSkip/train/0.7/samples/predictions.jsonl"),
    (0.9, f"outputs/{model_name}/tokenskip/math/TokenSkip/train/0.9/samples/predictions.jsonl"),
    (1.0, f"outputs/{model_name}/tokenskip/math/Original/train/samples/predictions.jsonl"),
]



# Read all files
datasets = []
for cr, filename in files:
    with open(filename, "r", encoding="utf-8") as f:
        datasets.append((cr, [json.loads(line) for line in f]))

# Ensure all files have the same number of samples
n = len(datasets[0][1])
assert all(len(ds) == n for _, ds in datasets), "Files have different lengths."

output = []

for i in range(n):
    for cr, ds in datasets:
        sample = ds[i]
        if sample.get("accuracy", False):
            sample = sample.copy()
            sample["optimal_compression_ratio"] = cr
            output.append(sample)
            break

# Write output as JSONL
with open(f"outputs/{model_name}/bandit/math/bandit_data.jsonl", "w", encoding="utf-8") as f:
    for sample in output:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"Saved {len(output)} samples.")