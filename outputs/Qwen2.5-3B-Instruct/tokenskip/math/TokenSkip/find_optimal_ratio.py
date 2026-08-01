import json
import os

base_path = r"e:\MTP\tokenskip\TokenSkip\outputs\Qwen2.5-3B-Instruct\math\3b\TokenSkip\math-500-test"

# Define paths
files = {
    0.5: os.path.join(base_path, "0.5", "samples", "max_token_1024", "predictions.jsonl"),
    0.7: os.path.join(base_path, "0.7", "samples", "max_token_1024", "predictions.jsonl"),
    0.9: os.path.join(base_path, "0.9", "samples", "max_token_1024", "predictions.jsonl"),
    1.0: os.path.join(base_path, "1.0", "samples", "max_token_1024", "predictions.jsonl")
}

data = {} # ratio -> {id -> record}

# Load all data
for ratio, path in files.items():
    if not os.path.exists(path):
        print(f"Warning: File not found for ratio {ratio}: {path}")
        # Try fallback for 1.0 if strictly 1000 is not found?
        # User said "1.0/samples", I saw predictions_1000.jsonl there.
        data[ratio] = {}
        continue
    
    print(f"Loading {path}...")
    records = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
                if 'id' in rec:
                    records[rec['id']] = rec
            except json.JSONDecodeError:
                pass
    data[ratio] = records

# Collect all IDs
all_ids = set()
for r in data:
    all_ids.update(data[r].keys())

print(f"Total unique IDs found: {len(all_ids)}")

# Sort IDs for deterministic output
# Attempt to sort by numeric components if possible, else string sort
try:
    sorted_ids = sorted(list(all_ids), key=lambda x: [int(p) if p.isdigit() else p for p in x.replace('-', ' ').split()])
except:
    sorted_ids = sorted(list(all_ids))

results = []
stats = {0.5: 0, 0.7: 0, 0.9: 0, 1.0: 0}
incorrect_all_count = 0

for qid in sorted_ids:
    # Stats: Check if any is correct
    any_correct = False
    for r in [0.5, 0.7, 0.9, 1.0]:
        if qid in data[r] and data[r][qid].get('accuracy', False):
            any_correct = True
            break
    if not any_correct:
        incorrect_all_count += 1
        
    selected_ratio = 1.0
    selected_rec = None
    
    # Check 0.5
    if qid in data[0.5] and data[0.5][qid].get('accuracy', False):
        selected_ratio = 0.5
        selected_rec = data[0.5][qid]
    # Check 0.7
    elif qid in data[0.7] and data[0.7][qid].get('accuracy', False):
        selected_ratio = 0.7
        selected_rec = data[0.7][qid]
    # Check 0.9
    elif qid in data[0.9] and data[0.9][qid].get('accuracy', False):
        selected_ratio = 0.9
        selected_rec = data[0.9][qid]
    # Fallback to 1.0
    else:
        selected_ratio = 1.0
        if qid in data[1.0]:
            selected_rec = data[1.0][qid]
        else:
            # Fallback if 1.0 is missing this ID
            if qid in data[0.9]: selected_rec = data[0.9][qid]
            elif qid in data[0.7]: selected_rec = data[0.7][qid]
            elif qid in data[0.5]: selected_rec = data[0.5][qid]
    
    if selected_rec:
        out_rec = selected_rec.copy()
        out_rec['optimal_compression_ratio'] = selected_ratio
        results.append(out_rec)
        
        # Update statistics
        if any_correct:
            if selected_ratio in stats:
                stats[selected_ratio] += 1
            else:
                stats[selected_ratio] = 1

output_path = os.path.join(base_path, "optimal_predictions_math_500_test.jsonl")
print(f"Writing {len(results)} records to {output_path}...")
with open(output_path, 'w', encoding='utf-8') as f:
    for rec in results:
        f.write(json.dumps(rec) + "\n")

print("Done.")

print("\nPossible Optimal Ratios Statistics:")
for ratio in sorted(stats.keys()):
    print(f"Ratio {ratio}: {stats[ratio]}")

import statistics

print(f"\nQuestions with no correct predictions in any ratio: {incorrect_all_count}")

# Calculate mean and median for correct assignments
correct_ratios = []
for ratio, count in stats.items():
    correct_ratios.extend([ratio] * count)

if correct_ratios:
    mean_ratio = statistics.mean(correct_ratios)
    median_ratio = statistics.median(correct_ratios)
    print(f"\nMean Optimal Ratio: {mean_ratio:.4f}")
    print(f"Median Optimal Ratio: {median_ratio}")
else:
    print("\nNo correct predictions found to calculate statistics.")
