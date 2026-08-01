import json
import os
import numpy as np

def load_jsonl(path):
    data = []
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return data
    else:
        print(f"File not found: {path} (cwd: {os.getcwd()})")
        return []

# File paths
compression_ratio = 1.0
file_path_tokenskip = f"math-500-test/{compression_ratio}/samples/max_token_1024/predictions.jsonl"
file_path_original = "predictions.jsonl"

# Load data
data_tokenskip = load_jsonl(file_path_tokenskip)
data_original = load_jsonl(file_path_original)


# Collect items in ratio range [0.9, 1.1] with accuracy
items_in_range_09_11 = []

for d1, d2 in zip(data_tokenskip, data_original):
    if d1.get('accuracy') and d2.get('accuracy'):
        d1_len = d1['cot_length']
        d2_len = d2['cot_length']
        
        ratio = d1_len / d2_len
        if 0.9 <= ratio <= 1.1:
            items_in_range_09_11.append((d1_len, d1, d2))

if items_in_range_09_11:
    cot_lengths = [x[0] for x in items_in_range_09_11]
    count = len(cot_lengths)
    avg_cot = np.mean(cot_lengths)
    std_cot = np.std(cot_lengths)
    min_cot = np.min(cot_lengths)
    max_cot = np.max(cot_lengths)

    print(f"Stats for cot_ratio in [0.9, 1.1]:")
    print(f"Count: {count}")
    print(f"Average: {avg_cot}")
    print(f"Std Dev: {std_cot}")
    print(f"Min: {min_cot}")
    print(f"Max: {max_cot}")

    # Sort by cot_length (d1_len) and take 5 smallest
    items_in_range_09_11.sort(key=lambda x: x[0])
    smallest_5 = items_in_range_09_11[:5]

    print("\n--- 5 Smallest COT Lengths (TokenSkip) in Range [0.9, 1.1] ---")
    for i, (length, d1, d2) in enumerate(smallest_5, 1):
        print(f"\nItem {i}:")
        print(f"Tokenskip COT Length: {length}")
        print(f"Optimal COT Length: {d2['cot_length']}")
        print(f"Ratio: {length / d2['cot_length']:.4f}")
        
        # d1 is tokenskip, d2 is optimal (based on variable names in previous context, though file says data_original which might be optimal depending on loading order. 
        # Checking loading: data_tokenskip from .../samples/max_token_1024/..., data_original from predictions.jsonl
        # The prompt says "both tokenskip and optimal". Assuming data_tokenskip is tokenskip and data_original is the baseline/optimal here.
        
        # Question is usually in 'question' or 'messages' depending on format. 
        # Looking at previous context JSON structure: 'question' key exists.
        
        if 'question' in d1:
            print(f"Question: {d1['question']}")
        elif 'messages' in d1:
             print(f"Question (from messages): {d1['messages'][0]['content']}")
        
        print(f"TokenSkip Answer: {d1.get('model_output', 'N/A')}")
        print(f"Optimal Answer: {d2.get('model_output', 'N/A')}")
        print("-" * 20)

else:
    print("No samples found in range [0.9, 1.1]")
