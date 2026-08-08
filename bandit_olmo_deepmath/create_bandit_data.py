"""
Create bandit_data.jsonl from optimal_traces_correct.jsonl for OLMo MATH-500.

This mirrors the structure used by the bandit pipeline:
  - id, prompt, answer, original_cot_length, target_compression_ratio, accuracy, compressed_trace

Usage:
    python create_bandit_data.py
"""
import json
import os

INPUT = "/home/sourangshu/vaishnovi/tokenskip/TokenSkip/outputs/Olmo-3-7B-Instruct/math/7b/TokenSkip/train/optimal_traces_correct.jsonl"
OUTPUT = "dataset/bandit_data.jsonl"


def create_bandit_data():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    count = 0
    with open(INPUT, 'r', encoding='utf-8') as infile, \
         open(OUTPUT, 'w', encoding='utf-8') as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            merged_record = {
                'id': data['id'],
                'prompt': data['prompt'],
                'answer': data['answer'],
                'original_cot_length': data['cot_length'],
                'target_compression_ratio': data['optimal_compression_ratio'],
                'accuracy': True,
                'compressed_trace': data['optimal_trace'],
            }
            outfile.write(json.dumps(merged_record) + '\n')
            count += 1

    print(f"Successfully created {OUTPUT} with {count} entries (all accuracy=true)")


if __name__ == '__main__':
    create_bandit_data()
