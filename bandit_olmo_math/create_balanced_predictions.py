#!/usr/bin/env python
"""Create a balanced dataset of 250 correct + 250 incorrect predictions for EOS predictor training."""
import json, random

PREDICTIONS_PATH = "/home/sourangshu/vaishnovi/tokenskip/TokenSkip/outputs/Olmo-3-7B-Instruct/math/7b/Original/train/samples/predictions.jsonl"
OUTPUT_PATH = "dataset/balanced_predictions.jsonl"
NUM_PER_CLASS = 250

def main():
    correct, incorrect = [], []
    with open(PREDICTIONS_PATH, 'r') as f:
        for line in f:
            d = json.loads(line.strip())
            if d.get('accuracy'):
                correct.append(d)
            else:
                incorrect.append(d)

    print(f"Total correct: {len(correct)}, incorrect: {len(incorrect)}")

    random.seed(42)
    random.shuffle(correct)
    random.shuffle(incorrect)

    selected = correct[:NUM_PER_CLASS] + incorrect[:NUM_PER_CLASS]
    random.shuffle(selected)

    with open(OUTPUT_PATH, 'w') as f:
        for d in selected:
            f.write(json.dumps(d) + '\n')

    print(f"Saved {len(selected)} samples ({NUM_PER_CLASS} correct + {NUM_PER_CLASS} incorrect) to {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
