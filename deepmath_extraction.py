from datasets import load_dataset

# Load dataset
dataset = load_dataset("zwhe99/DeepMath-103K", split="train")

# Filter by difficulty
filtered = dataset.filter(lambda x: x["difficulty"] >= 7.5)

print(f"Filtered samples: {len(filtered)}")

# Train-test split (90% train, 10% test)
splits = filtered.train_test_split(
    test_size=0.1,
    seed=42,
    shuffle=True
)

train_dataset = splits["train"]
test_dataset = splits["test"]

print(f"Train: {len(train_dataset)}")
print(f"Test : {len(test_dataset)}")

# Save
train_dataset.to_json("deepmath_train_difficulty_ge_7_5.jsonl")
test_dataset.to_json("deepmath_test_difficulty_ge_7_5.jsonl")