import json

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------- Inputs ----------
all_questions_path = "outputs/Qwen2.5-7B-Instruct/bandit/math/bandit_data.jsonl"
selected_questions_path = "outputs/Qwen2.5-7B-Instruct/bandit/math/final_training_data.jsonl"

# ---------- Load ----------
all_questions = load_jsonl(all_questions_path)
selected_questions = load_jsonl(selected_questions_path)

# ---------- Find unselected ----------
selected_ids = {str(sample["id"]) for sample in selected_questions}

unselected_questions = [
    sample
    for sample in all_questions
    if str(sample["id"]) not in selected_ids
]

# ---------- Save ----------
save_jsonl(unselected_questions, "unselected_questions.jsonl")

print(f"Total questions      : {len(all_questions)}")
print(f"Selected questions   : {len(selected_questions)}")
print(f"Unselected questions : {len(unselected_questions)}")