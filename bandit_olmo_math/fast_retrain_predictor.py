#!/usr/bin/env python
"""
Fast predictor retraining for OLMo-3-7B-Instruct.
Loads base + LoRA, extracts hidden-state features, trains MLP, saves checkpoint.
Adapted from sagnibha/fast_retrain_predictor.py for OLMo architecture.
"""
import os, sys, json, gc, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

FIXED_PARAMS = {
    "hidden_dim": 512, "dropout": 0.1, "lr": 1e-4, "epochs": 10,
    "threshold_high": 0.864, "threshold_low": 0.203, "threshold_fallback": 0.304,
}
MODEL_NAME = "allenai/Olmo-3-7B-Instruct"
TRAIN_DATA = "dataset/balanced_predictions.jsonl"
OUTPUT_PATH = "predictor_best_optimized.pth"
MAX_LENGTH = 2048
STEP_SIZE = 16


class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.1, mean=None, std=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        if mean is not None:
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)

    def forward(self, x):
        if hasattr(self, "mean"):
            x = torch.clamp(x, -50, 50)
            x = (x - self.mean) / torch.clamp(self.std, min=1e-3)
            x = torch.clamp(x, -10, 10)
        return self.net(x).squeeze(-1)


def format_olmo_prompt(question):
    """Format a math question in OLMo chat template."""
    return (
        f"<|user|>\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n"
        f"{question}\n<|assistant|>\n"
    )


def extract_features(input_ids, model, eos_token_id, step_size, prompt_len=0):
    """Extract hidden-state + logit features at every step_size tokens."""
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    logits = outputs.logits[0]
    hidden_states = outputs.hidden_states[-1][0]
    T = input_ids.shape[1]
    indices = torch.arange(prompt_len + step_size, T, step_size, device=input_ids.device)
    if len(indices) == 0:
        return torch.empty(0, hidden_states.shape[1] + 5, device=input_ids.device)
    logits_sliced = torch.clamp(logits[indices - 1], -50, 50)
    h_sliced = torch.clamp(hidden_states[indices - 1], -50, 50)
    probs = F.softmax(logits_sliced, dim=-1)
    probs_safe = torch.clamp(probs, min=1e-6, max=1.0)
    if isinstance(eos_token_id, list):
        eos_prob = probs_safe[:, eos_token_id].sum(dim=-1)
    else:
        eos_prob = probs_safe[:, eos_token_id]
    max_prob, _ = probs_safe.max(dim=-1)
    entropy = -(probs_safe * torch.log(probs_safe)).sum(dim=-1)
    prev_indices = torch.clamp(indices - 1, min=0)
    prev_logits = torch.clamp(logits[prev_indices - 1], -50, 50)
    prev_probs = F.softmax(prev_logits, dim=-1)
    prev_probs = torch.clamp(prev_probs, min=1e-6, max=1.0)
    prev_entropy = -(prev_probs * torch.log(prev_probs)).sum(dim=-1)
    if isinstance(eos_token_id, list):
        prev_eos = prev_probs[:, eos_token_id].sum(dim=-1)
    else:
        prev_eos = prev_probs[:, eos_token_id]
    features = torch.cat([
        h_sliced,
        eos_prob.unsqueeze(1), entropy.unsqueeze(1), max_prob.unsqueeze(1),
        (entropy - prev_entropy).unsqueeze(1), (eos_prob - prev_eos).unsqueeze(1)
    ], dim=1)
    return features


def build_dataset(data, model, tokenizer, device):
    """Build feature dataset from predictions for training the MLP."""
    all_X, all_y = [], []
    eos_token_id = tokenizer.eos_token_id
    print(f"Extracting features from {len(data)} samples...")
    for sample in tqdm(data):
        # Parse prompt: extract user question from ChatML format if needed
        raw_prompt = sample.get("prompt", "")
        prefix = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
        if raw_prompt.startswith(prefix):
            user_content = raw_prompt[len(prefix):]
        else:
            user_content = raw_prompt.split("Please reason step by step, and put your final answer within \\boxed{}.\n")[-1]
        user_content = user_content.split("<|im_end|>")[0]
        user_content = user_content.split("<|eot_id|>")[0]
        user_content = user_content.strip()

        prompt = format_olmo_prompt(user_content)
        trace = sample.get("model_output", "")
        if not trace:
            continue
        label = int(sample.get("accuracy", 0))

        prompt_tokens = tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_tokens["input_ids"].shape[1]
        full_text = prompt + trace
        tokens = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        input_ids = tokens["input_ids"].to(device)
        features = extract_features(input_ids, model, eos_token_id, STEP_SIZE, prompt_len=prompt_len)
        num_steps = features.shape[0]
        if num_steps == 0:
            continue

        # Temporal features
        steps = (torch.arange(num_steps, device=features.device).float() + 1) * STEP_SIZE
        t_norm = steps / MAX_LENGTH
        max_probs = features[:, -3]
        max_probs_cpu = max_probs.cpu().tolist()
        recent_avgs = []
        for i in range(num_steps):
            window = max_probs_cpu[max(0, i - 9):i + 1]
            recent_avgs.append(sum(window) / len(window))
        trends = [0.0] * num_steps
        for i in range(10, num_steps):
            trends[i] = max_probs_cpu[i] - max_probs_cpu[i - 10]
        extra = torch.tensor([t_norm.cpu().tolist(), recent_avgs, trends], device="cpu").T
        features_aug = torch.cat([features.cpu(), extra], dim=1)

        all_X.append(features_aug)
        all_y.append(label)

    return all_X, all_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading training data from {TRAIN_DATA}...")
    with open(TRAIN_DATA, 'r') as f:
        train_data = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(train_data)} samples")

    print(f"Loading {MODEL_NAME} + adapter...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    if os.path.exists(args.adapter):
        base_model = PeftModel.from_pretrained(base_model, args.adapter)
        print(f"Loaded adapter from {args.adapter}")
    base_model.eval()

    X_train, y_train = build_dataset(train_data, base_model, tokenizer, device)
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    if not X_train:
        print("ERROR: No features extracted!")
        return

    print(f"\nTraining predictor ({FIXED_PARAMS['epochs']} epochs)...")
    X_flat = torch.cat(X_train)
    y_flat = torch.tensor([y_train[i] for i in range(len(y_train)) for _ in range(len(X_train[i]))]).float()
    mean, std = X_flat.mean(0), X_flat.std(0)

    # Weight positive class heavily to maximize recall
    num_pos = (y_flat == 1).sum().item()
    num_neg = (y_flat == 0).sum().item()
    pos_weight = torch.tensor([(num_neg / max(num_pos, 1)) * 10.0]).float().to(device)

    model = Predictor(X_flat.shape[1], hidden_dim=FIXED_PARAMS["hidden_dim"],
                      dropout=FIXED_PARAMS["dropout"], mean=mean, std=std).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=FIXED_PARAMS["lr"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader = DataLoader(TensorDataset(X_flat, y_flat), batch_size=256, shuffle=True)

    for epoch in range(FIXED_PARAMS["epochs"]):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}, Loss: {total_loss / len(loader):.4f}")

    # Optuna threshold calibration
    print("\nRunning Optuna threshold search...")
    model.eval()
    val_scores = []
    with torch.no_grad():
        for x_seq in X_train:
            scores = torch.sigmoid(model(x_seq.to(device))).cpu().tolist()
            val_scores.append(scores)

    def simulate_early_stop(scores_list, labels, t_high, t_low, t_fall):
        preds = []
        for scores in scores_list:
            p = -1
            for i in range(len(scores)):
                tokens_generated = (i + 1) * 16
                if tokens_generated >= 256:
                    recent = scores[max(0, i + 1 - 20):i + 1]
                    avg = sum(recent) / len(recent)
                    if avg > t_high:
                        p = 1
                        break
                    elif avg < t_low:
                        p = 0
                        break
            if p == -1:
                recent = scores[-10:]
                avg = sum(recent) / len(recent) if recent else 0
                p = 1 if avg > t_fall else 0
            preds.append(p)
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        score = (tp * 1.0) + (tn * 0.05) - (fn * 10.0) - (fp * 1.0)
        return score

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        t_high = trial.suggest_float("threshold_high", 0.2, 0.95)
        t_low = trial.suggest_float("threshold_low", -0.1, 0.01)
        t_fall = trial.suggest_float("threshold_fallback", -0.1, 0.05)
        return simulate_early_stop(val_scores, y_train, t_high, t_low, t_fall)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    best_params = study.best_params
    print(f"Best thresholds found: {best_params}")
    print(f"Best score: {study.best_value:.4f}")

    FIXED_PARAMS.update(best_params)

    torch.save({"model": model.state_dict(), "config": FIXED_PARAMS, "mean": mean, "std": std}, OUTPUT_PATH)
    print(f"Predictor saved to {OUTPUT_PATH} (dim={X_flat.shape[1]})")


if __name__ == "__main__":
    main()
