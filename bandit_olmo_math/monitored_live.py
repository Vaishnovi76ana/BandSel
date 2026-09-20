#!/usr/bin/env python
"""
Live monitored inference with dynamic EOS early stopping for OLMo-3-7B-Instruct.
Loads base model + LoRA adapter + trained MLP predictor, runs token-by-token
generation, and stops early when the predictor signals the answer is complete.
Adapted from sagnibha/monitored_live_bfloat16.py for OLMo architecture.
"""
import os, sys, re, torch, json, time
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from eval_utils import extract_math_answer, eval_math


# --- Predictor architecture (must match fast_retrain_predictor.py) ---
class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.1, mean=None, std=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
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


class LiveEOSController:
    """Loads the trained predictor and tracks per-batch generation state."""

    def __init__(self, model_path, device="cuda"):
        checkpoint = torch.load(model_path, map_location=device)
        self.params = checkpoint["config"]
        input_dim = checkpoint["mean"].shape[0]
        self.model = Predictor(
            input_dim,
            hidden_dim=self.params["hidden_dim"],
            dropout=0.0,
            mean=checkpoint["mean"],
            std=checkpoint["std"]
        ).to(device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.device = device
        self.reset()

    def reset(self, batch_size=1):
        self.batch_size = batch_size
        self.history_max_prob = [[] for _ in range(batch_size)]
        self.history_entropy = [[] for _ in range(batch_size)]
        self.history_eos_prob = [[] for _ in range(batch_size)]
        self.all_scores = [[] for _ in range(batch_size)]

    @torch.no_grad()
    def check_score_batch(self, h_states, logits, eos_token_id, step, total_steps, active_mask=None):
        B = h_states.shape[0]
        probs = F.softmax(torch.clamp(logits, -50, 50), dim=-1)
        probs = torch.clamp(probs, min=1e-6, max=1.0)

        if isinstance(eos_token_id, list):
            eos_probs = probs[:, eos_token_id].sum(dim=-1)
        else:
            eos_probs = probs[:, eos_token_id]

        max_probs, _ = probs.max(dim=-1)
        entropies = -(probs * torch.log(probs)).sum(dim=-1)

        features_list = []
        valid_indices = []

        for i in range(B):
            if active_mask is not None and not active_mask[i]:
                self.all_scores[i].append(0.0)
                continue

            eos_prob = eos_probs[i].item()
            max_prob = max_probs[i].item()
            entropy = entropies[i].item()

            entropy_slope = entropy - (self.history_entropy[i][-1] if self.history_entropy[i] else entropy)
            eos_slope = eos_prob - (self.history_eos_prob[i][-1] if self.history_eos_prob[i] else eos_prob)

            self.history_max_prob[i].append(max_prob)
            self.history_entropy[i].append(entropy)
            self.history_eos_prob[i].append(eos_prob)

            t_norm = step / total_steps
            recent_window = self.history_max_prob[i][-10:]
            avg_recent = sum(recent_window) / len(recent_window) if recent_window else 0.0
            trend_10 = max_prob - (self.history_max_prob[i][-11] if len(self.history_max_prob[i]) > 10 else self.history_max_prob[i][0])

            base_features = torch.cat([
                h_states[i],
                torch.tensor([eos_prob, entropy, max_prob, entropy_slope, eos_slope], device=self.device)
            ], dim=0)

            extra_features = torch.tensor([t_norm, avg_recent, trend_10], device=self.device)
            full_features = torch.cat([base_features, extra_features], dim=0)

            features_list.append(full_features)
            valid_indices.append(i)

        if features_list:
            batch_features = torch.stack(features_list)
            logits_out = self.model(batch_features)
            scores = torch.sigmoid(logits_out).squeeze(-1).tolist()
            if not isinstance(scores, list):
                scores = [scores]

            for idx, score in zip(valid_indices, scores):
                self.all_scores[idx].append(score)


def has_repetition(token_ids, min_len=6, max_len=50, min_repeats=3):
    n = len(token_ids)
    for l in range(min_len, max_len + 1):
        if n < l * min_repeats:
            continue
        last_chunk = token_ids[-l:]
        match = True
        for r in range(1, min_repeats):
            chunk = token_ids[-l * (r + 1) : -l * r]
            if chunk != last_chunk:
                match = False
                break
        if match:
            return True
    return False


# --- Inference Runner ---
class LiveMonitoredInference:
    def __init__(self, model_name, adapter_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading Base Model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
            attn_implementation="sdpa"
        )

        if adapter_path and os.path.exists(adapter_path):
            from peft import PeftModel
            print(f"Loading Adapter from {adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()

    @torch.no_grad()
    def run_eval(self, test_data, eos_ctrl, max_new_tokens=2048, batch_size=16):
        results = []
        eos_token_id = self.tokenizer.eos_token_id  # 100257 for OLMo
        if not isinstance(eos_token_id, list):
            eos_token_id = [eos_token_id]

        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"\nStarting Live Monitored Inference on {len(test_data)} samples with batch size {batch_size}...")

        for i in range(0, len(test_data), batch_size):
            batch_items = test_data[i:i + batch_size]
            B = len(batch_items)
            prompts = []
            for item in batch_items:
                prompt = item.get("prompt", "")
                if not prompt and "messages" in item:
                    # Extract user content from messages
                    for msg in item["messages"]:
                        if msg["role"] == "user":
                            prompt = format_olmo_prompt(msg["content"])
                            break
                prompts.append(prompt)

            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
            generated_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            current_kv_cache = None

            eos_ctrl.reset(batch_size=B)
            stopped_early = [False] * B
            prediction = [-1] * B
            stop_step = [max_new_tokens] * B
            is_finished = [False] * B
            total_tokens = [max_new_tokens] * B

            print(f"[{i + 1}-{i + B}/{len(test_data)}] Gen...", end="", flush=True)

            for step in range(max_new_tokens):
                outputs = self.model(
                    input_ids=generated_ids if step == 0 else next_token_ids,
                    attention_mask=attention_mask,
                    past_key_values=current_kv_cache,
                    use_cache=True,
                    output_hidden_states=True
                )

                logits = outputs.logits[:, -1, :]
                next_token_ids = torch.argmax(logits, dim=-1).unsqueeze(-1)
                current_kv_cache = outputs.past_key_values
                generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((B, 1), device=attention_mask.device)
                ], dim=-1)

                # Check EOS predictor every 16 steps
                if (step + 1) % 16 == 0:
                    h_states = outputs.hidden_states[-1][:, -1, :]
                    eos_ctrl.check_score_batch(
                        h_states, logits, eos_token_id, step, max_new_tokens,
                        active_mask=[not f for f in is_finished]
                    )

                    if (step + 1) >= 128:
                        for b_idx in range(B):
                            if is_finished[b_idx]:
                                continue

                            scores = eos_ctrl.all_scores[b_idx]
                            recent_avg = sum(scores[-20:]) / len(scores[-20:])

                            if recent_avg > eos_ctrl.params["threshold_high"]:
                                prediction[b_idx] = 1
                            elif recent_avg < eos_ctrl.params["threshold_low"]:
                                prediction[b_idx] = 0
                                stopped_early[b_idx] = True
                                stop_step[b_idx] = step + 1
                                total_tokens[b_idx] = step + 1
                                is_finished[b_idx] = True

                # Check for repetition loop
                for b_idx in range(B):
                    if is_finished[b_idx]:
                        continue
                    tokens_gen = generated_ids[b_idx, inputs["input_ids"].shape[1]:].tolist()
                    if has_repetition(tokens_gen, min_len=6, max_len=50, min_repeats=3):
                        prediction[b_idx] = 0
                        stopped_early[b_idx] = True
                        stop_step[b_idx] = step + 1
                        total_tokens[b_idx] = step + 1
                        is_finished[b_idx] = True

                # Check for natural EOS token
                for b_idx in range(B):
                    if not is_finished[b_idx] and next_token_ids[b_idx].item() in eos_token_id:
                        stop_step[b_idx] = step + 1
                        total_tokens[b_idx] = step + 1
                        is_finished[b_idx] = True

                if all(is_finished):
                    break

            # Handle unfinished sequences
            for b_idx in range(B):
                if not is_finished[b_idx]:
                    total_tokens[b_idx] = step + 1

            # Fallback predictions
            for b_idx in range(B):
                if prediction[b_idx] == -1:
                    recent = eos_ctrl.all_scores[b_idx][-10:]
                    avg = sum(recent) / len(recent) if recent else 0
                    prediction[b_idx] = 1 if avg > eos_ctrl.params["threshold_fallback"] else 0

            # Decode and evaluate
            for b_idx, item in enumerate(batch_items):
                start_idx = inputs["input_ids"].shape[1]
                end_idx = start_idx + stop_step[b_idx]
                model_output = self.tokenizer.decode(
                    generated_ids[b_idx, start_idx:end_idx], skip_special_tokens=True
                )

                query = item.get("prompt", "")
                base_accuracy = item.get("accuracy")
                base_answer = item.get("prediction")
                true_answer = item.get("answer")
                pred_answer = extract_math_answer(query, model_output, task='cot')

                if isinstance(base_answer, list):
                    base_answer = base_answer[0] if len(base_answer) > 0 else ""
                if isinstance(pred_answer, list):
                    pred_answer = pred_answer[0] if len(pred_answer) > 0 else ""
                if isinstance(true_answer, list):
                    true_answer = true_answer[0] if len(true_answer) > 0 else ""

                base_answer = str(base_answer or "")
                pred_answer = str(pred_answer or "")
                true_answer = str(true_answer or "")

                match = pred_answer.strip().lower() == true_answer.strip().lower()
                try:
                    eval_item = {"answer": true_answer, "prediction": pred_answer}
                    match = eval_math(eval_item)
                except (NotImplementedError, Exception):
                    pass

                orig_tokens = item.get("cot_length", item.get("total_tokens", stop_step[b_idx]))
                save_pct = 100.0 - (stop_step[b_idx] / max(orig_tokens, 1)) * 100.0

                results.append({
                    "id": item.get("id", i + b_idx),
                    "prompt": item.get("prompt", ""),
                    "answer": true_answer,
                    "base_answer": base_answer,
                    "prediction": pred_answer,
                    "base_accuracy": base_accuracy,
                    "eos_accuracy": match,
                    "predicted_label": prediction[b_idx],
                    "stopped_early": stopped_early[b_idx],
                    "stop_token": stop_step[b_idx],
                    "total_tokens": orig_tokens,
                    "output": model_output
                })
                print(f" Done {i + b_idx + 1}. [P={prediction[b_idx]}, GT={int(base_accuracy)}, Save={save_pct:.1f}%]")

        return results


def format_olmo_prompt(question):
    """Format a math question in OLMo chat template."""
    return (
        f"<|user|>\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n"
        f"{question}\n<|assistant|>\n"
    )


def summarize(results):
    total = len(results)
    correct_actual = sum(1 for r in results if r['eos_accuracy'])

    pred_correct = sum(1 for r in results if int(r['eos_accuracy']) == r['base_accuracy'])

    stats = {"TP": [], "TN": [], "FP": [], "FN": []}
    for r in results:
        label = int(r['base_accuracy'])
        pred = r['predicted_label']
        token = r['stop_token']
        if label == 1 and pred == 1: stats["TP"].append(token)
        elif label == 0 and pred == 0: stats["TN"].append(token)
        elif label == 0 and pred == 1: stats["FP"].append(token)
        elif label == 1 and pred == 0: stats["FN"].append(token)

    avg_savings = sum(1.0 - (r['stop_token'] / max(r['total_tokens'], 1)) for r in results) / total

    print("\n" + "=" * 40)
    print("      LIVE MONITORED SUMMARY        ")
    print("=" * 40)
    print(f"Total Questions:    {total}")
    print(f"Actual Accuracy:    {correct_actual / total * 100:.2f}%")
    print(f"Predictor Accuracy: {pred_correct / total * 100:.2f}%")
    print(f"Avg Token Savings:  {avg_savings * 100:.2f}%")

    print("\nConfusion Matrix (Predictor vs Reality):")
    print(f"            Reality: T    Reality: F")
    print(f"Pred: T      {len(stats['TP']):<12} {len(stats['FP']):<12} (Success calls)")
    print(f"Pred: F      {len(stats['FN']):<12} {len(stats['TN']):<12} (Failure calls)")

    print("\nAvg Stop Token by Category:")
    for cat, vals in stats.items():
        avg = sum(vals) / len(vals) if vals else 0
        print(f"  {cat} (n={len(vals)}): {avg:.1f} tokens")
    print("=" * 40)


if __name__ == "__main__":
    TEST_DATA_PATH = "dataset/predictions.jsonl"
    EOS_MODEL_PATH = "predictor_best_optimized.pth"
    ADAPTER_PATH = "bandit_output/adapter_iter_FINAL"
    MODEL_NAME = "allenai/Olmo-3-7B-Instruct"

    # 1. Load Data
    data = []
    with open(TEST_DATA_PATH, "r") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                # Format prompts for OLMo
                if "messages" in d:
                    for msg in d["messages"]:
                        if msg["role"] == "user":
                            d["prompt"] = format_olmo_prompt(msg["content"])
                            break
                data.append(d)

    # 2. Setup
    eos_ctrl = LiveEOSController(EOS_MODEL_PATH)
    eos_ctrl.params['threshold_low'] = 0.08  # Reduce false negatives
    print(f"Monitor Thresholds: High={eos_ctrl.params['threshold_high']:.3f}, "
          f"Low={eos_ctrl.params['threshold_low']:.3f}, "
          f"Fallback={eos_ctrl.params['threshold_fallback']:.3f}")
    runner = LiveMonitoredInference(MODEL_NAME, ADAPTER_PATH)

    # 3. Run
    results = runner.run_eval(data, eos_ctrl, batch_size=16)

    # 4. Summary
    summarize(results)

    # 5. Save details
    os.makedirs("bandit_output", exist_ok=True)
    save_path = "bandit_output/live_monitored_results.jsonl"
    with open(save_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nDetailed results saved to {save_path}")
