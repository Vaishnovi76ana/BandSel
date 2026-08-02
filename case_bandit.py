import sys
import os
import re
import json
import torch
import numpy as np
from copy import deepcopy
from time import time
from tqdm import tqdm
from typing import List, Tuple, Dict
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import argparse

from bandit_core import SetBanditDataManager, LinearBandit, GapIndexSwapManager, FEATURE_COLUMNS, save_results, prepare_finetuning_data, compute_reward


def read_data(path):
    if path.endswith("json"):
        data = json.load(open(path, "r"))
    elif path.endswith("jsonl"):
        data = []
        with open(path, "r") as file:
            for line in file:
                line = json.loads(line)
                data.append(line)
    else:
        raise NotImplementedError()
    return data

from eval.utils import generate_completions
from data_processing.process_utils import *
from data_processing.answer_extraction import *
from eval.eval_script import *

@dataclass
class SFTConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: str = "outputs/Qwen2.5-7B-Instruct/bandit/math"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    max_seq_length: int = 1024
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


class QwenSFTTrainer:
    def __init__(self, config=None):
        self.config = config or SFTConfig()
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self._initial_lora_state = None
    
    def load_model(self):
        print(f"Loading {self.config.model_name}...", flush=True)
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
        self.model = prepare_model_for_kbit_training(self.model)
        lora_config = LoraConfig(r=self.config.lora_r, lora_alpha=self.config.lora_alpha, lora_dropout=self.config.lora_dropout, target_modules=self.config.target_modules, bias="none", task_type="CAUSAL_LM")
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        self.is_loaded = True

    def reset_adapter(self):
        if self._initial_lora_state is None: return
        ct = 0
        for name, param in self.model.named_parameters():
            if name in self._initial_lora_state:
                param.data.copy_(self._initial_lora_state[name].to(param.device))
                ct += 1
        print(f'  Reset {ct} LoRA params', flush=True)

    def train(self, training_samples, output_dir, iteration=0):
        if not self.is_loaded: self.load_model()
        texts = []
        for s in training_samples:
            text = s.get('prompt', '') + s.get('output', '') + '<|im_end|>\n'
            texts.append(text)
        train_dataset = Dataset.from_dict({'text': texts})
        adapter_path = os.path.join(output_dir, f'adapter_iter_{iteration}')
        os.makedirs(adapter_path, exist_ok=True)
        args = TrainingArguments(output_dir=adapter_path, num_train_epochs=self.config.num_train_epochs, per_device_train_batch_size=self.config.per_device_train_batch_size, gradient_accumulation_steps=self.config.gradient_accumulation_steps, learning_rate=self.config.learning_rate, warmup_ratio=self.config.warmup_ratio, logging_steps=10, save_strategy='no', fp16=True, optim='paged_adamw_8bit', report_to='none', remove_unused_columns=False)
        def tok_fn(ex): return self.tokenizer(ex['text'], truncation=True, max_length=self.config.max_seq_length, padding='max_length')
        tok_ds = train_dataset.map(tok_fn, batched=True, remove_columns=['text'])
        print(f'Training iter {iteration} ({len(texts)} samples, {self.config.num_train_epochs} epochs)...', flush=True)
        trainer = Trainer(model=self.model, args=args, train_dataset=tok_ds, data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False))
        trainer.train()
        self.model.save_pretrained(adapter_path)
        print(f'Saved to {adapter_path}', flush=True)
        return adapter_path
    

    def infer(self, test_data, answer_extraction_fn, max_new_tokens=1024):

        compression_ratio = 1.0
        prompts = []
        model = self.model
        tokenizer = self.tokenizer

        for example in test_data:
            prompt = ""
            for mess in example['messages']:
                if mess['role'] == 'user':

                    if compression_ratio < 1.0:
                        prompt += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{}.\n" + \
                                  f"{mess['content']}<|eot_id|>{compression_ratio}<|eot_id|><|im_end|>\n<|im_start|>assistant\n"
                    else:
                        prompt += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{}.\n" + \
                                  f"{mess['content']}<|im_end|>\n<|im_start|>assistant\n"

                elif mess['role'] == 'assistant':
                    prompt += mess['content'].rstrip()

                prompt = prompt.lstrip()

            example['prompt'] = prompt
            prompts.append(prompt)

        print(f"\nRunning inference on {len(prompts)} samples...")

        tokenizer.padding_side = "left"

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)

        torch.cuda.synchronize()
        start_time = time()

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

        torch.cuda.synchronize()
        total_time = time() - start_time

        prompt_len = inputs["input_ids"].shape[1]

        model_outputs = tokenizer.batch_decode(
            outputs[:, prompt_len:],
            skip_special_tokens=True
        )

        cot_lengths = []
        for model_completion in model_outputs:
            cot = model_completion.split('\n\nThe final answer is:')[0]
            cot_length = tokenizer(cot, return_tensors="pt")['input_ids'].shape[1]
            cot_lengths.append(cot_length)

        predictions = [
            eval(answer_extraction_fn)(
                item['messages'][-2]['content'],
                output,
                task='cot'
            )
            for item, output in tqdm(
                zip(test_data, model_outputs),
                desc="Extracting answers",
                total=len(model_outputs)
            )
        ]

        print("\nEvaluating predictions...")

        results = []

        pbar = tqdm(
            zip(test_data, model_outputs, predictions, cot_lengths),
            total=len(model_outputs),
            desc="Evaluating outputs"
        )

        for example, output, pred, cot_length in pbar:

            item = deepcopy(example)

            item.update({
                'model_output': output,
                'prediction': pred,
                'cot_length': cot_length,
            })

            if len(pred) == 0:
                item['accuracy'] = False
            else:
                item['accuracy'] = eval_math(item)

            results.append(item)

            elapsed = time() - start_time
            avg_time = elapsed / len(results)

            pbar.set_postfix({
                "avg_s/sample": f"{avg_time:.3f}"
            })

        print("\nCalculating accuracy...")

        acc = sum(item['accuracy'] for item in results) / len(results)
        avg_cot_length = sum(item['cot_length'] for item in results) / len(results)

        print(f"Accuracy = {acc*100:.5f}")
        print(f"Avg CoT Length = {avg_cot_length:.5f}")
        print(f"Sample latency = {total_time/len(test_data):.5f}")

        return {
            "results": results,
            "accuracy": acc,
            "avg_cot_length": avg_cot_length,
            "sample_latency": total_time/len(test_data),
            "total_time": total_time
        }

    @torch.no_grad()
    
    def evaluate(self, val_samples,  output_dir, max_new_tokens=1024, batch_size=8):
        """Batched evaluation: processes batch_size samples in parallel for ~4-8x speedup."""
        
        self.model.eval()
        total = len(val_samples)
        correct_count = 0
        actual_crs = []
        answer_extraction_fn = "extract_math_answer"

        num_batches = (total + batch_size - 1) // batch_size
        print(f"\n>>> EVALUATING {total} SAMPLES in {num_batches} batches (bs={batch_size}) <<<", flush=True)

        for batch_idx in range(num_batches):

            start = batch_idx * batch_size
            end = min(start + batch_size, total)
            batch_samples = val_samples[start:end]

            # Convert batch samples into infer-compatible format
            test_data = batch_samples
            results = self.infer(test_data, answer_extraction_fn)

            for j, item in enumerate(results["results"]):
                generated = item["model_output"]
                gen_len = item["cot_length"]
                orig_len = batch_samples[j]["cot_length"]
                actual_cr = gen_len / orig_len if orig_len > 0 else 1.0
                actual_crs.append(actual_cr)

                accuracy = item["accuracy"]

                if accuracy:
                    correct_count += 1

                print(f"  [{start+j+1}/{total}] CR={actual_cr:.2f}, correct={accuracy}", flush=True)

            print(f"  -- batch {batch_idx+1}/{num_batches} done", flush=True)
            
        with open(f"{args.output_dir}/predictions.jsonl", "w") as f:
            for p in results:
                f.write(json.dumps(p) + "\n")
        accuracy = correct_count / total if total > 0 else 0
        avg_cr = sum(actual_crs) / len(actual_crs) if actual_crs else 1.0

        print(f"\n>>> Acc={accuracy:.2%} ({correct_count}/{total}), Avg_CR={avg_cr:.4f} <<<\n", flush=True)

        return {"accuracy": accuracy, "avg_cr": avg_cr}

def bandit_run(args, CSV_PATH, BANDIT_PATH, ORG_PATH):
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    total_arms = args.num_total_traces // args.arm_size
    rest = total_arms - args.num_training_arms - args.num_challenger_arms
    print(f"Arms: {total_arms} total, {args.num_training_arms} U_t, {args.num_challenger_arms}    N_t, {rest} rest pool")
    print(f"M_t exploration: {args.num_exploration} arms sampled per round")
    print(f"Reward: accuracy + {args.beta} * (1 - avg_CR)")
    print(f"Mode: FULL CASE ALGORITHM with M_t exploration")

    print("=" * 60)
    print("CASE BANDIT: Algorithm 1 — CASE with M_t Exploration")
    print("=" * 60)

    data_manager = SetBanditDataManager(CSV_PATH, BANDIT_PATH, ORG_PATH, arm_size=args.arm_size,
    num_training_arms=args.num_training_arms, num_challenger_arms=args.num_challenger_arms,
    num_total_traces=args.num_total_traces, num_validation=args.num_validation,
    num_exploration=args.num_exploration)
    feature_dim = len(FEATURE_COLUMNS)
    bandit = LinearBandit(feature_dim, args.lambda_reg,learning_rate=1.0, random_seed = args.random_seed)
    data_manager.form_arms_and_initialize(bandit.alpha, args.random_seed)
    swap_manager = GapIndexSwapManager(confidence_scale=args.confidence_scale)
    rng = np.random.RandomState(args.random_seed)

    print("\nInitial arm scores:")
    for aid in sorted(data_manager.all_arms.keys(), key=lambda x: int(x.split('_')[1])):
        arm = data_manager.all_arms[aid]
        if aid in data_manager.training_arms:
            loc = "U_t"
        elif aid in data_manager.challenger_arms:
            loc = "N_t"
        else:
            loc = "rest"
        print(f"  {aid}: score={arm.score:.4f} [{loc}]")

    skip_sft = False
    if args.skip_sft == 1:
        skip_sft = True
    if not skip_sft:
        print("\nLoading model...", flush=True)
        trainer = QwenSFTTrainer(SFTConfig(model_name=args.model_path))
        trainer.load_model()

    history = []
    #val_samples = data_manager.get_validation_samples()[:args.val_eval_size]
    #val_data = read_data("/kaggle/input/datasets/vaishnoviarun/bandit-dataset/predictions.jsonl")
    #val_samples = val_data[:args.val_eval_size]
    val_samples = [
    s.original_data
    for s in data_manager.get_validation_samples()[:args.val_eval_size]
    ]


    for iteration in range(1, args.max_iterations + 1):
        print(f"\n{'#' * 60}")
        print(f"# ITERATION {iteration}/{args.max_iterations}")
        print(f"{'#' * 60}", flush=True)

        # === Step 8-13: SWAP worst(U_t) vs best(N_t) — NO M_t ===
        print(f"\n[Step 8-13] Swap check (U_t vs N_t)...", flush=True)
        swapped, removed, added, gap, gap_idx = swap_manager.execute_swap(data_manager, bandit)

        # === Step 14: Sample M_t from (U_t ∪ N_t)^c ===
        mt_ids = data_manager.sample_exploration_set(rng)
        print(f"\n[Step 14] M_t sampled: {mt_ids} ({len(mt_ids)} arms from rest)", flush=True)

        # === Step 15: Reconstruct N_t = top_m'(M_t ∪ old_N_t) ===
        print(f"\n[Step 15] Reconstruct N_t...", flush=True)
        swap_manager.reconstruct_nt(data_manager, mt_ids)
        print(f"    Sets: U_t={len(data_manager.training_arms)}, N_t={len(data_manager.challenger_arms)}, Rest={len(data_manager.rest_arms)}", flush=True)
    
        # === Step 20: CASE selection from U_t ∪ N_t ===
        print(f"\n[Step 20] CASE arm selection (U_t ∪ N_t)...", flush=True)
        pulled_arm = swap_manager.select_most_uncertain_arm(data_manager, bandit)
        arm_samples = data_manager.get_arm_samples(pulled_arm)
        training_data = prepare_finetuning_data(arm_samples)
        loc = "U_t" if pulled_arm.id in data_manager.training_arms else "N_t"
        print(f"    -> {pulled_arm.id} [{loc}] ({len(arm_samples)} traces, pull #{pulled_arm.n_pulls + 1})", flush=True)
    
        # === Step 21-23: Pull arm (SFT + eval), get reward, update α ===
        if not skip_sft:
            print(f"\n[Step 21] Resetting LoRA adapter...", flush=True)
            trainer.reset_adapter()
            print(f"\n[Step 21] SFT on {len(training_data)} traces...", flush=True)
            trainer.train(training_data, args.output_dir, iteration)
            print(f"\n[Step 21] Evaluating {len(val_samples)} val samples...", flush=True)
            result = trainer.evaluate(val_samples, args.output_dir, args.max_new_tokens, args.eval_batch_size)
            accuracy, avg_cr = result["accuracy"], result["avg_cr"]
        else:
            accuracy = np.mean([1.0 if s.accuracy else 0.0 for s in arm_samples])
            avg_cr = np.mean([s.optimal_compression_ratio for s in arm_samples])

        # === Step 22-23: Update reward + ridge regression ===
        reward = compute_reward(accuracy, avg_cr, args.beta)
        pulled_arm.record_pull(reward)
        bandit.update_weights(pulled_arm.avg_features, reward)
        data_manager.update_arm_scores(bandit.alpha)

        print(f"\n[Step 22-23] {pulled_arm.id}: acc={accuracy:.4f}, CR={avg_cr:.4f}, REWARD={reward:.4f}    (pulls: {pulled_arm.n_pulls})", flush=True)
        print(f"    Weights: [{', '.join(f'{a:.4f}' for a in bandit.alpha)}]", flush=True)
    
        # === Calculate Tracking Metrics ===
        predicted_reward = np.dot(pulled_arm.avg_features, bandit.alpha)
        prediction_error = abs(predicted_reward - reward)
        ut_scores = [data_manager.all_arms[aid].score for aid in data_manager.training_arms]
        ut_min, ut_max, ut_median = np.min(ut_scores), np.max(ut_scores), np.median(ut_scores)

        history.append({"iter": iteration, "arm": pulled_arm.id, "arm_set": loc,
                    "reward": reward, "acc": accuracy, "cr": avg_cr,
                    "weights": bandit.alpha.tolist(),
                    "swapped": swapped, "gap": gap, "gap_index": gap_idx,
                    "mt_ids": list(mt_ids),
                    "prediction_error": prediction_error,
                    "ut_min": ut_min, "ut_max": ut_max, "ut_median": ut_median})
    
        # === Step 7: Convergence check ===
        if swap_manager.has_converged(data_manager, bandit):
            print(f"\nConverged at iteration {iteration}!", flush=True)
            break

    print("\n" + "=" * 60)
    save_results(args.output_dir, data_manager, history, bandit.alpha, FEATURE_COLUMNS)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/Qwen2.5-7B-Instruct/bandit/math", help="default to `model_path`_predictions")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer-path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-name", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--model-type", type=str, choices=['llama3', 'qwen'], default="qwen")
    parser.add_argument("--benchmark", type=str, choices=['gsm8k', 'math'], default="gsm8k")
    parser.add_argument("--max_num_examples", type=int, default=100000000000000, help="maximum number of examples to evaluate.")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--eval_batch_size", type=int, default=32, help="batch size for evaluation.")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--arm_size", type=int, default=25, help="size of each bandit arm")
    parser.add_argument("--num_training_arms", type=int, default=10, help="number of training arms")
    parser.add_argument("--num_challenger_arms", type=int, default=10, help="number of challenger arms")
    parser.add_argument("--num_total_traces", type=int, default=1000, help="number of reasoning traces")
    parser.add_argument("--num_validation", type=int, default=1000, help="number of validation traces")
    parser.add_argument("--num_exploration", type=int, default=5, help="number of exploration traces")
    parser.add_argument("--max_iterations", type=int, default=30, help="maximum number of bandit iterations")
    parser.add_argument("--lambda_reg", type=float, default=1.0, help="parameter")
    parser.add_argument("--confidence_scale", type=float, default=1.0, help="parameter")
    parser.add_argument("--beta", type=float, default=0.5, help="parameter")
    parser.add_argument("--val_eval_size", type=int, default=50, help="parameter")
    parser.add_argument("--skip_sft", type=int, default=0, help="1 for true, 0 for false")
    args, unparsed_args = parser.parse_known_args()

    CSV_PATH = f"outputs/{args.model_name}/bandit/{args.benchmark}/reasoning_scores_subset.csv"
    BANDIT_PATH = f"outputs/{args.model_name}/bandit/{args.benchmark}/bandit_data.jsonl"
    ORG_PATH = f"outputs/{args.model_name}/tokenskip/{args.benchmark}/Original/train/samples/predictions.jsonl"
    
    # Check dataset availability
    print(f"Ready: CSV={os.path.exists(CSV_PATH)}, JSONL={os.path.exists(BANDIT_PATH)}")
    bandit_run(args, CSV_PATH, BANDIT_PATH, ORG_PATH)
