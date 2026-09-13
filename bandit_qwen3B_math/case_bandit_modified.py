"""
CASE Bandit Algorithm for Qwen2.5-7B-Instruct on MATH-500.

Co-Active Selection and Exploration (CASE) linear contextual bandit
for selecting optimal training data subsets.

Adapted from Sagnibha's case_bandit_bfloat16.py for Qwen/Phi4,
with Qwen-specific prompt formatting and LoRA target modules.

Usage:
    python case_bandit.py                    # Full run with SFT
    python case_bandit.py --skip_sft         # Dry run (no model training)
"""
import sys
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import re
import json
import torch
import numpy as np

import pandas as pd
from copy import deepcopy
from time import time
from peft import PeftModel, LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, GenerationConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from pathlib import Path
from datasets import Dataset
from datetime import datetime



HAS_VLLM = False



from eval_utils import extract_math_answer, eval_math


# ═══════════════════════════════════════════════════════════════════
# File paths
# ═══════════════════════════════════════════════════════════════════
CSV_PATH = "dataset/reasoning_scores.csv"
BANDIT_PATH = "dataset/bandit_data.jsonl"
VAL_DATA_PATH = "dataset/orig_test.jsonl"
ORIG_PATH = "dataset/orig_train.jsonl"
OUTPUT_DIR = "bandit_output_beta_05_new"
BASE_ACCURACY_PATH = "dataset/metrics.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Ready: CSV={os.path.exists(CSV_PATH)}, BANDIT={os.path.exists(BANDIT_PATH)}, VAL={os.path.exists(VAL_DATA_PATH)}")

accuracy_base = 0
with open(BASE_ACCURACY_PATH, "r", encoding="utf-8") as f:
    metric_data = json.load(f)
    accuracy_base = metric_data["accuracy"]

print(f"Base accuracy: {accuracy_base} ")



# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════
def read_data(path):
    if path.endswith("json"):
        data = json.load(open(path, "r"))
    elif path.endswith("jsonl"):
        data = []
        with open(path, "r") as file:
            for line in file:
                data.append(json.loads(line))
    else:
        raise NotImplementedError()
    return data

def iterative_inversion(A, x):
    return A - np.outer(A @ x, x @ A) / (1.0 + x @ A @ x)


FEATURE_COLUMNS = [
    'signal_ratio',
    'distinct_1_ratio', 'distinct_2_ratio', 'distinct_3_ratio', 'distinct_4_ratio',
    'top_k_mass_ratio', 'windowed_jaccard',
    'longest_repeat_ratio', 'llm_judge_score'
]


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for f in self.files:
            f.write(text)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f"case_output_{timestamp}.txt"

log_file = open(log_path, "w", buffering=1)

sys.stdout = Tee(sys.__stdout__, log_file)


# ═══════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════
@dataclass
class Sample:
    id: str
    features: np.ndarray
    accuracy: bool
    original_cot_length: int
    optimal_compression_ratio: float
    prompt: str = ""
    answer: str = ""
    compressed_trace: str = ""

    def to_dict(self):
        return {"id": self.id, "accuracy": self.accuracy,
                "original_cot_length": self.original_cot_length,
                "optimal_compression_ratio": self.optimal_compression_ratio,
                "prompt": self.prompt, "answer": self.answer,
                "compressed_trace": self.compressed_trace}


@dataclass
class Arm:
    id: str
    sample_ids: List[str]
    avg_features: np.ndarray
    score: float = 0.0
    n_pulls: int = 0
    total_reward: float = 0.0

    @property
    def avg_reward(self):
        return self.total_reward / self.n_pulls if self.n_pulls > 0 else 0.0

    def update_score(self, alpha):
        self.score = float(np.dot(alpha, self.avg_features))

    def record_pull(self, reward):
        self.n_pulls += 1
        self.total_reward += reward

    def to_dict(self):
        return {"id": self.id, "sample_ids": self.sample_ids,
                "score": self.score, "n_pulls": self.n_pulls,
                "total_reward": self.total_reward, "avg_reward": self.avg_reward,
                "avg_features": self.avg_features.tolist()}


# ═══════════════════════════════════════════════════════════════════
# SetBanditDataManager
# ═══════════════════════════════════════════════════════════════════
class SetBanditDataManager:
    def __init__(self, csv_path, bandit_path, orig_path, arm_size=25, num_training_arms=10, num_challenger_arms=10, num_arms=50, num_validation=100):
        self.csv_path = csv_path
        self.bandit_path = bandit_path
        self.orig_path = orig_path
        self.arm_size = arm_size
        self.num_training_arms = num_training_arms
        self.num_challenger_arms = num_challenger_arms
        self.num_arms = num_arms
        self.num_validation = num_validation
        self.all_samples = {}
        self.all_arms = {}
        self.training_arms = set()     # U_t
        self.challenger_arms = set()   # N_t
        self.exploration_pool = set()  # M_t
        self.validation_ids = set()
        self._load_data()
        

    def _load_data(self):
        df_features = pd.read_csv(self.csv_path)
        trace_data = {}
        orig_data = {}
        with open(self.bandit_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                trace_data[data['id']] = data
        with open(self.orig_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                orig_data[data['id']] = data
        for _, row in df_features.iterrows():
            sample_id = str(row['id'])
            if sample_id not in trace_data:
                continue
            trace = trace_data[sample_id]
            orig = orig_data[sample_id]
            features = np.array([row[col] for col in FEATURE_COLUMNS], dtype=np.float32)
            features = np.nan_to_num(features, nan=0.5)
            target_cr = trace.get('optimal_compression_ratio', 1.0)
            raw_prompt = trace.get('prompt', '')
            prefix = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
            if raw_prompt.startswith(prefix):
                user_content = raw_prompt[len(prefix):]
            else:
                user_content = raw_prompt.split("Please reason step by step, and put your final answer within \\boxed{}.\n")[-1]
            user_content = user_content.split("<|im_end|>")[0]
            user_content = user_content.split("<|eot_id|>")[0]
            user_content = user_content.strip()
            
            formatted_prompt = user_content

            self.all_samples[sample_id] = Sample(
                id=sample_id, features=features,
                accuracy=trace.get('accuracy', False),
                original_cot_length=orig.get('cot_length', 0),
                optimal_compression_ratio=target_cr,
                prompt=formatted_prompt,
                answer=trace.get('answer', ''),
                compressed_trace=trace.get('model_output', '')
            )

        print(f"Loaded {len(self.all_samples)} samples with {len(FEATURE_COLUMNS)} features each")


    def form_arms_and_initialize(self, alpha, random_seed=42, max_arm_similarity=0.2, max_attempts=100000):
        rng = np.random.RandomState(random_seed)
        all_ids = list(self.all_samples.keys())
        rng.shuffle(all_ids)
        self.validation_ids = set( all_ids[:self.num_validation])
        remaining = [sid for sid in all_ids if sid not in self.validation_ids]

        if len(remaining) < self.arm_size:
            raise ValueError(
                f"Need at least {self.arm_size} samples, but only {len(remaining)} remain.")

        max_overlap = int(max_arm_similarity * self.arm_size)
        print( f"\nCreating {self.num_arms} arms {self.arm_size} samples" )
        print( f"Maximum pairwise overlap: {max_overlap} samples ({max_arm_similarity:.1%})")
    
        self.all_arms = {}
        attempts = 0

        while len(self.all_arms) < self.num_arms:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(f"Could not create {self.num_arms} arms with maximum overlap of {max_arm_similarity:.1%}. Only created {len(self.all_arms)} arms after {max_attempts} attempts.")

            # -----------------------------------------------------
            # Randomly sample arm_size DISTINCT samples from the SAME remaining pool.
            # -----------------------------------------------------
            chunk = rng.choice(remaining, size=self.arm_size, replace=False).tolist()
            chunk_set = set(chunk)
            
            # -----------------------------------------------------
            # Check overlap with every existing arm
            # -----------------------------------------------------
            valid = True
            for existing_arm in self.all_arms.values():
                existing_set = set(existing_arm.sample_ids)
                overlap = len(chunk_set & existing_set)
                if overlap > max_overlap:
                    valid = False
                    break

            # -----------------------------------------------------
            # Accept the arm only if it satisfies the constraint
            # -----------------------------------------------------
            if not valid:
                continue
            arm_id = f"arm_{len(self.all_arms)}"

            # Calculate average features
            member_features = np.vstack([self.all_samples[sid].features for sid in chunk])

            # Create arm
            self.all_arms[arm_id] = Arm(id=arm_id, sample_ids=chunk,avg_features=member_features.mean(axis=0))

        # ---------------------------------------------------------
        # 5. Score all arms
        # ---------------------------------------------------------
        for arm in self.all_arms.values():
            arm.update_score(alpha)

        # ---------------------------------------------------------
        # 6. Sort arms by score
        # ---------------------------------------------------------
        sorted_arms = sorted(self.all_arms.values(),key=lambda a: a.score, reverse=True)

        # ---------------------------------------------------------
        # 7. Assign training / challenger / rest arms
        # ---------------------------------------------------------
        self.training_arms = set()
        self.challenger_arms = set()
        self.exploration_pool = set()

        for i, arm in enumerate(sorted_arms):
            if i < self.num_training_arms:
                self.training_arms.add(arm.id)
            elif i < (self.num_training_arms + self.num_challenger_arms):
                self.challenger_arms.add(arm.id)
    
        # ---------------------------------------------------------
        # 8. Print summary
        # ---------------------------------------------------------
        print(f"\nFormed {len(self.all_arms)} arms of {self.arm_size} traces each")
        print(f"  - Training arms (U_t): {len(self.training_arms)}")
        print(f"  - Challenger arms (N_t): {len(self.challenger_arms)}")
        print(f"  - Exploration arms (M_t): {len(self.exploration_pool)}")
        print(f"  - Validation samples: {len(self.validation_ids)}")
        total = sum(len(self.all_arms[aid].sample_ids) for aid in self.training_arms)
        print(f"  - Total training samples: {total}")

    def update_arm_scores(self, alpha):
        for arm in self.all_arms.values():
            arm.update_score(alpha)

    def get_random_training_arm(self, rng=None):
        arm_ids = list(self.training_arms)
        idx = rng.choice(len(arm_ids)) if rng else np.random.choice(len(arm_ids))
        return self.all_arms[arm_ids[idx]]

    def get_arm_samples(self, arm):
        return [self.all_samples[sid] for sid in arm.sample_ids]

    def get_all_training_samples(self):
        samples = []
        for arm_id in self.training_arms:
            samples.extend(self.get_arm_samples(self.all_arms[arm_id]))
        return samples

    def get_validation_samples(self):
        return [self.all_samples[sid] for sid in self.validation_ids]

    def get_training_arms_list(self):
        return [self.all_arms[aid] for aid in self.training_arms]

    def get_challenger_arms_list(self):
        return [self.all_arms[aid] for aid in self.challenger_arms]

    def get_exploration_pool_list(self):
        return [self.all_arms[aid] for aid in self.exploration_pool]


# ═══════════════════════════════════════════════════════════════════
# LinearBandit (Ridge Regression)
# ═══════════════════════════════════════════════════════════════════
class LinearBandit:
    def __init__(self, feature_dim=9, random_seed=42):
        self.feature_dim = feature_dim
        rng = np.random.RandomState(random_seed)
        self.alpha = rng.randn(feature_dim)
        self.A_inv = np.eye(feature_dim)
        #self.A_inv = np.eye(feature_dim) / lambda_reg
        self.Xtr = np.zeros(self.feature_dim)

    def predict_score(self, features):
        return float(np.dot(self.alpha, features))

    def update_weights(self, arm_features, reward):

        x = arm_features
        # Update V_(t+1)^(-1)
        # Update V_(t+1)^(-1)
        self.A_inv = iterative_inversion(self.A_inv, x)

        # Update sum r_l x_l
        self.Xtr += reward * x

        # alpha_hat_(t+1)
        self.alpha = self.A_inv @ self.Xtr

    def V_inverse_norm(self, arm_features):
        return float(np.sqrt(arm_features @ self.A_inv @ arm_features))

    def get_weights(self):
        return {f"w_{i}": float(self.alpha[i]) for i in range(self.feature_dim)}


# ═══════════════════════════════════════════════════════════════════
# GapIndexSwapManager (CASE Algorithm 1)
# ═══════════════════════════════════════════════════════════════════
class GapIndexSwapManager:
    """
    Full CASE algorithm (Algorithm 1) from the paper.

    Order per round:
      Step 8-13:  Swap worst(U_t-1) vs best(N_t-1)
      Step 14:    Sample M_t from (U_t ∪ N_t-1)^c
      Step 15:    Reconstruct N_t = top_m'(M_t ∪ N_t-1; ρ̂)
      Step 16-18: Compute ambiguous arms for convergence
      Step 20:    CASE selection from U_t ∪ N_t (N_t now has best M_t arms absorbed)
      Step 21-23: Pull arm, reward, update α
    """

    def __init__(self, rng, sigma = 0.5, delta = 0.05, lambda_reg = 1.0):
        self.iteration = 0
        self.no_swap_count = 0
        self.rng = rng
        self.sigma = sigma
        self.delta = delta
        self.lambda_reg = lambda_reg
        

    def compute_C_t_delta(self, data_manager, bandit, beta_type = "Heuristic"):
        if self.iteration <= 0:
            raise ValueError(f"CASE iteration {self.iteration} must be > 0")
        if beta_type == "Heuristic":
            return np.sqrt(2.0 * np.log((np.log(self.iteration) + 1.0) / self.delta))
        
        N = 1
        #N = n.pulls
        L = max(np.linalg.norm(arm.avg_features) for arm in data_manager.all_arms.values())
        S = float(np.linalg.norm(bandit.alpha, 2))
        # 2 * log(1 / delta)
        term_1 = 2 * np.log(1 / self.delta)

        # N * log(1 + ((t + 1) * L^2) / (lambda^2 * N))
        numerator = (self.iteration + 1) * L**2
        denominator = self.lambda_reg**2 * N
        term_2 = N * np.log(1 + numerator / denominator)

        # Square root of Term 1 + Term 2
        confidence_term = np.sqrt(term_1 + term_2)
        regularization_term = (np.sqrt(self.lambda_reg) / self.sigma) * S

        # C_{t,delta}
        C_t_delta = (confidence_term + regularization_term)
        return C_t_delta

    def compute_gap(self, data_manager, bandit):
        """Gap = best(N_t-1).score - worst(U_t-1).score (paper step 9-10)."""
        training_arms = data_manager.get_training_arms_list()
        challenger_arms = data_manager.get_challenger_arms_list()
        if not training_arms or not challenger_arms:
            return 0.0, 0.0, None, None
        worst_training = min(training_arms, key=lambda a: a.score)
        best_challenger = max(challenger_arms, key=lambda a: a.score)    
        reward_gap = (best_challenger.score - worst_training.score)
        return reward_gap, worst_training, best_challenger

    def compute_B(self, data_manager, worst_training, best_challenger, bandit, beta_type):
        # C_{t,delta}
        C_t_delta = self.compute_C_t_delta(data_manager, bandit, beta_type)

        # sqrt(x_i^T V_t^{-1} x_i)
        b_ut = bandit.V_inverse_norm(worst_training.avg_features)

        # sqrt(x_j^T V_t^{-1} x_j)
        b_ch = bandit.V_inverse_norm(best_challenger.avg_features)

        # ||x_i||_{Sigma_hat} + ||x_j||_{Sigma_hat}
        uncertainty = self.sigma * (b_ut + b_ch)

        # W_t(i,j)
        W = C_t_delta * uncertainty
        reward_gap = (best_challenger.score - worst_training.score)
        #print(f"Reward gap : {reward_gap}, W : {W}")
        return reward_gap + W

    def sample_exploration_set(self, data_manager):
        """
        CASE step 14:
        M_t ~ (U_t ∪ N_{t-1}')^c

        At this point:
        training_arms    = U_t
        challenger_arms  = modified N_{t-1}
        """
        rng = self.rng
        if rng is None:
            rng = np.random.RandomState()

        # U_t
        training_ids = set(data_manager.training_arms)
        # Modified N_{t-1}
        challenger_ids = set(data_manager.challenger_arms)
        # Arms that cannot be sampled for M_t
        excluded_ids = training_ids | challenger_ids

        # Complement:
        # S \ (U_t ∪ N_{t-1}')
        candidate_ids = (set(data_manager.all_arms.keys())- excluded_ids)
        if not candidate_ids:
            return set()
        candidate_list = list(candidate_ids)

        # m' exploration arms
        m_prime = data_manager.num_challenger_arms
        m = min(m_prime, len(candidate_list))
        chosen = rng.choice(candidate_list, size=m, replace=False)

        return set(chosen)


    def execute_swap(self, data_manager, bandit, beta_type):
        """Paper steps 8-13: Swap worst(U_t-1) vs best(N_t-1).
           Paper step 14 : Sample M_t from (U_t ∪ N_t-1')^c.
           Paper Step 15: Reconstruct N_t = top_m'(M_t ∪ N_t-1')     
        """

        is_swap = False
        data_manager.update_arm_scores(bandit.alpha)
        reward_gap, worst_training, best_challenger = self.compute_gap(data_manager, bandit)
        B = self.compute_B(data_manager, worst_training, best_challenger, bandit, beta_type)
        if worst_training is None or best_challenger is None:
            return False, None, None, 0.0

        print(f"  Reward Gap: {reward_gap:.4f} ", flush=True)
        print(f"  Worst U_{self.iteration-1}: {worst_training.id} (score={worst_training.score:.4f})", flush=True)
        print(f"  Best  N_{self.iteration-1}: {best_challenger.id} (score={best_challenger.score:.4f})", flush=True)

        if reward_gap >= 0:
            self.no_swap_count = 0
            data_manager.training_arms.remove(worst_training.id)
            data_manager.challenger_arms.add(worst_training.id)
            data_manager.challenger_arms.remove(best_challenger.id)
            data_manager.training_arms.add(best_challenger.id)
            is_swap = True
            print(f"  >> SWAPPED: {worst_training.id} -> N_(t-1)', {best_challenger.id} -> U_t", flush=True)
        else:
            self.no_swap_count += 1
            print(f"  >> No swap (streak: {self.no_swap_count})", flush=True)


        M_t = self.sample_exploration_set(data_manager)
        data_manager.exploration_pool = M_t
        print(f"\n[Step 14] M_{self.iteration} sampled: {M_t} ({len(M_t)} arms from exploration pool)", flush=True)
        # --------------------------------------------------
        # N_t = top_m'(M_t ∪ N_(t-1)'; rho_hat_t)
        # --------------------------------------------------
        print(f"\n[Step 15] Reconstruct N_{self.iteration}...", flush=True)
        self.reconstruct_nt(data_manager)
        print(f"    Sets: U_{self.iteration}={len(data_manager.training_arms)}, N_{self.iteration}={len(data_manager.challenger_arms)}", flush=True)
        
        if is_swap:
            return True, worst_training.id, best_challenger.id, reward_gap
        return False, None, None, reward_gap
    
    def reconstruct_nt(self, data_manager):
        """
        N_t = top_m'(M_t ∪ N_(t-1)'; rho_hat_t)
        At this point:
            data_manager.training_arms   = U_t
            data_manager.challenger_arms = modified N_(t-1)
        """

        # N_(t-1)' ∪ M_t
        pool_ids = (set(data_manager.challenger_arms) | set(data_manager.exploration_pool))

        if not pool_ids:
            data_manager.challenger_arms = set()
            return

        # Rank all candidates by current estimated reward:
        # score(a) = rho_hat_t(a)
        pool_arms = [(aid, data_manager.all_arms[aid].score) for aid in pool_ids]
        pool_arms.sort(key=lambda x: x[1], reverse=True)

        m_prime = data_manager.num_challenger_arms
        new_nt = {aid for aid, score in pool_arms[:m_prime]}

        # For logging
        old_nt = set(data_manager.challenger_arms)
        mt_ids = set(data_manager.exploration_pool)
        promoted = list(new_nt & mt_ids)
        demoted = list(old_nt - new_nt)

        # Update challenger set
        data_manager.challenger_arms = new_nt

        # Logging
        if promoted:
            print(f"  >> N_t reconstructed: promoted {promoted} from M_t -> N_t", flush=True)
        if demoted:
            print(f"  >> N_t reconstructed: demoted {demoted} from N_(t-1)' -> rest", flush=True)
        if not promoted and not demoted:
            print("  >> N_t reconstructed: no changes", flush=True)

    def find_most_ambiguous_arms(self, data_manager, bandit, beta_type):
        """
        Step 16-18: Compute ambiguous arms for convergence   
        CASE:

            b_(t+1) = argmax_{b in U_t} max_{a in N_t} B_t(a,b)

            s_(t+1) = argmax_{s in N_t} B_t(s,b_(t+1))

        Returns:
            bt1              = most ambiguous arm in U_t
            st1              = most ambiguous challenger in N_t
            max_B            = B_t(st1, bt1)
        """

        training_arms = data_manager.get_training_arms_list()
        challenger_arms = data_manager.get_challenger_arms_list()
        if not training_arms or not challenger_arms:
            print("No training or challenger arm")
            return None, None, 0.0

        # --------------------------------------------------
        # Step 1:
        # For every b in U_t, find:
        # max_{a in N_t} B_t(a,b)
        # --------------------------------------------------

        best_b = None
        best_s = None
        best_B = -float("inf")

        for b in training_arms:
            for s in challenger_arms:
                B = self.compute_B(data_manager, s, b, bandit, beta_type)

                if B > best_B:
                    best_B = B
                    best_b = b
                    best_s = s

        return best_b, best_s, best_B

    def greedy_selection_rule(self, data_manager, bandit, bt1, st1):
        """
        Paper step 20: selection_rule(U_t, N_t).
        CASE greedy selection rule:
        a* = argmin_{a in U_t ∪ N_t}
             ||x_bt1 - x_st1||_
             (V_t^{-1} + x_a x_a^T)^(-1)

        where:
        ||z||_M = sqrt(z^T M z)
        """

        x_b = bt1.avg_features
        x_s = st1.avg_features

        # Difference between the two ambiguous arms
        x_diff = x_b - x_s

        # Candidate arms = U_t ∪ N_t
        candidate_ids = (set(data_manager.training_arms) | set(data_manager.challenger_arms))

        best_arm = None
        best_value = float("inf")

        for arm_id in candidate_ids:
            arm = data_manager.all_arms[arm_id]
            x_a = arm.avg_features

            # (V_t + x_a x_a^T)^(-1)
            updated_matrix = iterative_inversion(bandit.A_inv, x_a)
            # ||x_b - x_s||_M
            value = np.sqrt(x_diff @ updated_matrix @ x_diff)
            if value < best_value:
                best_value = value
                best_arm = arm

        return best_arm, best_value


# ═══════════════════════════════════════════════════════════════════
# Qwen Prompt Formatting
# ═══════════════════════════════════════════════════════════════════
def format_qwen_prompt(user_content, tokenizer):
    """
    Use Qwen2.5's tokenizer chat template instead of manually reproducing
    special tokens. This is important for generation because Qwen's template
    controls the assistant generation boundary.
    """
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def prepare_finetuning_data(samples):
    """Prepare training data for SFT from Sample objects."""
    finetuning_data = []
    for sample in samples:
        # Store the raw user content; Qwen's tokenizer applies the official chat template during SFT.
        finetuning_data.append({
            "prompt": sample.prompt,
            "output": sample.compressed_trace or "",
            "id": sample.id,
            "optimal_compression_ratio": sample.optimal_compression_ratio
        })
    return finetuning_data


def compute_reward(accuracy, avg_compression_ratio, beta=0.3):
    print(f"Accuracy : {accuracy}, Accuracy Base : {accuracy_base}, Diff : {accuracy - accuracy_base}, beta : {beta}, Accuracy Part : {(1-beta)*(accuracy - accuracy_base)}")
    print(f"Avg Compression Ratio : {avg_compression_ratio}, Diff : {1.0 - avg_compression_ratio}, beta : {beta}, Avg CR Part : {beta*(1.0 - avg_compression_ratio)}")
    return (1-beta)* (accuracy - accuracy_base) + beta * (1.0 - avg_compression_ratio)


# ═══════════════════════════════════════════════════════════════════
# Save Results
# ═══════════════════════════════════════════════════════════════════
def save_results(output_dir, data_manager, history, alpha, feature_columns):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    all_training = data_manager.get_all_training_samples()
    finetuning_data = prepare_finetuning_data(all_training)
    with open(output_path / "final_training_data.jsonl", 'w', encoding='utf-8') as f:
        for item in finetuning_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    arm_info = [data_manager.all_arms[aid].to_dict() for aid in data_manager.training_arms]
    with open(output_path / "final_training_arms.json", 'w') as f:
        json.dump(arm_info, f, indent=2)
    weights = {col: float(alpha[i]) for i, col in enumerate(feature_columns)}
    with open(output_path / "final_weights.json", 'w') as f:
        json.dump(weights, f, indent=2)
    with open(output_path / "bandit_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    total_samples = sum(len(data_manager.all_arms[aid].sample_ids) for aid in data_manager.training_arms)
    print(f"\nResults saved to {output_dir}/")
    print(f"  - final_training_data.jsonl ({total_samples} samples)")
    print(f"  - final_training_arms.json ({len(data_manager.training_arms)} arms)")
    print(f"  - final_weights.json")
    print(f"  - bandit_history.json")


# ═══════════════════════════════════════════════════════════════════
# Qwen SFT Trainer
# ═══════════════════════════════════════════════════════════════════
@dataclass
class SFTConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    max_seq_length: int = 2048
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1

    def __post_init__(self):
        if self.target_modules is None:
            # Qwen uses separate attention and MLP projection modules.
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


class QwenSFTTrainer:
    def __init__(self, config=None):
        self.config = config or SFTConfig()
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self._initial_lora_state = None
        self.latest_adapter_path = None

    def load_model(self):
        print(f"Loading {self.config.model_name} in bfloat16...", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            use_fast=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )

        self.model.config.use_cache = False

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        self.model.enable_input_require_grads()

        # Save a clean copy of the initial LoRA weights. Every CASE arm must
        # start from exactly the same base+LoRA state.
        self._initial_lora_state = {
            name: param.detach().cpu().clone()
            for name, param in self.model.named_parameters()
            if "lora_" in name
        }

        self.is_loaded = True

    def reset_adapter(self):
        """Restore the LoRA adapter to its initial zero/random initialization."""
        if self._initial_lora_state is None:
            raise RuntimeError("Initial LoRA state has not been saved.")

        missing = []
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    if name not in self._initial_lora_state:
                        missing.append(name)
                    else:
                        param.copy_(
                            self._initial_lora_state[name].to(
                                device=param.device, dtype=param.dtype
                            )
                        )

        if missing:
            raise RuntimeError(f"Missing initial LoRA parameters: {missing[:5]}")

        self.latest_adapter_path = None
        self.model.train()
        self.model.config.use_cache = False

    def _build_training_features(self, training_samples):
        """
        Build tokenized Qwen2.5 SFT examples.

        Crucially, labels for the user prompt are -100, so loss is computed
        only on the compressed reasoning/answer response.
        """
        features = []
        skipped_empty = 0

        for s in training_samples:
            user_content = (s.get("prompt") or "").strip()
            response = (s.get("output") or "").strip()

            if not user_content or not response:
                skipped_empty += 1
                continue

            prompt_text = format_qwen_prompt(user_content, self.tokenizer)

            prompt_ids = self.tokenizer(
                prompt_text,
                add_special_tokens=False,
            )["input_ids"]

            response_ids = self.tokenizer(
                response,
                add_special_tokens=False,
            )["input_ids"]

            # Explicit EOS guarantees a clean response termination target.
            eos_id = self.tokenizer.eos_token_id
            if eos_id is not None:
                response_ids = response_ids + [eos_id]

            input_ids = prompt_ids + response_ids
            labels = ([-100] * len(prompt_ids)) + response_ids.copy()

            if len(input_ids) > self.config.max_seq_length:
                input_ids = input_ids[:self.config.max_seq_length]
                labels = labels[:self.config.max_seq_length]

                # Never leave the last token as an ignored label if truncation
                # cut the response; the remaining response tokens still train.
                if all(x == -100 for x in labels):
                    skipped_empty += 1
                    continue

            attention_mask = [1] * len(input_ids)

            features.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

        print(
            f"Prepared {len(features)} training examples; "
            f"skipped {skipped_empty} empty/invalid examples.",
            flush=True,
        )

        if not features:
            raise RuntimeError(
                "No non-empty training examples remain. "
                "Check compressed_trace in dataset/bandit_data.jsonl."
            )

        return features

    def _collate(self, features):
        max_len = max(len(x["input_ids"]) for x in features)
        pad_id = self.tokenizer.pad_token_id

        input_ids = []
        attention_mask = []
        labels = []

        for x in features:
            n = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [pad_id] * n)
            attention_mask.append(x["attention_mask"] + [0] * n)
            labels.append(x["labels"] + [-100] * n)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def train(self, training_samples, output_dir, iteration=0, num_epochs=None):
        if not self.is_loaded:
            self.load_model()

        # Verify data before launching a potentially expensive GPU run.
        empty = [
            s.get("id", "?")
            for s in training_samples
            if not (s.get("output") or "").strip()
        ]
        if empty:
            print(
                f"WARNING: {len(empty)}/{len(training_samples)} training traces "
                f"are empty. First IDs: {empty[:10]}",
                flush=True,
            )

        features = self._build_training_features(training_samples)

        adapter_path = os.path.join(output_dir, f"adapter_iter_{iteration}")
        os.makedirs(adapter_path, exist_ok=True)

        epochs = num_epochs if num_epochs is not None else self.config.num_train_epochs

        args = TrainingArguments(
            output_dir=adapter_path,
            num_train_epochs=epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=10,
            save_strategy="no",
            bf16=True,
            optim="adamw_torch",
            report_to="none",
            remove_unused_columns=False,
            gradient_checkpointing=True,
        )

        print(
            f"Training iter {iteration} ({len(features)} samples, "
            f"{epochs} epochs)...",
            flush=True,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=Dataset.from_list(features),
            data_collator=self._collate,
        )

        trainer.train()
        self.model.save_pretrained(adapter_path)
        print(f"Saved to {adapter_path}", flush=True)
        self.latest_adapter_path = adapter_path
        return adapter_path

    @torch.no_grad()
    def infer(self, test_data, answer_extraction_fn, max_new_tokens=2048):
        """Generate with Qwen2.5 using its official chat template."""
        self.model.eval()
        prompts = []

        for example in test_data:
            user_content = ""
            for mess in example["messages"]:
                if mess["role"] == "user":
                    user_content = mess["content"]

            prompt = format_qwen_prompt(user_content, self.tokenizer)
            example["prompt"] = prompt
            prompts.append(prompt)

        print(f"\nRunning inference on {len(prompts)} samples...")

        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
            add_special_tokens=False,
        ).to(self.model.device)

        # Debug the generation boundary once per batch.
        if prompts:
            print(
                f"  Prompt tokens: {inputs['input_ids'].shape[1]}, "
                f"EOS id: {self.tokenizer.eos_token_id}, "
                f"pad id: {self.tokenizer.pad_token_id}",
                flush=True,
            )

        torch.cuda.synchronize()
        start_time = time()

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        torch.cuda.synchronize()
        total_time = time() - start_time

        # With left padding, the generated tokens begin after the padded
        # input width, so this slice is correct for the whole batch.
        prompt_len = inputs["input_ids"].shape[1]
        model_outputs = self.tokenizer.batch_decode(
            outputs[:, prompt_len:],
            skip_special_tokens=True,
        )

        # Explicitly report empty generations.
        empty_count = sum(not out.strip() for out in model_outputs)
        if empty_count:
            print(
                f"  WARNING: {empty_count}/{len(model_outputs)} generations "
                f"are empty.",
                flush=True,
            )
            for i, out in enumerate(model_outputs):
                if not out.strip():
                    print(
                        f"    Empty generation #{i}: "
                        f"last input token id="
                        f"{inputs['input_ids'][i, -1].item()}",
                        flush=True,
                    )

        cot_lengths = []
        for model_completion in model_outputs:
            cot = model_completion.split("\n\nThe final answer is:")[0]
            cot_length = self.tokenizer(
                cot,
                add_special_tokens=False,
            )["input_ids"].__len__()
            cot_lengths.append(cot_length)

        predictions = [
            extract_math_answer(
                item["messages"][-2]["content"]
                if isinstance(item.get("messages"), list)
                and len(item["messages"]) >= 2
                else item.get("prompt", ""),
                output,
                task="cot",
            )
            for item, output in tqdm(
                zip(test_data, model_outputs),
                desc="Extracting answers",
                total=len(model_outputs),
            )
        ]

        print("\nEvaluating predictions...")

        results = []
        pbar = tqdm(
            zip(test_data, model_outputs, predictions, cot_lengths),
            total=len(model_outputs),
            desc="Evaluating outputs",
        )

        for example, output, pred, cot_length in pbar:
            item = deepcopy(example)
            item.update({
                "model_output": output,
                "prediction": pred,
                "cot_length": cot_length,
            })

            if len(pred) == 0:
                item["accuracy"] = False
            else:
                item["accuracy"] = eval_math(item)

            results.append(item)

            elapsed = time() - start_time
            avg_time = elapsed / len(results)
            pbar.set_postfix({"avg_s/sample": f"{avg_time:.3f}"})

        print("\nCalculating accuracy...")

        acc = sum(item["accuracy"] for item in results) / len(results)
        avg_cot_length = sum(item["cot_length"] for item in results) / len(results)

        print(f"Accuracy = {acc*100:.5f}")
        print(f"Avg CoT Length = {avg_cot_length:.5f}")
        print(f"Sample latency = {total_time/len(test_data):.5f}")

        return {
            "results": results,
            "accuracy": acc,
            "avg_cot_length": avg_cot_length,
            "sample_latency": total_time / len(test_data),
            "total_time": total_time,
        }

    @torch.no_grad()
    def evaluate(self, val_samples, max_new_tokens=2048, batch_size=16):
        """Evaluate Qwen using Hugging Face generation."""
        self.model.eval()
        total = len(val_samples)
        correct_count = 0
        actual_crs = []
        all_results = []

        num_batches = (total + batch_size - 1) // batch_size
        print(
            f"\n>>> EVALUATING {total} SAMPLES in {num_batches} batches "
            f"(bs={batch_size}) [Qwen HF] <<<",
            flush=True,
        )

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, total)
            batch_samples = val_samples[start:end]

            results = self.infer(
                batch_samples,
                "extract_math_answer",
                max_new_tokens,
            )



            for j, item in enumerate(results["results"]):
                gen_len = item["cot_length"]
                orig_len = batch_samples[j].get("cot_length", gen_len)

                actual_cr = gen_len / orig_len if orig_len > 0 else 1.0
                actual_crs.append(actual_cr)

                accuracy = item["accuracy"]

                is_correct = accuracy                  

                if is_correct:
                    correct_count += 1

                print(
                    f"  [{start+j+1}/{total}] CR={actual_cr:.2f}, "
                    f"correct={is_correct}",
                    flush=True,
                )

            all_results.extend(results["results"])

        pred_file = os.path.join(OUTPUT_DIR, "bandit_predictions.jsonl")
        with open(pred_file, "w", encoding="utf-8") as f:
            for p in all_results:
                f.write(json.dumps(p, default=str) + "\n")

        accuracy = correct_count / total if total > 0 else 0.0
        avg_cr = sum(actual_crs) / len(actual_crs) if actual_crs else 1.0

        print(
            f"\n>>> Acc={accuracy:.2%} ({correct_count}/{total}), "
            f"Avg_CR={avg_cr:.4f} <<<\n",
            flush=True,
        )

        return {"accuracy": accuracy, "avg_cr": avg_cr}

    @torch.no_grad()
    def sanity_check_generation(self, val_samples, max_new_tokens=128):
        """
        Verify that the base/reset Qwen model can generate non-empty text
        before CASE starts. This catches prompt/tokenizer/model issues early.
        """
        if not val_samples:
            return

        example = val_samples[0]
        user_content = next(
            (m["content"] for m in example["messages"] if m["role"] == "user"),
            "",
        )
        prompt = format_qwen_prompt(user_content, self.tokenizer)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.model.device)

        print("\n=== QWEN GENERATION SANITY CHECK ===", flush=True)
        print(
            f"Input tokens={inputs['input_ids'].shape[1]}, "
            f"EOS={self.tokenizer.eos_token_id}, "
            f"PAD={self.tokenizer.pad_token_id}",
            flush=True,
        )

        self.model.eval()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        generated = self.tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        print(f"Generated tokens={outputs.shape[1] - inputs['input_ids'].shape[1]}")
        print(f"Generated text: {repr(generated[:500])}", flush=True)

        if not generated.strip():
            raise RuntimeError(
                "Qwen generated an empty response BEFORE CASE/SFT. "
                "This indicates a model/tokenizer/prompt-generation issue."
            )



# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════
CONFIG = {
    "model_name": "Qwen/Qwen2.5-3B-Instruct",
    "arm_size": 50,
    "num_arms" : 50,
    "num_training_arms": 10,
    "num_challenger_arms": 10,
    "num_validation": 50,
    "max_iterations": 100,
    "lambda_reg": 1.0,
    "beta": 0.5,
    "random_seed": 42,
    "skip_sft": "--skip_sft" in sys.argv,
    "val_eval_size": 50,
    "gen_max_tokens": 1024,
    "eval_batch_size": 8,
    "sanity_max_tokens": 128,
    "min_valid_cr": 0.10,
    "epsilon" : 0.01,
    "sigma": 0.5,
    "delta": 0.1,
    "beta_type": "Heuristic" #Frequentist
}


# ═══════════════════════════════════════════════════════════════════
# Main Loop
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print(f"{CONFIG['num_training_arms']} U_t, {CONFIG['num_challenger_arms']} N_t")
    print(f"Reward: (1-beta)*(accuracy-base_accuracy) + beta*(1-avg_CR), beta={CONFIG['beta']}")
    print(f"Mode: {'DRY RUN (skip_sft)' if CONFIG['skip_sft'] else 'FULL CASE ALGORITHM with SFT'}")

    print("=" * 60)
    print("CASE BANDIT: Algorithm 1 — Qwen2.5-3B-Instruct on DeepMath")
    print("=" * 60)

    # Initialize data manager
    data_manager = SetBanditDataManager(
        CSV_PATH, BANDIT_PATH, ORIG_PATH,
        arm_size=CONFIG["arm_size"],
        num_training_arms=CONFIG["num_training_arms"],
        num_challenger_arms=CONFIG["num_challenger_arms"],
        num_arms = CONFIG["num_arms"],
        num_validation=CONFIG["num_validation"]
    )
    feature_dim = len(FEATURE_COLUMNS)
    bandit = LinearBandit(feature_dim, random_seed=CONFIG["random_seed"])
    data_manager.form_arms_and_initialize(bandit.alpha, CONFIG["random_seed"])
    print("U_0, N_0 initialized ..........")
    rng = np.random.RandomState(CONFIG["random_seed"])
    swap_manager = GapIndexSwapManager(rng =rng, sigma = CONFIG["sigma"], delta = CONFIG["delta"], lambda_reg = CONFIG["lambda_reg"])
    

    # Print initial arm scores
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

    # Load model (if not dry run)
    if not CONFIG["skip_sft"]:
        print("\nLoading model...", flush=True)
        trainer = QwenSFTTrainer(SFTConfig(model_name=CONFIG["model_name"]))
        trainer.load_model()

    # Load the validation set from the reserved bandit pool before training.
    # This 50-sample set is used for the CASE reward loop and stays isolated
    # from the training arms.
    history = []
    bandit_records = read_data(BANDIT_PATH)
    bandit_by_id = {item["id"]: item for item in bandit_records}
    validation_ids = sorted(data_manager.validation_ids)
    val_data = [bandit_by_id[sid] for sid in validation_ids[:CONFIG["val_eval_size"]] if sid in bandit_by_id]
    val_samples = val_data

    # IMPORTANT: verify the untouched Qwen2.5 model generates text before
    # running any SFT. If this fails, CASE should not start.
    if not CONFIG["skip_sft"]:
        trainer.sanity_check_generation(
            val_samples[:1],
            max_new_tokens=CONFIG["sanity_max_tokens"],
        )

    # ── Main Bandit Loop ──
    swap_manager.iteration = 1
    while swap_manager.iteration <= CONFIG["max_iterations"]:
        
        print(f"\n{'#' * 60}")
        print(f"# ITERATION {swap_manager.iteration}/{CONFIG['max_iterations']}")
        print(f"{'#' * 60}", flush=True)

        # === Step 8-13: SWAP worst(U_t-1) vs best(N_t-1)
        # === Paper step 14 : Sample M_t from (U_t ∪ N_t-1')^c.
        # === Paper Step 15: Reconstruct N_t = top_m'(M_t ∪ N_t-1')
        print(f"\n[Step 8-13] Swap check (U_{swap_manager.iteration-1} vs N_{swap_manager.iteration-1})...", flush=True)
        swapped, removed, added, gap = swap_manager.execute_swap(data_manager, bandit, CONFIG["beta_type"])

        # === Step 16-18: Compute ambiguous arms for convergence
        bt1, st1, B = swap_manager.find_most_ambiguous_arms(data_manager, bandit, CONFIG["beta_type"])
        print(f"\n[Step 16-18] Most ambiguous arms:", flush=True)
        print(f"    b_({swap_manager.iteration+1}): {bt1.id}", flush=True)
        print(f"    s_({swap_manager.iteration+1}): {st1.id}", flush=True)
        print(f"    B_{swap_manager.iteration}(s,b): {B:.6f}", flush=True)

        # === Convergence check ===
        if B <= CONFIG["epsilon"]:
            print(f"\nConverged at iteration {swap_manager.iteration}: B={B:.6f} <= epsilon={CONFIG['epsilon']}", flush=True)
            break

        # === Step 20: CASE selection from U_t ∪ N_t ===
        print(f"\n[Step 20] CASE arm selection (U_{swap_manager.iteration} ∪ N_{swap_manager.iteration})...", flush=True)
        pulled_arm, selection_value = (swap_manager.greedy_selection_rule(data_manager, bandit, bt1,st1))   
        arm_samples = data_manager.get_arm_samples(pulled_arm)
        training_data = prepare_finetuning_data(arm_samples)
        loc = "U_t" if pulled_arm.id in data_manager.training_arms else "N_t"
        print(f"    -> {pulled_arm.id} [{loc}] ({len(arm_samples)} traces, pull #{pulled_arm.n_pulls + 1})", flush=True)

        # === Step 21-23: Pull arm (SFT + eval), get reward, update α ===
        if not CONFIG["skip_sft"]:
            print(f"\n[Step 21] Resetting LoRA adapter...", flush=True)
            trainer.reset_adapter()
            print(f"\n[Step 21] SFT on {len(training_data)} traces...", flush=True)
            trainer.train(training_data, OUTPUT_DIR, swap_manager.iteration)
            print(f"\n[Step 21] Evaluating {len(val_samples)} val samples...", flush=True)
            result = trainer.evaluate(val_samples, CONFIG["gen_max_tokens"], CONFIG["eval_batch_size"])
            accuracy, avg_cr = result["accuracy"], result["avg_cr"]
        else:
            accuracy = np.mean([1.0 if s.accuracy else 0.0 for s in arm_samples])
            avg_cr = np.mean([s.optimal_compression_ratio for s in arm_samples])
        
        reward = compute_reward(accuracy, avg_cr, CONFIG["beta"])
        print(f"Reward calculated : {reward}")
        pulled_arm.record_pull(reward)

        # === Step 22-23: Update reward + ridge regression ===
        bandit.update_weights(pulled_arm.avg_features, reward)
        data_manager.update_arm_scores(bandit.alpha)

        print(f"\n[Step 22-23] {pulled_arm.id}: acc={accuracy:.4f}, CR={avg_cr:.4f}, REWARD={reward:.4f} (pulls: {pulled_arm.n_pulls})", flush=True)
        print(f"    Weights: [{', '.join(f'{a:.4f}' for a in bandit.alpha)}]", flush=True)

        # === Calculate Tracking Metrics ===
        predicted_reward = np.dot(pulled_arm.avg_features, bandit.alpha)
        prediction_error = abs(predicted_reward - reward)
        ut_scores = [data_manager.all_arms[aid].score for aid in data_manager.training_arms]
        ut_min, ut_max, ut_median = np.min(ut_scores), np.max(ut_scores), np.median(ut_scores)

        history.append({
            "iter": swap_manager.iteration, "arm": pulled_arm.id, "arm_set": loc,
            "reward": reward, "acc": accuracy, "cr": avg_cr,
            "weights": bandit.alpha.tolist(),
            "swapped": swapped, "gap": gap,
            "M_t": [str(mid) for mid in data_manager.exploration_pool],
            "prediction_error": prediction_error,
            "ut_min": ut_min, "ut_max": ut_max, "ut_median": ut_median
        })

        swap_manager.iteration += 1


    # ── Save results ──
    print("\n" + "=" * 60)
    save_results(OUTPUT_DIR, data_manager, history, bandit.alpha, FEATURE_COLUMNS)

    # ── Final SFT on selected U_t arms ──
    if not CONFIG["skip_sft"]:
        final_training_arms = [data_manager.all_arms[aid] for aid in data_manager.training_arms]
        final_training_samples = []
        for arm in final_training_arms:
            final_training_samples.extend(data_manager.get_arm_samples(arm))

        print(f"Total samples for final fine-tuning: {len(final_training_samples)}")

        trainer.reset_adapter()
        final_data = prepare_finetuning_data(final_training_samples)
        trainer.train(final_data, output_dir=OUTPUT_DIR, iteration="FINAL", num_epochs=8)



        # ── Final test evaluation on the original test set ──
        # The 50-sample validation pool is reserved from bandit_data.jsonl for
        # the CASE reward loop. The final reported accuracy should still be
        # computed on the untouched original test set.
        raw_data = read_data(VAL_DATA_PATH)

        test_samples = []
        for d in raw_data:
            sample = d.copy()
            user_content = ""
            for mess in d['messages']:
                if mess['role'] == 'user':
                    user_content = mess['content']
            sample['prompt'] = format_qwen_prompt(user_content, trainer.tokenizer)

            # Calculate CR relative to original model output token length
            if 'model_output' in d:
                tokens = trainer.tokenizer.encode(d['model_output'], add_special_tokens=False)
                sample['cot_length'] = len(tokens)
            else:
                sample['cot_length'] = 1

            test_samples.append(sample)

        print(f"Prepared {len(test_samples)} samples for final test evaluation.")
        result = trainer.evaluate(test_samples, max_new_tokens=2048, batch_size=CONFIG["eval_batch_size"])
        print(f"\nFinal Test Accuracy: {result['accuracy']:.2%}")
        print(f"Average Compression Ratio: {result['avg_cr']:.4f}")