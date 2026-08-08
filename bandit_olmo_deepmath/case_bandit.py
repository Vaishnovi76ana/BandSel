"""
CASE Bandit Algorithm for OLMo-3-7B-Instruct on MATH-500.

Co-Active Selection and Exploration (CASE) linear contextual bandit
for selecting optimal training data subsets.

Adapted from Sagnibha's case_bandit_bfloat16.py for Qwen/Phi4,
with OLMo-specific prompt formatting and LoRA target modules.

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

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.model_executor.models.registry import ModelRegistry
    # Patch Olmo2ForCausalLM to support LoRA before registering it
    import vllm.model_executor.models.olmo2 as olmo2
    olmo2.Olmo2ForCausalLM.supports_lora = True
    olmo2.Olmo2ForCausalLM.packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }
    olmo2.Olmo2ForCausalLM.supported_lora_modules = [
        "qkv_proj", "o_proj", "gate_up_proj", "down_proj", "embed_tokens",
        "lm_head"
    ]
    olmo2.Olmo2ForCausalLM.embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings"
    }
    olmo2.Olmo2ForCausalLM.embedding_padding_modules = ["lm_head"]

    # Patch Olmo2Model and Olmo2ForCausalLM init methods to support LoRA extra vocab size
    def patched_olmo2_model_init(self, *, vllm_config, prefix=""):
        import torch.nn as nn
        from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
        from vllm.model_executor.models.utils import make_layers, make_empty_intermediate_tensors_factory
        from vllm.model_executor.layers.layernorm import RMSNorm
        
        nn.Module.__init__(self)
        self.config = vllm_config.model_config.hf_config
        
        lora_config = vllm_config.lora_config
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = self.config.vocab_size + lora_vocab
        
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            self.config.hidden_size,
            org_num_embeddings=self.config.vocab_size,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            self.config.num_hidden_layers,
            lambda prefix: olmo2.Olmo2DecoderLayer(vllm_config=vllm_config,
                                             prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(["hidden_states"],
                                                    self.config.hidden_size))

    olmo2.Olmo2Model.__init__ = patched_olmo2_model_init

    def patched_olmo2_for_causal_lm_init(self, *, vllm_config, prefix=""):
        import torch.nn as nn
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.model_executor.layers.sampler import Sampler
        from vllm.model_executor.models.utils import maybe_prefix
        
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.config = config
        self.lora_config = vllm_config.lora_config
        self.model = olmo2.Olmo2Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))
        
        self.unpadded_vocab_size = config.vocab_size
        if self.lora_config:
            self.unpadded_vocab_size += self.lora_config.lora_extra_vocab_size
            
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
            
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size, config.vocab_size)
        self.sampler = Sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    olmo2.Olmo2ForCausalLM.__init__ = patched_olmo2_for_causal_lm_init

    if "Olmo3ForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("Olmo3ForCausalLM", "vllm.model_executor.models.olmo2:Olmo2ForCausalLM")
    
    # Rebind Olmo2Config to accept Olmo3Config to bypass assert isinstance(config, Olmo2Config)
    try:
        from transformers.models.olmo3.configuration_olmo3 import Olmo3Config
        olmo2.Olmo2Config = (olmo2.Olmo2Config, Olmo3Config)
    except ImportError:
        pass
    if "--skip_vllm" in sys.argv:
        HAS_VLLM = False
    else:
        HAS_VLLM = True
except ImportError:
    HAS_VLLM = False



from eval_utils import extract_math_answer, eval_math


# ═══════════════════════════════════════════════════════════════════
# File paths
# ═══════════════════════════════════════════════════════════════════
CSV_PATH = "dataset/reasoning_scores.csv"
JSONL_PATH = "dataset/bandit_data.jsonl"
VAL_DATA_PATH = "dataset/predictions.jsonl"
OUTPUT_DIR = "bandit_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Ready: CSV={os.path.exists(CSV_PATH)}, JSONL={os.path.exists(JSONL_PATH)}, VAL={os.path.exists(VAL_DATA_PATH)}")


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


FEATURE_COLUMNS = [
    'signal_ratio',
    'distinct_1_ratio', 'distinct_2_ratio', 'distinct_3_ratio', 'distinct_4_ratio',
    'top_k_mass_ratio', 'windowed_jaccard',
    'longest_repeat_ratio', 'llm_judge_score'
]


# ═══════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════
@dataclass
class Sample:
    id: str
    features: np.ndarray
    accuracy: bool
    original_cot_length: int
    target_compression_ratio: float
    prompt: str = ""
    answer: str = ""
    compressed_trace: str = ""

    def to_dict(self):
        return {"id": self.id, "accuracy": self.accuracy,
                "original_cot_length": self.original_cot_length,
                "target_compression_ratio": self.target_compression_ratio,
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
    def __init__(self, csv_path, jsonl_path, arm_size=25, num_training_arms=10,
                 num_challenger_arms=10, num_total_traces=500, num_validation=100,
                 num_exploration=5):
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path
        self.arm_size = arm_size
        self.num_training_arms = num_training_arms
        self.num_challenger_arms = num_challenger_arms
        self.num_total_traces = num_total_traces
        self.num_validation = num_validation
        self.num_exploration = num_exploration  # |M_t| per round
        self.all_samples = {}
        self.all_arms = {}
        self.training_arms = set()     # U_t
        self.challenger_arms = set()   # N_t
        self.rest_arms = set()         # (U_t ∪ N_t)^c — exploration pool
        self.validation_ids = set()
        self._load_data()

    def _load_data(self):
        df_features = pd.read_csv(self.csv_path)
        trace_data = {}
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                trace_data[data['id']] = data
        for _, row in df_features.iterrows():
            sample_id = str(row['id'])
            if sample_id not in trace_data:
                continue
            trace = trace_data[sample_id]
            features = np.array([row[col] for col in FEATURE_COLUMNS], dtype=np.float32)
            features = np.nan_to_num(features, nan=0.5)
            target_cr = trace.get('target_compression_ratio',
                                  trace.get('optimal_compression_ratio', 0.7))
            raw_prompt = trace.get('prompt', '')
            prefix = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
            if raw_prompt.startswith(prefix):
                user_content = raw_prompt[len(prefix):]
            else:
                user_content = raw_prompt.split("Please reason step by step, and put your final answer within \\boxed{}.\n")[-1]
            user_content = user_content.split("<|im_end|>")[0]
            user_content = user_content.split("<|eot_id|>")[0]
            user_content = user_content.strip()
            
            formatted_prompt = format_olmo_prompt(user_content, compression_ratio=None)

            self.all_samples[sample_id] = Sample(
                id=sample_id, features=features,
                accuracy=trace.get('accuracy', trace.get('compressed_accuracy', False)),
                original_cot_length=trace.get('original_cot_length', 0),
                target_compression_ratio=target_cr,
                prompt=formatted_prompt,
                answer=trace.get('answer', ''),
                compressed_trace=trace.get('compressed_trace', '')
            )

        print(f"Loaded {len(self.all_samples)} samples with {len(FEATURE_COLUMNS)} features each")

    def form_arms_and_initialize(self, alpha, random_seed=42):
        rng = np.random.RandomState(random_seed)
        all_ids = list(self.all_samples.keys())
        rng.shuffle(all_ids)
        self.validation_ids = set(all_ids[:self.num_validation])
        remaining = [sid for sid in all_ids if sid not in self.validation_ids]
        arm_ids_pool = remaining[:self.num_total_traces]
        num_arms = len(arm_ids_pool) // self.arm_size
        for i in range(num_arms):
            start = i * self.arm_size
            chunk = arm_ids_pool[start:start + self.arm_size]
            arm_id = f"arm_{i}"
            member_features = np.vstack([self.all_samples[sid].features for sid in chunk])
            self.all_arms[arm_id] = Arm(id=arm_id, sample_ids=chunk,
                                        avg_features=member_features.mean(axis=0))
        for arm in self.all_arms.values():
            arm.update_score(alpha)
        sorted_arms = sorted(self.all_arms.values(), key=lambda a: a.score, reverse=True)
        for i, arm in enumerate(sorted_arms):
            if i < self.num_training_arms:
                self.training_arms.add(arm.id)
            elif i < self.num_training_arms + self.num_challenger_arms:
                self.challenger_arms.add(arm.id)
            else:
                self.rest_arms.add(arm.id)
        print(f"\nFormed {len(self.all_arms)} arms of {self.arm_size} traces each")
        print(f"  - Training arms (U_t): {len(self.training_arms)}")
        print(f"  - Challenger arms (N_t): {len(self.challenger_arms)}")
        print(f"  - Rest pool (for M_t): {len(self.rest_arms)}")
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

    def get_rest_arms_list(self):
        return [self.all_arms[aid] for aid in self.rest_arms]

    def sample_exploration_set(self, rng=None):
        """Sample M_t from rest_arms — paper: (U_t ∪ N_t)^c."""
        if not self.rest_arms:
            return set()
        rest_list = list(self.rest_arms)
        m = min(self.num_exploration, len(rest_list))
        if rng is None:
            rng = np.random.RandomState()
        chosen = rng.choice(rest_list, size=m, replace=False)
        return set(chosen)


# ═══════════════════════════════════════════════════════════════════
# LinearBandit (Ridge Regression)
# ═══════════════════════════════════════════════════════════════════
class LinearBandit:
    def __init__(self, feature_dim=9, lambda_reg=1.0, learning_rate=1.0, random_seed=42):
        self.feature_dim = feature_dim
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate
        rng = np.random.RandomState(random_seed)
        self.alpha = rng.randn(feature_dim) * 0.1
        self.X_history = []
        self.r_history = []
        self.A_inv = np.eye(feature_dim) / lambda_reg

    def predict_score(self, features):
        return float(np.dot(self.alpha, features))

    def update_weights(self, arm_features, reward):
        self.X_history.append(arm_features)
        self.r_history.append(reward)
        X_mat = np.vstack(self.X_history)
        r_vec = np.array(self.r_history)

        A = X_mat.T @ X_mat + self.lambda_reg * np.eye(self.feature_dim)
        Xtr = X_mat.T @ r_vec

        try:
            new_alpha = np.linalg.solve(A, Xtr)
            self.A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            new_alpha = np.linalg.lstsq(A, Xtr, rcond=None)[0]
            self.A_inv = np.linalg.pinv(A)

        self.alpha = (1 - self.learning_rate) * self.alpha + self.learning_rate * new_alpha

    def get_confidence_bound(self, arm_features):
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
      Step 8-13:  Swap worst(U_t) vs best(N_t) — NO M_t
      Step 14:    Sample M_t from (U_t ∪ N_t)^c
      Step 15:    Reconstruct N_t = top_m'(M_t ∪ old_N_t; ρ̂)
      Step 16-18: Compute ambiguous arms for convergence
      Step 20:    CASE selection from U_t ∪ N_t (N_t now has best M_t arms absorbed)
      Step 21-23: Pull arm, reward, update α
    """

    def __init__(self, confidence_scale=1.0):
        self.confidence_scale = confidence_scale
        self.iteration = 0
        self.no_swap_count = 0

    def compute_gap(self, data_manager):
        """Gap = best(N_t).score - worst(U_t).score (paper step 9-10)."""
        training_arms = data_manager.get_training_arms_list()
        challenger_arms = data_manager.get_challenger_arms_list()
        if not training_arms or not challenger_arms:
            return 0.0, None, None
        worst_training = min(training_arms, key=lambda a: a.score)
        best_challenger = max(challenger_arms, key=lambda a: a.score)
        gap = best_challenger.score - worst_training.score
        return gap, worst_training, best_challenger

    def compute_gap_index(self, worst_ut, best_challenger, bandit):
        b_ut = bandit.get_confidence_bound(worst_ut.avg_features) * self.confidence_scale
        b_ch = bandit.get_confidence_bound(best_challenger.avg_features) * self.confidence_scale
        return b_ut + b_ch

    def select_most_uncertain_arm(self, data_manager, bandit):
        """
        Paper step 20: selection_rule(U_t, N_t).
        Picks the arm that maximizes Information Gain (Greedy Variance Reduction)
        regarding the gap between Worst U_t and Best N_t.
        """
        data_manager.update_arm_scores(bandit.alpha)

        training = data_manager.get_training_arms_list()
        challengers = data_manager.get_challenger_arms_list()

        if not training or not challengers:
            return data_manager.get_random_training_arm()

        worst_ut = min(training, key=lambda a: a.score)
        best_nt = max(challengers, key=lambda a: a.score)

        d = worst_ut.avg_features - best_nt.avg_features
        Ainv = bandit.A_inv

        best_arm = None
        max_gain = float('-inf')

        candidate_arms = training + challengers
        for arm in candidate_arms:
            x = arm.avg_features
            Ainv_x = Ainv @ x
            gain = (np.dot(d, Ainv_x)**2) / (1.0 + np.dot(x, Ainv_x))

            if gain > max_gain:
                max_gain = gain
                best_arm = arm

        loc = 'U_t' if best_arm.id in data_manager.training_arms else 'N_t'
        print(f"  CASE select (Greedy Gain): {best_arm.id} [{loc}] "
              f"(gain={max_gain:.6f}, pulls={best_arm.n_pulls})", flush=True)
        return best_arm

    def execute_swap(self, data_manager, bandit):
        """Paper steps 8-13: Swap worst(U_t) vs best(N_t)."""
        self.iteration += 1
        data_manager.update_arm_scores(bandit.alpha)
        gap, worst_training, best_challenger = self.compute_gap(data_manager)
        if worst_training is None or best_challenger is None:
            return False, None, None, 0.0, float('inf')

        gap_index = self.compute_gap_index(worst_training, best_challenger, bandit)
        print(f"  Gap: {gap:.4f} | Gap-index: {gap_index:.4f}", flush=True)
        print(f"  Worst U_t: {worst_training.id} (score={worst_training.score:.4f})", flush=True)
        print(f"  Best  N_t: {best_challenger.id} (score={best_challenger.score:.4f})", flush=True)

        if gap > 0:
            self.no_swap_count = 0
            data_manager.training_arms.remove(worst_training.id)
            data_manager.challenger_arms.add(worst_training.id)
            data_manager.challenger_arms.remove(best_challenger.id)
            data_manager.training_arms.add(best_challenger.id)
            print(f"  >> SWAPPED: {worst_training.id} -> N_t, {best_challenger.id} -> U_t", flush=True)
            return True, worst_training.id, best_challenger.id, gap, gap_index

        self.no_swap_count += 1
        print(f"  >> No swap (streak: {self.no_swap_count})", flush=True)
        return False, None, None, gap, gap_index

    def reconstruct_nt(self, data_manager, mt_ids):
        """Paper step 15: N_t ← top_m'(M_t ∪ N_{t-1}; ρ̂(t))."""
        if not mt_ids:
            return
        pool_ids = list(data_manager.challenger_arms) + list(mt_ids)
        pool_arms = [(aid, data_manager.all_arms[aid].score) for aid in pool_ids]
        pool_arms.sort(key=lambda x: x[1], reverse=True)

        m_prime = data_manager.num_challenger_arms
        new_nt = set()
        demoted = []
        promoted = []

        for rank, (aid, score) in enumerate(pool_arms):
            if rank < m_prime:
                new_nt.add(aid)
                if aid in mt_ids and aid not in data_manager.challenger_arms:
                    promoted.append(aid)
            else:
                if aid in data_manager.challenger_arms:
                    demoted.append(aid)

        for aid in promoted:
            data_manager.rest_arms.discard(aid)
        for aid in demoted:
            data_manager.rest_arms.add(aid)
        data_manager.challenger_arms = new_nt

        if promoted:
            print(f"  >> N_t reconstructed: promoted {promoted} from M_t -> N_t", flush=True)
        if demoted:
            print(f"  >> N_t reconstructed: demoted {demoted} from N_t -> rest", flush=True)
        if not promoted and not demoted:
            print(f"  >> N_t reconstructed: no changes (M_t arms scored lower)", flush=True)

    def has_converged(self, data_manager, bandit):
        """Paper step 7: convergence when gap-index criterion is met."""
        data_manager.update_arm_scores(bandit.alpha)
        gap, worst_ut, best_nt = self.compute_gap(data_manager)
        if worst_ut is None or best_nt is None:
            return False

        gap_index = self.compute_gap_index(worst_ut, best_nt, bandit)

        gap_converged = abs(gap) <= gap_index and gap_index < 0.5

        if gap_converged:
            print(f"\n>> CONVERGED (gap-index)! |gap|={abs(gap):.4f} <= gap_index={gap_index:.4f}",
                  flush=True)
            return True
        return False


# ═══════════════════════════════════════════════════════════════════
# OLMo Prompt Formatting
# ═══════════════════════════════════════════════════════════════════
def format_olmo_prompt(user_content, compression_ratio=None):
    """
    Format prompt using OLMo's template matching LlamaFactory SFT training.
    Uses <|user|> / <|assistant|> tags (no system block).
    """
    if compression_ratio is not None and compression_ratio < 1.0:
        prompt = (
            f"<|user|>\nPlease reason step by step, be as concise as possible, and put your final answer within \\boxed{{}}.\n"
            f"{user_content}<|eot_id|>{compression_ratio}<|eot_id|>\n"
            f"<|assistant|>\n"
        )
    else:
        prompt = (
            f"<|user|>\nPlease reason step by step, be as concise as possible, and put your final answer within \\boxed{{}}.\n"
            f"{user_content}\n"
            f"<|assistant|>\n"
        )
    return prompt


def prepare_finetuning_data(samples):
    """Prepare training data for SFT from Sample objects."""
    finetuning_data = []
    for sample in samples:
        # The prompt field already has the full chat template from the original data
        finetuning_data.append({
            "prompt": sample.prompt,
            "output": sample.compressed_trace or "",
            "id": sample.id,
            "target_compression_ratio": sample.target_compression_ratio
        })
    return finetuning_data


def compute_reward(accuracy, avg_compression_ratio, beta=0.3):
    return accuracy + beta * (1.0 - avg_compression_ratio)


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
# OLMo SFT Trainer
# ═══════════════════════════════════════════════════════════════════
@dataclass
class SFTConfig:
    model_name: str = "allenai/Olmo-3-7B-Instruct"
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
            # OLMo architecture uses fused qkv_proj instead of separate q/k/v
            self.target_modules = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]


class OlmoSFTTrainer:
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
            self.config.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa"
        )


        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        self.model.enable_input_require_grads()
        self.is_loaded = True

    def reset_adapter(self):
        # Model is reloaded fresh when SFT starts, so manual reset is not needed
        pass


    def train(self, training_samples, output_dir, iteration=0, num_epochs=None):
        if not self.is_loaded:
            self.load_model()

        texts = []
        for s in training_samples:
            # OLMo uses <|endoftext|> as EOS token
            text = s.get('prompt', '') + s.get('output', '') + '<|endoftext|>\n'
            texts.append(text)

        train_dataset = Dataset.from_dict({'text': texts})
        adapter_path = os.path.join(output_dir, f'adapter_iter_{iteration}')
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
            save_strategy='no',
            bf16=True,
            optim='adamw_torch',
            report_to='none',
            remove_unused_columns=False,
            gradient_checkpointing=True
        )


        def tok_fn(ex):
            return self.tokenizer(
                ex['text'], truncation=True,
                max_length=self.config.max_seq_length,
                padding='max_length'
            )

        tok_ds = train_dataset.map(tok_fn, batched=True, remove_columns=['text'])
        print(f'Training iter {iteration} ({len(texts)} samples, {epochs} epochs)...', flush=True)


        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tok_ds,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=False
            )
        )
        trainer.train()
        self.model.save_pretrained(adapter_path)
        print(f'Saved to {adapter_path}', flush=True)
        self.latest_adapter_path = adapter_path
        return adapter_path


    def infer(self, test_data, answer_extraction_fn, max_new_tokens=2048):
        """Run inference on test data with OLMo prompt formatting."""
        prompts = []
        model = self.model
        tokenizer = self.tokenizer

        for example in test_data:
            # Build OLMo-formatted prompt from messages
            user_content = ""
            for mess in example['messages']:
                if mess['role'] == 'user':
                    user_content = mess['content']

            prompt = format_olmo_prompt(user_content)
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
            extract_math_answer(
                item['messages'][-2]['content'] if isinstance(item.get('messages'), list) and len(item['messages']) >= 2
                    else item.get('prompt', ''),
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
            pbar.set_postfix({"avg_s/sample": f"{avg_time:.3f}"})

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
    def evaluate(self, val_samples, max_new_tokens=2048, batch_size=16):
        """Evaluation supporting vLLM-acceleration with a fallback to HF generation."""
        if not HAS_VLLM:
            # Fallback to HF generation
            self.model.eval()
            total = len(val_samples)
            correct_count = 0
            actual_crs = []
            num_batches = (total + batch_size - 1) // batch_size
            print(f"\n>>> EVALUATING {total} SAMPLES in {num_batches} batches (bs={batch_size}) [HF Fallback] <<<", flush=True)
            all_results = []
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, total)
                batch_samples = val_samples[start:end]
                results = self.infer(batch_samples, "extract_math_answer", max_new_tokens)
                for j, item in enumerate(results["results"]):
                    gen_len = item["cot_length"]
                    orig_len = batch_samples[j].get("cot_length", gen_len)
                    actual_cr = gen_len / orig_len if orig_len > 0 else 1.0
                    actual_crs.append(actual_cr)
                    pred_answer = item["prediction"][0] if item["prediction"] else ""
                    true_answer = batch_samples[j]["answer"][0] if batch_samples[j]["answer"] else ""
                    is_correct = False
                    if pred_answer and true_answer:
                        is_correct = (
                            pred_answer.lower() == true_answer.lower()
                            or true_answer.lower() in pred_answer.lower()
                            or pred_answer.lower() in true_answer.lower()
                        )
                    if is_correct:
                        correct_count += 1
                    print(f"  [{start+j+1}/{total}] CR={actual_cr:.2f}, correct={is_correct}", flush=True)
                all_results.extend(results["results"])
            pred_file = os.path.join(OUTPUT_DIR, "bandit_predictions.jsonl")
            with open(pred_file, "w") as f:
                for p in all_results:
                    f.write(json.dumps(p, default=str) + "\n")
            accuracy = correct_count / total if total > 0 else 0
            avg_cr = sum(actual_crs) / len(actual_crs) if actual_crs else 1.0
            print(f"\n>>> Acc={accuracy:.2%} ({correct_count}/{total}), Avg_CR={avg_cr:.4f} <<<\n", flush=True)
            return {"accuracy": accuracy, "avg_cr": avg_cr}

        # vLLM-accelerated evaluation
        import gc
        
        # Unload the Hugging Face model from GPU memory to prevent OOM
        if self.model is not None:
            print("Unloading Hugging Face model from GPU to free memory for vLLM...", flush=True)
            del self.model
            self.model = None
            self.is_loaded = False
            gc.collect()
            torch.cuda.empty_cache()

        # Format prompts for evaluation
        prompts = []
        for example in val_samples:
            user_content = ""
            for mess in example['messages']:
                if mess['role'] == 'user':
                    user_content = mess['content']
            prompt = format_olmo_prompt(user_content)
            example['prompt'] = prompt
            prompts.append(prompt)

        print(f"\n>>> EVALUATING {len(val_samples)} SAMPLES WITH vLLM <<<", flush=True)

        # Load vLLM
        print("Loading vLLM engine...", flush=True)
        llm = LLM(
            model=self.config.model_name,
            trust_remote_code=True,
            enable_lora=True,
            max_model_len=2048,
            gpu_memory_utilization=0.70
        )


        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
            stop=["<|endoftext|>"]
        )

        lora_request = None
        if self.latest_adapter_path is not None:
            print(f"Loading adapter: {self.latest_adapter_path}", flush=True)
            lora_request = LoRARequest("bandit_adapter", 1, self.latest_adapter_path)

        torch.cuda.synchronize()
        start_time = time()
        
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
        
        torch.cuda.synchronize()
        total_time = time() - start_time

        # Sort outputs by request_id and extract text
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        model_outputs = [output.outputs[0].text for output in outputs]

        # Destroy vLLM engine immediately to free GPU memory for SFT
        print("Destroying vLLM engine and freeing GPU memory...", flush=True)
        if hasattr(llm, "llm_engine") and hasattr(llm.llm_engine, "model_executor"):
            llm.llm_engine.model_executor.shutdown()
        
        from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
        try:
            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass
        
        del llm
        gc.collect()
        torch.cuda.empty_cache()

        # Parse and evaluate predictions
        cot_lengths = []
        for model_completion in model_outputs:
            cot = model_completion.split('\n\nThe final answer is:')[0]
            cot_length = self.tokenizer(cot, return_tensors="pt")['input_ids'].shape[1]
            cot_lengths.append(cot_length)

        predictions = [
            extract_math_answer(
                item['messages'][-2]['content'] if isinstance(item.get('messages'), list) and len(item['messages']) >= 2
                    else item.get('prompt', ''),
                output,
                task='cot'
            )
            for item, output in zip(val_samples, model_outputs)
        ]

        correct_count = 0
        actual_crs = []
        all_results = []

        for j, (example, output, pred, cot_length) in enumerate(zip(val_samples, model_outputs, predictions, cot_lengths)):
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

            gen_len = item["cot_length"]
            orig_len = val_samples[j].get("cot_length", gen_len)
            actual_cr = gen_len / orig_len if orig_len > 0 else 1.0
            actual_crs.append(actual_cr)

            pred_answer = item["prediction"][0] if item["prediction"] else ""
            true_answer = val_samples[j]["answer"][0] if val_samples[j]["answer"] else ""

            is_correct = False
            if pred_answer and true_answer:
                is_correct = (
                    pred_answer.lower() == true_answer.lower()
                    or true_answer.lower() in pred_answer.lower()
                    or pred_answer.lower() in true_answer.lower()
                )

            if is_correct:
                correct_count += 1

            all_results.append(item)
            print(f"  [{j+1}/{len(val_samples)}] CR={actual_cr:.2f}, correct={is_correct}", flush=True)

        # Save predictions
        pred_file = os.path.join(OUTPUT_DIR, "bandit_predictions.jsonl")
        with open(pred_file, "w") as f:
            for p in all_results:
                f.write(json.dumps(p, default=str) + "\n")

        accuracy = correct_count / len(val_samples) if val_samples else 0
        avg_cr = sum(actual_crs) / len(actual_crs) if actual_crs else 1.0

        print(f"\n>>> Acc={accuracy:.2%} ({correct_count}/{len(val_samples)}), Avg_CR={avg_cr:.4f} <<<\n", flush=True)

        return {"accuracy": accuracy, "avg_cr": avg_cr}



# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════
CONFIG = {
    "model_name": "allenai/Olmo-3-7B-Instruct",
    "arm_size": 50,
    "num_training_arms": 10,
    "num_challenger_arms": 10,
    "num_total_traces": 2000,        # More traces to create rest pool for M_t
    "num_validation": 100,
    "num_exploration": 5,             # |M_t| per round
    "max_iterations": 30,
    "lambda_reg": 1.0,
    "confidence_scale": 1.0,
    "beta": 3.0,

    "random_seed": 42,
    "skip_sft": "--skip_sft" in sys.argv,
    "val_eval_size": 50,
    "gen_max_tokens": 2048,
    "eval_batch_size": 32,
}


# ═══════════════════════════════════════════════════════════════════
# Main Loop
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    total_arms = CONFIG['num_total_traces'] // CONFIG['arm_size']
    rest = total_arms - CONFIG['num_training_arms'] - CONFIG['num_challenger_arms']
    print(f"Arms: {total_arms} total, {CONFIG['num_training_arms']} U_t, {CONFIG['num_challenger_arms']} N_t, {rest} rest pool")
    print(f"M_t exploration: {CONFIG['num_exploration']} arms sampled per round")
    print(f"Reward: accuracy + {CONFIG['beta']} * (1 - avg_CR)")
    print(f"Mode: {'DRY RUN (skip_sft)' if CONFIG['skip_sft'] else 'FULL CASE ALGORITHM with SFT'}")

    print("=" * 60)
    print("CASE BANDIT: Algorithm 1 — OLMo-3-7B-Instruct on DeepMath")
    print("=" * 60)

    # Initialize data manager
    data_manager = SetBanditDataManager(
        CSV_PATH, JSONL_PATH,
        arm_size=CONFIG["arm_size"],
        num_training_arms=CONFIG["num_training_arms"],
        num_challenger_arms=CONFIG["num_challenger_arms"],
        num_total_traces=CONFIG["num_total_traces"],
        num_validation=CONFIG["num_validation"],
        num_exploration=CONFIG["num_exploration"]
    )
    feature_dim = len(FEATURE_COLUMNS)
    bandit = LinearBandit(
        feature_dim, CONFIG["lambda_reg"],
        learning_rate=CONFIG.get("learning_rate", 1.0),
        random_seed=CONFIG["random_seed"]
    )
    data_manager.form_arms_and_initialize(bandit.alpha, CONFIG["random_seed"])
    swap_manager = GapIndexSwapManager(confidence_scale=CONFIG["confidence_scale"])
    rng = np.random.RandomState(CONFIG["random_seed"])

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
        trainer = OlmoSFTTrainer(SFTConfig(model_name=CONFIG["model_name"]))
        trainer.load_model()

    # Load validation data
    history = []
    val_data = read_data(VAL_DATA_PATH)
    val_samples = val_data[:CONFIG["val_eval_size"]]

    # ── Main Bandit Loop ──
    for iteration in range(1, CONFIG["max_iterations"] + 1):
        print(f"\n{'#' * 60}")
        print(f"# ITERATION {iteration}/{CONFIG['max_iterations']}")
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
        if not CONFIG["skip_sft"]:
            print(f"\n[Step 21] Resetting LoRA adapter...", flush=True)
            trainer.reset_adapter()
            print(f"\n[Step 21] SFT on {len(training_data)} traces...", flush=True)
            trainer.train(training_data, OUTPUT_DIR, iteration)
            print(f"\n[Step 21] Evaluating {len(val_samples)} val samples...", flush=True)
            result = trainer.evaluate(val_samples, CONFIG["gen_max_tokens"], CONFIG["eval_batch_size"])
            accuracy, avg_cr = result["accuracy"], result["avg_cr"]
        else:
            accuracy = np.mean([1.0 if s.accuracy else 0.0 for s in arm_samples])
            avg_cr = np.mean([s.target_compression_ratio for s in arm_samples])

        # === Step 22-23: Update reward + ridge regression ===
        reward = compute_reward(accuracy, avg_cr, CONFIG["beta"])
        pulled_arm.record_pull(reward)
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
            "iter": iteration, "arm": pulled_arm.id, "arm_set": loc,
            "reward": reward, "acc": accuracy, "cr": avg_cr,
            "weights": bandit.alpha.tolist(),
            "swapped": swapped, "gap": gap, "gap_index": gap_idx,
            "mt_ids": [str(mid) for mid in mt_ids],
            "prediction_error": prediction_error,
            "ut_min": ut_min, "ut_max": ut_max, "ut_median": ut_median
        })

        # === Step 7: Convergence check ===
        if swap_manager.has_converged(data_manager, bandit):
            print(f"\nConverged at iteration {iteration}!", flush=True)
            break

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



        # ── Final Test Evaluation ──
        test_file = VAL_DATA_PATH
        raw_data = read_data(test_file)

        test_samples = []
        for d in raw_data:
            sample = d.copy()
            user_content = ""
            for mess in d['messages']:
                if mess['role'] == 'user':
                    user_content = mess['content']
            sample['prompt'] = format_olmo_prompt(user_content)

            # Calculate CR relative to original model output token length
            if 'model_output' in d:
                tokens = trainer.tokenizer.encode(d['model_output'], add_special_tokens=False)
                sample['cot_length'] = len(tokens)
            else:
                sample['cot_length'] = 1

            test_samples.append(sample)

        print(f"Prepared {len(test_samples)} samples for final evaluation.")
        result = trainer.evaluate(test_samples, max_new_tokens=2048, batch_size=CONFIG["eval_batch_size"])
        print(f"\nFinal Test Accuracy: {result['accuracy']:.2%}")
        print(f"Average Compression Ratio: {result['avg_cr']:.4f}")

