"""
CASE Bandit Core — Full CASE Algorithm with M_t Exploration
===========================================================
Three-set structure matching the CASE paper:
  U_t: training arms (top-k)
  N_t: challenger arms (next-n)
  Rest pool: (U_t ∪ N_t)^c — source for M_t exploration

Each round:
  1. Sample M_t from rest pool
  2. CASE selection from U_t ∪ N_t ∪ M_t (most uncertain arm)
  3. Competitor for U_t arms = best(N_t ∪ M_t)
  4. Swap: worst(U_t) vs best(N_t ∪ M_t)
  5. M_t is ephemeral — returns to rest unless swapped into U_t
"""

import numpy as np
import pandas as pd
import json
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from pathlib import Path
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "signal_ratio",
    "distinct_1_ratio",
    "distinct_2_ratio",
    "distinct_3_ratio",
    "distinct_4_ratio",
    "top_k_mass_ratio",
    "windowed_jaccard",
    "longest_repeat_ratio",
    "llm_judge_score",
]

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
    original_data: dict = field(default_factory=dict)

    def to_dict(self):
        return {"id": self.id, "accuracy": self.accuracy,
                "original_cot_length": self.original_cot_length,
                "optimal_compression_ratio": self.optimal_compression_ratio,
                "prompt": self.prompt, "answer": self.answer,
                "compressed_trace": self.compressed_trace,
                "original_data": self.original_data}


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


class SetBanditDataManager:
    def __init__(self, csv_path, bandit_data_path, original_data_path, arm_size=25, num_training_arms=10,
                 num_challenger_arms=10, num_total_traces=500, num_validation=100,
                 num_exploration=5):
        self.csv_path = csv_path
        self.bandit_data_path = bandit_data_path
        self.original_data_path = original_data_path
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

        # Handle NaNs before scaling
        df_features[FEATURE_COLUMNS] = df_features[FEATURE_COLUMNS].fillna(
            df_features[FEATURE_COLUMNS].mean()
        )

        # Standardize features
        scaler = StandardScaler()
        df_features[FEATURE_COLUMNS] = scaler.fit_transform(
            df_features[FEATURE_COLUMNS]
        )

        bandit_trace_data = {}
        original_trace_data = {}
        with open(self.bandit_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                bandit_trace_data[data['id']] = data

        with open(self.original_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                original_trace_data[data['id']] = data

        for _, row in df_features.iterrows():
            sample_id = str(row['id'])
            if sample_id not in bandit_trace_data:
                continue
            bandit_trace = bandit_trace_data[sample_id]
            original_trace = original_trace_data[sample_id]
            features = np.array([row[col] for col in FEATURE_COLUMNS], dtype=np.float32)
            optimal_cr = bandit_trace.get('optimal_compression_ratio', 1.0)
            self.all_samples[sample_id] = Sample(
                id=sample_id, features=features,
                accuracy=bandit_trace.get('accuracy'),
                original_cot_length=original_trace.get('cot_length', 0),
                optimal_compression_ratio=optimal_cr,
                prompt=original_trace.get('prompt', ''),
                answer=original_trace.get('answer', ''),
                compressed_trace=bandit_trace.get('model_output', ''),
                original_data=bandit_trace
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


class LinearBandit:
    def __init__(self, feature_dim=9, lambda_reg=1.0, learning_rate=1.0, random_seed=42):
        self.feature_dim = feature_dim
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate
        rng = np.random.RandomState(random_seed)
        self.alpha = rng.randn(feature_dim) * 0.1
        self.X_history = []
        self.r_history = []
        # Cache A_inv for efficient confidence and greedy gain calls
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
            
        # Perform the blend (Weight Blending)
        self.alpha = (1 - self.learning_rate) * self.alpha + self.learning_rate * new_alpha

    def get_confidence_bound(self, arm_features):
        # Using pre-computed A_inv: sqrt(x^T A_inv x)
        return float(np.sqrt(arm_features @ self.A_inv @ arm_features))

    def get_weights(self):
        return {f"w_{i}": float(self.alpha[i]) for i in range(self.feature_dim)}


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

    # ---- CASE SAMPLING (Step 20) ----

    def select_most_uncertain_arm(self, data_manager, bandit):
        """
        Paper step 20: selection_rule(U_t, N_t).
        Picks the arm that maximizes the Information Gain (Greedy Variance Reduction)
        regarding the gap between the Worst U_t and Best N_t.
        
        This aligns with the GSM8K script: i* = argmin ||d||_{(A + x_k x_k^T)^{-1}}
        Using Sherman-Morrison for efficiency.
        """
        data_manager.update_arm_scores(bandit.alpha)
        
        training = data_manager.get_training_arms_list()
        challengers = data_manager.get_challenger_arms_list()

        if not training or not challengers:
            return data_manager.get_random_training_arm()

        worst_ut = min(training, key=lambda a: a.score)
        best_nt = max(challengers, key=lambda a: a.score)

        # Gap direction vector: Worst in Top-10 vs Best Challenger
        d = worst_ut.avg_features - best_nt.avg_features
        Ainv = bandit.A_inv
        
        best_arm = None
        max_gain = float('-inf')

        # Selection Union: U_t ∪ N_t
        candidate_arms = training + challengers
        for arm in candidate_arms:
            x = arm.avg_features
            # Reduction term from Sherman-Morrison: (d^T A^-1 x)^2 / (1 + x^T A^-1 x)
            # This is proportional to the literal variance reduction of the gap
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
        """
        Paper steps 8-13: Swap worst(U_t) vs best(N_t).
        Uses only N_t — M_t is NOT involved in swap.
        """
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
            # Remove worst from U_t, send to N_t
            data_manager.training_arms.remove(worst_training.id)
            data_manager.challenger_arms.add(worst_training.id)
            # Promote best N_t into U_t
            data_manager.challenger_arms.remove(best_challenger.id)
            data_manager.training_arms.add(best_challenger.id)
            print(f"  >> SWAPPED: {worst_training.id} -> N_t, {best_challenger.id} -> U_t", flush=True)
            return True, worst_training.id, best_challenger.id, gap, gap_index

        self.no_swap_count += 1
        print(f"  >> No swap (streak: {self.no_swap_count})", flush=True)
        return False, None, None, gap, gap_index

    def reconstruct_nt(self, data_manager, mt_ids):
        """
        Paper step 15: N_t ← top_m'(M_t ∪ N_{t-1}; ρ̂(t)).
        Reconstruct N_t by keeping the top-m' scored arms from
        M_t ∪ old_N_t. Remaining arms go to rest.
        |N_t| stays fixed at num_challenger_arms.
        """
        if not mt_ids:
            return
        # Scores are already up-to-date from execute_swap / main loop
        # Pool = current N_t ∪ M_t
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
                # Track M_t arms that got promoted into N_t
                if aid in mt_ids and aid not in data_manager.challenger_arms:
                    promoted.append(aid)
            else:
                # Demote to rest
                if aid in data_manager.challenger_arms:
                    demoted.append(aid)

        # Apply changes
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
        
        # 1. CASE criterion (gap_index < 0.5 guard prevents auto-converge at step 1)
        gap_converged = abs(gap) <= gap_index and gap_index < 0.5

        if gap_converged:
            print(f"\n>> CONVERGED (gap-index)! |gap|={abs(gap):.4f} <= gap_index={gap_index:.4f}",
                  flush=True)
            return True
        return False


def prepare_finetuning_data(samples):
    finetuning_data = []
    for sample in samples:
        # Keep the prompt as-is — it already contains the full chat template
        # (system + user + assistant prefix). No need to strip and re-wrap.
        finetuning_data.append({
            "prompt": sample.prompt,
            "output": sample.compressed_trace or "",
            "id": sample.id,
            "optimal_compression_ratio": sample.optimal_compression_ratio
        })
    return finetuning_data


def compute_reward(accuracy, avg_compression_ratio, beta=0.3):
    return accuracy + beta * (1.0 - avg_compression_ratio)


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