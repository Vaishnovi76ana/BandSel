"""
Compute reasoning quality scores for bandit_data.jsonl entries (OLMo DeepMath).

Metrics computed on the 'compressed_trace' field:
  - signal_ratio:       fraction of tokens containing math/numbers/LaTeX
  - distinct_1/2/3/4_ratio: unique n-grams / total n-grams
  - top_k_mass_ratio:   probability mass in top-10% frequent tokens
  - windowed_jaccard:   avg Jaccard similarity between consecutive windows
  - longest_repeat_ratio: longest repeated substring / total char length
  - llm_judge_score:    heuristic composite of the above features

Usage:
    python compute_reasoning_scores.py
"""

import json
import re
import csv
from collections import Counter


# ── helpers ──────────────────────────────────────────────────────────
def distinct_n_ratio(tokens, n):
    """Ratio of unique n-grams to total n-grams."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 0.0


def signal_ratio(tokens):
    """Fraction of tokens that contain mathematical / LaTeX content."""
    if not tokens:
        return 0.0
    math_pattern = re.compile(
        r'[0-9\+\-\*/=\^{}()\[\]<>]|\\\\|boxed|frac|sqrt|cdot|times|'
        r'implies|therefore|leq|geq|neq|sum|prod|int|log|sin|cos|tan|'
        r'infty|pi|theta|alpha|beta|gamma|delta|lambda|phi|psi|omega'
    )
    count = sum(1 for t in tokens if math_pattern.search(t))
    return count / len(tokens)


def top_k_mass(tokens, k_frac=0.1):
    """Probability mass of the top-k% most frequent tokens."""
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    k = max(1, int(k_frac * len(freq)))
    top_counts = sum(c for _, c in freq.most_common(k))
    return top_counts / len(tokens)


def windowed_jaccard(tokens, n_windows=10):
    """Average Jaccard similarity between consecutive equal-size windows."""
    n = len(tokens)
    if n < 20:
        return 0.0
    window_size = max(10, n // n_windows)
    scores = []
    for i in range(0, n - 2 * window_size + 1, window_size):
        w1 = set(tokens[i:i + window_size])
        w2 = set(tokens[i + window_size:i + 2 * window_size])
        union = w1 | w2
        if union:
            scores.append(len(w1 & w2) / len(union))
    return sum(scores) / len(scores) if scores else 0.0


def longest_repeat_ratio(text, max_check=200):
    """Ratio of the longest repeated substring to total string length."""
    n = len(text)
    if n < 10:
        return 0.0
    best = 0
    hi = min(max_check, n // 2)
    for length in range(hi, 3, -1):
        seen = set()
        found = False
        for i in range(n - length + 1):
            sub = text[i:i + length]
            if sub in seen:
                best = length
                found = True
                break
            seen.add(sub)
        if found:
            break
    return best / n if n > 0 else 0.0


def llm_judge_heuristic(sig, red, wj, topk, lr, explore):
    """
    Heuristic composite score approximating reasoning quality.
    High signal, low redundancy, low repetition ≈ better reasoning.
    """
    score = (
        0.35 * sig
        + 0.25 * (1 - red)
        + 0.15 * (1 - wj)
        + 0.10 * (1 - topk)
        + 0.10 * (1 - lr)
        + 0.05 * explore
    )
    return max(0.0, min(1.0, score))


# ── main scoring function ───────────────────────────────────────────
def compute_scores(text):
    """Compute all reasoning quality scores for a text trace."""
    tokens = text.split()
    n = len(tokens)

    if n == 0:
        return {
            'redundancy_ratio': 0, 'signal_ratio': 0, 'explore_ratio': 1,
            'distinct_1_ratio': 0, 'distinct_2_ratio': 0,
            'distinct_3_ratio': 0, 'distinct_4_ratio': 0,
            'top_k_mass_ratio': 0, 'windowed_jaccard': 0,
            'longest_repeat_ratio': 0, 'llm_judge_score': 0,
        }

    d1 = distinct_n_ratio(tokens, 1)
    d2 = distinct_n_ratio(tokens, 2)
    d3 = distinct_n_ratio(tokens, 3)
    d4 = distinct_n_ratio(tokens, 4)
    sig = signal_ratio(tokens)
    exp = distinct_n_ratio(tokens, 2)  # explore ≈ unique bigrams / total
    topk = top_k_mass(tokens)
    wj = windowed_jaccard(tokens)
    lr = longest_repeat_ratio(' '.join(tokens))
    llm = llm_judge_heuristic(sig, d1, wj, topk, lr, exp)

    return {
        'redundancy_ratio': round(d1, 4),
        'signal_ratio': round(sig, 4),
        'explore_ratio': round(exp, 4),
        'distinct_1_ratio': round(d1, 4),
        'distinct_2_ratio': round(d2, 4),
        'distinct_3_ratio': round(d3, 4),
        'distinct_4_ratio': round(d4, 4),
        'top_k_mass_ratio': round(topk, 4),
        'windowed_jaccard': round(wj, 4),
        'longest_repeat_ratio': round(lr, 4),
        'llm_judge_score': round(llm, 4),
    }


# ── main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    BANDIT = "dataset/bandit_data.jsonl"
    CSV_OUT = "dataset/reasoning_scores.csv"

    HEADER = [
        "redundancy_ratio", "signal_ratio", "explore_ratio",
        "distinct_1_ratio", "distinct_2_ratio", "distinct_3_ratio",
        "distinct_4_ratio", "top_k_mass_ratio", "windowed_jaccard",
        "longest_repeat_ratio", "llm_judge_score", "id",
    ]

    # 1. Load all entries from bandit_data.jsonl
    all_entries = []
    with open(BANDIT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_entries.append(json.loads(line))

    print(f"Total entries to score: {len(all_entries)}")

    # 2. Compute scores for every entry
    rows = []
    for i, entry in enumerate(all_entries):
        trace = entry.get("compressed_trace", "")
        scores = compute_scores(trace)
        row = [
            scores['redundancy_ratio'],
            scores['signal_ratio'],
            scores['explore_ratio'],
            scores['distinct_1_ratio'],
            scores['distinct_2_ratio'],
            scores['distinct_3_ratio'],
            scores['distinct_4_ratio'],
            scores['top_k_mass_ratio'],
            scores['windowed_jaccard'],
            scores['longest_repeat_ratio'],
            scores['llm_judge_score'],
            entry["id"],
        ]
        rows.append(row)
        if (i + 1) % 500 == 0 or (i + 1) == len(all_entries):
            print(f"  Computed {i + 1}/{len(all_entries)} ...", flush=True)

    # 3. Write fresh CSV
    with open(CSV_OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for row in rows:
            writer.writerow(row)

    print(f"\nDone! Wrote {len(rows)} entries to {CSV_OUT}")
