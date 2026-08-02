
"""
reasoning_feature_extractor_features.py

Extracts:
signal_ratio
explore_score
distinct_1_ratio
distinct_2_ratio
distinct_3_ratio
top_k_mass_ratio
windowed_jaccard
longest_repeat_ratio
llm_judge_score

Requires llm_judge.py only if judge_model is supplied.
"""

import re
import json
from collections import Counter
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

try:
    from llm_judge import LLMJudge
except Exception:
    LLMJudge=None

EXPLORE_WORDS={"wait","alternatively","instead","however","perhaps","maybe","suppose","assume","consider","otherwise"}
MATH_PATTERN=re.compile(r"^\d+(\.\d+)?$|^[+\-*/=<>%^(){}\[\]]$|^(sqrt|frac|sum|prod|log|ln|sin|cos|tan|max|min)$",re.I)

class ReasoningFeatureExtractor:
    def __init__(self,tokenizer_name="Qwen/Qwen2.5-7B-Instruct",judge_model=None,top_k=10,window_size=64):
        self.tokenizer=AutoTokenizer.from_pretrained(tokenizer_name,trust_remote_code=True)
        self.top_k=top_k
        self.window_size=window_size
        self.judge=LLMJudge(judge_model) if (judge_model and LLMJudge) else None

    def tokenize(self,text):
        ids=self.tokenizer.encode(text,add_special_tokens=False)
        return self.tokenizer.convert_ids_to_tokens(ids)

    def clean(self,t): return t.replace("Ġ","").replace("▁","").lower()
    def ngrams(self,t,n): return [tuple(t[i:i+n]) for i in range(len(t)-n+1)]

    def signal_ratio(self,t):
        return sum(bool(MATH_PATTERN.match(self.clean(x))) for x in t)/len(t) if t else 0.0

    def explore_score(self,t):
        return 1-sum(self.clean(x) in EXPLORE_WORDS for x in t)/len(t) if t else 0.0

    def distinct(self,t,n):
        g=self.ngrams(t,n)
        return len(set(g))/len(g) if g else 0.0

    def topk(self,t):
        g=self.ngrams(t,2)
        if not g: return 0.0
        c=Counter(g)
        return sum(v for _,v in c.most_common(self.top_k))/len(g)

    def jac(self,a,b):
        A,B=set(a),set(b)
        return len(A&B)/len(A|B) if (A|B) else 1.0

    def windowed_jaccard(self,t):
        W=self.window_size
        if len(t)<2*W: return 0.0
        wins=[self.ngrams(t[i:i+W],2) for i in range(0,len(t)-W+1,W)]
        vals=[self.jac(wins[i],wins[i+1]) for i in range(len(wins)-1)]
        return float(np.mean(vals)) if vals else 0.0

    def longest_repeat_ratio(self,t,max_n=20):
        if not t: return 0.0
        best=0
        for n in range(1,min(max_n,len(t))+1):
            if any(v>1 for v in Counter(self.ngrams(t,n)).values()):
                best=n
        return best/len(t)

    def extract(self, traces, output_csv="reasoning_features.csv"):
        scores = self.judge.score_batch([tr.get("model_output", "") if isinstance(tr, dict) else tr for tr in traces]) if self.judge else [np.nan] * len(traces)
        rows = []
        for i, tr in enumerate(traces):
            if isinstance(tr, dict):
                sample_id = tr.get("id", "")
                trace_text = tr.get("model_output", "")
            else:
                sample_id = ""
                trace_text = tr
            tok = self.tokenize(trace_text)
            rows.append({
                "id": sample_id,
                "signal_ratio": self.signal_ratio(tok),
                "distinct_1_ratio": self.distinct(tok, 1),
                "distinct_2_ratio": self.distinct(tok, 2),
                "distinct_3_ratio": self.distinct(tok, 3),
                "distinct_4_ratio": self.distinct(tok, 4),
                "top_k_mass_ratio": self.topk(tok),
                "windowed_jaccard": self.windowed_jaccard(tok),
                "longest_repeat_ratio": self.longest_repeat_ratio(tok),
                "llm_judge_score": scores[i]
            })
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        return df

if __name__ == "__main__":

    model_name = "Qwen2.5-7B-Instruct"
    benchmark = "gsm8k"
    with open(f"outputs/{model_name}/bandit/{benchmark}/bandit_data.jsonl", "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    traces = [
        {
            "id": record.get("id", ""),
            "model_output": record.get("model_output", "")
        }
        for record in records
    ]

    extractor = ReasoningFeatureExtractor(
        judge_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    )

    features = extractor.extract(
        traces,
        f"outputs/{model_name}/bandit/{benchmark}/reasoning_scores_subset.csv",
    )

