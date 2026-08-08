
"""
llm_judge.py

Compute LLM-as-Judge score (average token log probability) for reasoning traces
using a Hugging Face causal language model.

Usage:
    from llm_judge import LLMJudge

    judge = LLMJudge("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    scores = judge.score_batch(traces)

Returns one score per trace. Higher (less negative) is better.
"""

from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMJudge:
    def __init__(
        self,
        model_name: str,
        dtype=torch.bfloat16,
        device_map="auto",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()

    @torch.no_grad()
    def score(self, trace: str) -> float:
        inputs = self.tokenizer(
            trace,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model(**inputs)

        logits = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = inputs.input_ids[:, 1:]

        log_probs = torch.log_softmax(
            shift_logits,
            dim=-1,
        )

        token_log_probs = log_probs.gather(
            -1,
            shift_labels.unsqueeze(-1),
        ).squeeze(-1)

        avg_logprob = token_log_probs.mean()
        return torch.exp(avg_logprob).item()

    @torch.no_grad()
    def score_batch(
        self,
        traces: List[str],
        batch_size: int = 8,
    ) -> List[float]:

        scores = []

        for start in range(0, len(traces), batch_size):

            batch = traces[start:start + batch_size]

            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=False,
            ).to(self.model.device)

            outputs = self.model(**enc)

            logits = outputs.logits

            shift_logits = logits[:, :-1]
            shift_labels = enc.input_ids[:, 1:]
            attention = enc.attention_mask[:, 1:]

            log_probs = torch.log_softmax(
                shift_logits,
                dim=-1,
            )

            token_log_probs = log_probs.gather(
                -1,
                shift_labels.unsqueeze(-1),
            ).squeeze(-1)

            token_log_probs *= attention

            lengths = attention.sum(dim=1)

            avg_logprob = token_log_probs.sum(dim=1) / lengths

            # Geometric mean token probability
            scores.extend(torch.exp(avg_logprob).cpu().tolist())

        return scores


if __name__ == "__main__":

    traces = [
        "Let's compute the answer carefully. We have x+y=5. Therefore the answer is 5.",
        "Wait. Alternatively suppose x=2 and y=3. Hence x+y=5."
    ]

    judge = LLMJudge(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    )

    scores = judge.score_batch(traces)

    for t, s in zip(traces, scores):
        print("=" * 80)
        print(t)
        print(f"Average log probability: {s:.4f}")
