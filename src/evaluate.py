"""Evaluation harness for the RAG assistant.

Computes three families of metrics over a sample of labeled questions:

1. Retrieval quality (does the vector search find the right document?)
   - Recall@k / Precision@k against the known gold source document
   - Mean Reciprocal Rank (MRR)

2. Answer correctness (does the generated answer match the known gold answer?)
   - Token-level F1 and Exact Match, using the standard SQuAD scoring method

3. Faithfulness (does the answer only use facts present in the retrieved
   context, or did the model hallucinate / use outside knowledge?)
   - Scored by using the LLM itself as a judge against the retrieved context

Results are written per-question to eval_results/results.csv and summarized
in eval_results/summary.json, plus a bar chart in eval_results/metrics.png.
"""

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import ollama
import pandas as pd
from tqdm import tqdm

from src.rag_pipeline import RagPipeline

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_SET_PATH = BASE_DIR / "data" / "processed" / "eval_set.jsonl"
RESULTS_DIR = BASE_DIR / "eval_results"
JUDGE_MODEL = "qwen2.5:7b-instruct"


# ---------- Answer correctness (SQuAD-style F1 / EM) ----------

def normalize_text(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_text(prediction) == normalize_text(gold))


def contains_gold(prediction: str, gold: str) -> float:
    """Looser correctness check: does the (normalized) prediction contain the
    gold answer as a substring? SQuAD gold answers are short extracted spans,
    but a good RAG assistant should answer in full sentences (e.g. "suspended
    sentences" -> "The protesters were given suspended sentences..."), which
    tanks strict F1/EM even when the fact stated is correct. This metric
    checks whether the correct fact actually made it into the answer."""
    return float(normalize_text(gold) in normalize_text(prediction))


def best_over_gold_answers(prediction: str, gold_answers: list[str], metric_fn) -> float:
    return max((metric_fn(prediction, g) for g in gold_answers), default=0.0)


# ---------- Retrieval metrics ----------

def retrieval_metrics(retrieved_doc_ids: list[str], gold_doc_id: str, k: int) -> dict:
    hit = gold_doc_id in retrieved_doc_ids
    rank = retrieved_doc_ids.index(gold_doc_id) + 1 if hit else None
    return {
        "recall_at_k": 1.0 if hit else 0.0,
        # exactly one relevant document per SQuAD question, so precision@k
        # collapses to 1/k when found, matching the single-relevant-item case
        "precision_at_k": (1.0 / k) if hit else 0.0,
        "reciprocal_rank": (1.0 / rank) if hit else 0.0,
    }


# ---------- Faithfulness (LLM-as-judge) ----------

# A model correctly declining to answer (because the retrieved context didn't
# contain the answer) is faithful by definition -- there's no claim to check
# for grounding. Scoring this with the LLM judge turned out to be unreliable:
# on a manual audit it flagged refusals as UNFAITHFUL about as often as not,
# even though the judge prompt explicitly said refusals are faithful. Rather
# than trust the judge on a case with an unambiguous rule-based answer, this
# short-circuits refusals to faithful=1.0 without spending a judge call on them.
REFUSAL_MARKERS = ("don't know", "do not know", "cannot answer", "can't answer")


def is_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


FAITHFULNESS_PROMPT = """You are auditing an AI assistant's answer for faithfulness to its source context.

Context:
{context}

Question: {question}
Answer given: {answer}

Does the answer rely ONLY on facts stated in the context (faithful), or does it add
claims not supported by the context, including outside/world knowledge (unfaithful)?

First, briefly list which specific facts in the answer are or are not supported by
the context (one line). Then, on a new final line, respond with exactly one word:
FAITHFUL or UNFAITHFUL."""


def judge_faithfulness(question: str, answer: str, context: str) -> float:
    if is_refusal(answer):
        return 1.0

    prompt = FAITHFULNESS_PROMPT.format(context=context, question=question, answer=answer)
    response = ollama.chat(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0},
    )
    # the verdict is the model's last line, after its one-line justification
    last_line = response["message"]["content"].strip().splitlines()[-1].upper()
    return 1.0 if "UNFAITHFUL" not in last_line and "FAITHFUL" in last_line else 0.0


# ---------- Main harness ----------

def run_evaluation(num_samples: int, k: int, seed: int) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    eval_rows = [json.loads(line) for line in EVAL_SET_PATH.open()]
    import random

    random.seed(seed)
    sample = random.sample(eval_rows, min(num_samples, len(eval_rows)))

    pipeline = RagPipeline()
    rows = []

    for row in tqdm(sample, desc="Evaluating"):
        question = row["question"]
        gold_doc_id = row["gold_context_id"]
        gold_answers = row["gold_answers"]

        chunks = pipeline.retrieve(question, k=k)
        retrieved_doc_ids = [c.source_doc_id for c in chunks]
        answer = pipeline.generate(question, chunks)

        r_metrics = retrieval_metrics(retrieved_doc_ids, gold_doc_id, k)
        f1 = best_over_gold_answers(answer, gold_answers, f1_score)
        em = best_over_gold_answers(answer, gold_answers, exact_match)
        contains = best_over_gold_answers(answer, gold_answers, contains_gold)

        context_text = "\n".join(c.text for c in chunks)
        faithful = judge_faithfulness(question, answer, context_text)

        rows.append(
            {
                "question": question,
                "generated_answer": answer,
                "gold_answers": " | ".join(gold_answers),
                "gold_doc_retrieved": bool(r_metrics["recall_at_k"]),
                "recall_at_k": r_metrics["recall_at_k"],
                "precision_at_k": r_metrics["precision_at_k"],
                "reciprocal_rank": r_metrics["reciprocal_rank"],
                "f1": f1,
                "exact_match": em,
                "contains_gold_answer": contains,
                "faithful": faithful,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "results.csv", index=False)

    summary = {
        "num_questions": len(df),
        "k": k,
        f"recall_at_{k}": round(df["recall_at_k"].mean(), 4),
        f"precision_at_{k}": round(df["precision_at_k"].mean(), 4),
        "mrr": round(df["reciprocal_rank"].mean(), 4),
        "answer_f1": round(df["f1"].mean(), 4),
        "answer_exact_match": round(df["exact_match"].mean(), 4),
        "answer_contains_gold": round(df["contains_gold_answer"].mean(), 4),
        "faithfulness_rate": round(df["faithful"].mean(), 4),
    }
    with (RESULTS_DIR / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Evaluation summary ===")
    for key, value in summary.items():
        print(f"{key:>20}: {value}")

    plot_summary(summary, k)


def plot_summary(summary: dict, k: int) -> None:
    metrics = {
        f"Recall@{k}": summary[f"recall_at_{k}"],
        f"Precision@{k}": summary[f"precision_at_{k}"],
        "MRR": summary["mrr"],
        "Answer F1": summary["answer_f1"],
        "Exact Match": summary["answer_exact_match"],
        "Contains Gold": summary["answer_contains_gold"],
        "Faithfulness": summary["faithfulness_rate"],
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(metrics.keys(), metrics.values(), color="#4C72B0")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(f"RAG Evaluation ({summary['num_questions']} questions)")
    ax.bar_label(bars, fmt="%.2f")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "metrics.png", dpi=150)
    print(f"\nSaved chart to {RESULTS_DIR / 'metrics.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    run_evaluation(args.num_samples, args.k, args.seed)
