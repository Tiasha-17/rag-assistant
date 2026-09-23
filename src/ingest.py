"""Download a subset of SQuAD and split it into:
  - data/processed/corpus.jsonl   the documents the RAG assistant retrieves from
  - data/processed/eval_set.jsonl labeled questions used by the evaluation harness

Each SQuAD row already pairs a question with the exact paragraph (context) that
contains the answer, so this gives us free ground truth for retrieval metrics
(precision/recall) without hand-labeling anything.
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def build(num_questions: int, seed: int) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading SQuAD (validation split) from Hugging Face...")
    ds = load_dataset("rajpurkar/squad", split="validation")
    ds = ds.shuffle(seed=seed).select(range(min(num_questions, len(ds))))

    contexts: dict[str, dict] = {}
    eval_rows = []

    for row in ds:
        context_text = row["context"]
        context_id = str(hash(context_text) & 0xFFFFFFFF)

        if context_id not in contexts:
            contexts[context_id] = {
                "id": context_id,
                "title": row["title"],
                "text": context_text,
            }

        eval_rows.append(
            {
                "question": row["question"],
                "gold_context_id": context_id,
                "gold_answers": row["answers"]["text"],
            }
        )

    corpus_path = PROCESSED_DIR / "corpus.jsonl"
    with corpus_path.open("w") as f:
        for doc in contexts.values():
            f.write(json.dumps(doc) + "\n")

    eval_path = PROCESSED_DIR / "eval_set.jsonl"
    with eval_path.open("w") as f:
        for row in eval_rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(contexts)} documents to {corpus_path}")
    print(f"Wrote {len(eval_rows)} labeled eval questions to {eval_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-questions", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(args.num_questions, args.seed)
