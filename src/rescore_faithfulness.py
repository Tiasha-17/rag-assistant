"""Re-score faithfulness on an existing eval_results/results.csv using the
current judge_faithfulness() logic, without re-running retrieval+generation
(those are deterministic given temperature=0, so the saved answers are still
valid -- only the judging step changed).

Useful after fixing/tuning the faithfulness judge: this re-checks a completed
run in a couple of minutes instead of re-running the full ~30 min pipeline.
"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.evaluate import judge_faithfulness, plot_summary
from src.rag_pipeline import RagPipeline

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "eval_results"


def rescore(k: int = 4) -> None:
    results_path = RESULTS_DIR / "results.csv"
    df = pd.read_csv(results_path)

    pipeline = RagPipeline()
    new_faithful = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Re-scoring faithfulness"):
        chunks = pipeline.retrieve(row["question"], k=k)
        context_text = "\n".join(c.text for c in chunks)
        new_faithful.append(judge_faithfulness(row["question"], row["generated_answer"], context_text))

    old_rate = df["faithful"].mean()
    df["faithful"] = new_faithful
    new_rate = df["faithful"].mean()
    df.to_csv(results_path, index=False)

    summary_path = RESULTS_DIR / "summary.json"
    import json

    summary = json.loads(summary_path.read_text())
    summary["faithfulness_rate"] = round(new_rate, 4)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\nFaithfulness rate: {old_rate:.4f} -> {new_rate:.4f}")
    plot_summary(summary, k)


if __name__ == "__main__":
    rescore()
