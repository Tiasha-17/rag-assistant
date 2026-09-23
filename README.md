# RAG Assistant with an Evaluation Harness

A retrieval-augmented question-answering assistant, built with a full evaluation
harness rather than just a chat demo. Most beginner RAG projects stop at "it
answers questions." This one also measures *how well* it retrieves and *whether
it can be trusted* — retrieval precision/recall, answer correctness, and
answer faithfulness (hallucination detection) — against a labeled question set.

Runs entirely locally and for free: no API keys, no cloud costs.

## Project Overview

1. Download a labeled Q&A dataset (SQuAD) and split it into a document corpus
   and a held-out evaluation set
2. Chunk and embed the corpus, store it in a local vector database
3. Answer questions with a retrieve-then-generate pipeline using a local LLM
4. Score the pipeline against the held-out labeled questions:
   - **Retrieval quality**: Recall@k, Precision@k, Mean Reciprocal Rank
   - **Answer correctness**: SQuAD-style F1/Exact Match, plus a looser
     "contains the correct fact" metric (full-sentence answers are penalized
     unfairly by strict span-matching — see [README_DETAILED.md](README_DETAILED.md))
   - **Faithfulness**: an LLM-as-judge check for whether the answer is
     actually supported by the retrieved context, or hallucinated
5. Serve it behind a Streamlit demo UI that shows retrieved sources alongside
   every answer

## Tech Stack

- Python
- `sentence-transformers` (MiniLM embeddings, local, free)
- ChromaDB (local vector store)
- Ollama running `qwen2.5:7b-instruct` (local LLM, free, no API key)
- pandas / matplotlib (metrics + charts)
- Streamlit (demo UI)
- Hugging Face `datasets` (SQuAD)

## Repository Structure

```text
.
├── src/
│   ├── ingest.py         # download SQuAD, build corpus + labeled eval set
│   ├── build_index.py    # chunk, embed, and index the corpus into Chroma
│   ├── rag_pipeline.py   # retrieve-then-generate RAG pipeline
│   └── evaluate.py       # evaluation harness (retrieval + correctness + faithfulness)
├── app.py                 # Streamlit demo UI
├── data/processed/        # corpus.jsonl, eval_set.jsonl
├── eval_results/          # results.csv, summary.json, metrics.png
├── requirements.txt
└── README_DETAILED.md
```

## How to Run

Requires [Ollama](https://ollama.com) installed locally with a model pulled:

```bash
ollama pull qwen2.5:7b-instruct
```

Install dependencies:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Build the corpus and index:

```bash
python src/ingest.py --num-questions 600
python src/build_index.py
```

Run the evaluation harness:

```bash
python -m src.evaluate --num-samples 150 --k 4
```

Launch the demo:

```bash
streamlit run app.py
```

## Evaluation Results

See [eval_results/summary.json](eval_results/summary.json) and
[eval_results/metrics.png](eval_results/metrics.png) for the latest run.
Per-question results (including every generated answer and its scores) are in
[eval_results/results.csv](eval_results/results.csv).

## What This Project Shows

- Building a full retrieve-then-generate RAG pipeline from scratch
- Designing a labeled evaluation set instead of eyeballing outputs
- Retrieval metrics (Recall@k, Precision@k, MRR)
- LLM-as-judge faithfulness scoring to catch hallucination
- Recognizing and correcting a flawed eval metric (strict span-matching
  penalizing correct, full-sentence answers), and debugging the LLM-as-judge
  faithfulness scorer itself after finding it mis-graded refusals — see the
  detailed writeup
- Running a complete LLM pipeline locally with zero API cost

## Detailed Walkthrough

[README_DETAILED.md](README_DETAILED.md)
