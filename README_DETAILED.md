# Detailed Build Notes

## Why an evaluation harness, not just a chat demo

Anyone can wire an LLM to a vector store and call it "RAG." The hard, useful
part — and the part hiring managers actually screen for — is knowing whether
the thing works: is retrieval finding the right documents, and is the model's
answer actually grounded in what was retrieved, or is it making things up?
This project treats evaluation as a first-class deliverable, not an
afterthought.

## Why SQuAD as the corpus

The evaluation harness needs ground truth to score against. SQuAD pairs each
question with the exact Wikipedia paragraph that contains the answer, so it
gives two things for free:

- A **retrieval label**: which document *should* have been retrieved for a
  given question, so Recall@k / Precision@k / MRR can be computed exactly.
- An **answer label**: the correct answer text, so generated answers can be
  scored for correctness.

A scraped docs site would look more "real world," but without hand-labeling
dozens of question/answer pairs there'd be nothing to score retrieval or
correctness against — the eval harness would just be vibes with extra steps.

## Architecture

```
question
   │
   ▼
[MiniLM embedder] ──► query vector
   │
   ▼
[ChromaDB similarity search] ──► top-k chunks (with similarity scores)
   │
   ▼
[qwen2.5:7b-instruct, temperature=0] ──► answer, constrained to only use
                                          the retrieved chunks
```

- **Embeddings**: `all-MiniLM-L6-v2` via `sentence-transformers` — small,
  fast, runs on CPU, no API key.
- **Vector store**: ChromaDB in persistent local mode (`./chroma_db`) —
  cosine similarity, no external service to stand up.
- **Chunking**: SQuAD paragraphs are already short (~100-150 words), so most
  pass through unchanged. `build_index.py` still splits anything longer
  (180-word windows, 30-word overlap) so the same pipeline works unmodified
  on a corpus of longer documents.
- **Generation**: Ollama running `qwen2.5:7b-instruct` locally, temperature 0
  for reproducible eval runs, with a system prompt that explicitly instructs
  the model to answer only from the provided context and say "I don't know"
  otherwise.

## The evaluation harness

Three metric families, computed per-question over a sample of labeled
questions and averaged (`src/evaluate.py`):

### 1. Retrieval quality

For each question, check whether the known correct source document appears
in the top-k retrieved chunks:

- **Recall@k** — was the correct document retrieved at all in the top k?
- **Precision@k** — since SQuAD has exactly one relevant document per
  question, this collapses to `1/k` when found, `0` otherwise.
- **MRR (Mean Reciprocal Rank)** — rewards ranking the correct document
  *higher*, not just retrieving it somewhere in the top k.

### 2. Answer correctness

Two versions, because the obvious one is misleading for a generative
assistant:

- **Strict F1 / Exact Match** — the standard SQuAD metric: token-overlap F1
  and exact-match against the gold answer span, after lowercasing, stripping
  punctuation and articles.
- **Contains-gold-answer** — a looser check for whether the gold answer
  string appears anywhere in the generated answer.

**Why both:** SQuAD's gold answers are short extracted spans ("1954",
"highly respected"), but a good RAG assistant answers in full sentences
("Stromules were first observed in 1962."). Scored strictly, that sentence
gets an F1 of ~0.3 even though it's completely correct — the metric is
penalizing good behavior (writing a real sentence) rather than measuring
answer quality. This was caught by inspecting the lowest-F1 transcripts during
development: every single one was a correct answer, just phrased naturally.
The contains-gold-answer metric fixes this by checking whether the right fact
made it into the answer, regardless of phrasing. Both are reported side by
side rather than silently swapping one for the other, since strict F1/EM is
still the standard way to compare against published SQuAD baselines.

### 3. Faithfulness (hallucination detection)

For each answer, a second LLM call acts as a judge: given the question, the
retrieved context, and the generated answer, it decides whether the answer
relies only on facts stated in the context (`FAITHFUL`) or introduces claims
not supported by it, including the model's own outside knowledge
(`UNFAITHFUL`). An answer that correctly says "I don't know" when the context
doesn't contain the answer counts as faithful — refusing to guess is the
correct behavior, not a failure.

This is the metric that actually catches the failure mode chat-only RAG
demos never surface: a model that sounds confident and coherent while quietly
answering from its own pretraining knowledge instead of the retrieved
documents.

### Debugging the judge itself

The first full run reported 88.67% faithfulness. Reading the individual
"unfaithful" transcripts (`eval_results/results.csv`) turned up a real bug:
**11 of the 17 flagged cases were the assistant correctly refusing to answer**
("I don't know based on the given context") — which the judge prompt
explicitly defines as faithful, yet the judge flagged it as unfaithful anyway.
A small 7B model apparently can't reliably hold that instruction across every
call. Two fixes went into `src/evaluate.py`:

1. **Rule-based short-circuit for refusals** — if the answer is a refusal, it's
   scored faithful=1.0 directly, without spending a judge call on a case that
   has an unambiguous correct answer.
2. **Chain-of-thought judging** — the judge prompt now asks for a one-line
   justification before the final verdict, a well-known technique for making
   small LLM judges more reliable.

Re-scoring (`src/rescore_faithfulness.py`, which re-uses the saved
deterministic answers instead of re-running the ~30-minute generation step)
brought faithfulness to **93.33%**.

Spot-checking the 10 still-flagged cases against their actual retrieved
context found the judge is *still* not perfectly reliable: at least 5 of the
10 (e.g. an answer of exactly `"Mercedes-Benz Superdome."` when the gold
answer is `"Mercedes-Benz Superdome"`) are near-verbatim quotes of the source
text that the judge flagged as unfaithful anyway — false positives. Two
others look like genuine model confusion (mixing up facts from two chunks).
This was left as-is rather than chased further: it's a known, well-documented
limitation of using a small model to grade itself, not something a prompt
tweak fully solves, and the honest number with a known error margin is more
useful here than an artificially polished one. See **Known limitations**
below.

## Known limitations

- **LLM-as-judge is itself an LLM call, and it's noisy.** Measured directly
  (see "Debugging the judge itself" above): on a manual audit of the 10
  cases the judge flagged as unfaithful in the final run, roughly half were
  false positives — answers that were near-verbatim, correctly grounded
  quotes of the retrieved context. The reported 93.33% faithfulness rate
  should be read as "at least ~93%, with a few points of judge noise in
  either direction," not as an exact figure. A production system would want
  either a materially stronger/independent judge model, or a sample of
  human-reviewed faithfulness labels to calibrate the judge against before
  trusting its number at face value.
- **Single relevant document per question** — real corpora often have
  multiple valid source documents per question, which would need a different
  precision/recall setup (graded relevance) than SQuAD's one-correct-document
  structure.
- **No re-ranking step** — retrieval is a single dense-similarity pass. A
  cross-encoder re-ranker after initial retrieval is the standard next
  improvement and would likely raise Precision@k.

## Possible extensions

- Swap in a cross-encoder re-ranker between retrieval and generation
- Add a hybrid retrieval mode (BM25 + dense) and compare metrics
- Track eval metrics over multiple corpus/prompt versions to show
  regression testing, not just a single snapshot
- Swap the local Ollama model for a hosted API model and compare
  cost/latency/quality trade-offs in the same harness
