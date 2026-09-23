"""Streamlit demo UI for the RAG assistant.

Run with: streamlit run app.py
"""

import json
from pathlib import Path

import streamlit as st

from src.rag_pipeline import RagPipeline

BASE_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = BASE_DIR / "eval_results" / "summary.json"

st.set_page_config(page_title="RAG Assistant", page_icon="🔎", layout="centered")


@st.cache_resource
def load_pipeline() -> RagPipeline:
    return RagPipeline()


st.title("🔎 RAG Assistant")
st.caption(
    "Retrieval-augmented Q&A over a Wikipedia passage corpus (SQuAD). "
    "Fully local: MiniLM embeddings + ChromaDB + Ollama (qwen2.5:7b-instruct)."
)

with st.sidebar:
    st.header("Evaluation harness")
    if SUMMARY_PATH.exists():
        summary = json.loads(SUMMARY_PATH.read_text())
        st.metric("Retrieval Recall@k", summary.get(f"recall_at_{summary['k']}", "-"))
        st.metric("Retrieval MRR", summary.get("mrr", "-"))
        st.metric("Answer F1 (strict)", summary.get("answer_f1", "-"))
        st.metric("Answer contains gold fact", summary.get("answer_contains_gold", "-"))
        st.metric("Faithfulness rate", summary.get("faithfulness_rate", "-"))
        st.caption(f"Scored on {summary['num_questions']} held-out labeled questions.")
        metrics_png = BASE_DIR / "eval_results" / "metrics.png"
        if metrics_png.exists():
            st.image(str(metrics_png))
    else:
        st.info("Run `python -m src.evaluate` to populate eval metrics here.")

    st.divider()
    k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=8, value=4)

pipeline = load_pipeline()

question = st.text_input("Ask a question", placeholder="e.g. When were stromules discovered?")

if question:
    with st.spinner("Retrieving context and generating answer..."):
        result = pipeline.answer(question, k=k)

    st.subheader("Answer")
    st.write(result.answer)

    st.subheader("Retrieved sources")
    for i, chunk in enumerate(result.chunks, start=1):
        with st.expander(f"[{i}] {chunk.title}  ·  similarity {chunk.similarity:.3f}"):
            st.write(chunk.text)
