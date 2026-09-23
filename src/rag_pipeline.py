"""Core RAG pipeline: embed a query, retrieve top-k chunks from Chroma,
then ask a local Ollama model to answer using only those chunks."""

from dataclasses import dataclass
from pathlib import Path

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "rag_corpus"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "qwen2.5:7b-instruct"

SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. Answer the question using ONLY "
    "the information in the provided context. If the context does not contain the "
    "answer, say \"I don't know based on the given context.\" Keep answers short and "
    "directly quote or restate facts from the context rather than adding outside knowledge."
)


@dataclass
class RetrievedChunk:
    text: str
    source_doc_id: str
    title: str
    similarity: float


@dataclass
class RagResult:
    question: str
    answer: str
    chunks: list[RetrievedChunk]


class RagPipeline:
    def __init__(self):
        self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._collection = client.get_collection(COLLECTION_NAME)

    def retrieve(self, question: str, k: int = 4) -> list[RetrievedChunk]:
        query_embedding = self._embedder.encode([question], convert_to_numpy=True)[0].tolist()
        results = self._collection.query(query_embeddings=[query_embedding], n_results=k)

        chunks = []
        for text, meta, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            # Chroma's cosine "distance" is 1 - cosine_similarity; convert back for readability.
            similarity = 1 - distance
            chunks.append(
                RetrievedChunk(
                    text=text,
                    source_doc_id=meta["source_doc_id"],
                    title=meta["title"],
                    similarity=similarity,
                )
            )
        return chunks

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        context = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(chunks))
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.0},
        )
        return response["message"]["content"].strip()

    def answer(self, question: str, k: int = 4) -> RagResult:
        chunks = self.retrieve(question, k=k)
        answer = self.generate(question, chunks)
        return RagResult(question=question, answer=answer, chunks=chunks)


if __name__ == "__main__":
    pipeline = RagPipeline()
    result = pipeline.answer("What is the capital of France?")
    print("Q:", result.question)
    print("A:", result.answer)
    print("\nRetrieved chunks:")
    for c in result.chunks:
        print(f"  [{c.similarity:.3f}] {c.title}: {c.text[:100]}...")
