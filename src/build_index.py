"""Chunk the corpus, embed it, and persist it into a local Chroma vector store."""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_PATH = BASE_DIR / "data" / "processed" / "corpus.jsonl"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "rag_corpus"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# SQuAD contexts are already short paragraphs (~100-150 words), so most need
# no splitting. This still chunks anything long, so the pipeline generalizes
# to a corpus of longer documents (e.g. scraped docs pages) without changes.
CHUNK_SIZE_WORDS = 180
CHUNK_OVERLAP_WORDS = 30


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def build_index() -> None:
    docs = [json.loads(line) for line in CORPUS_PATH.open()]

    ids, texts, metadatas = [], [], []
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            ids.append(f"{doc['id']}_{i}")
            texts.append(chunk)
            metadatas.append({"source_doc_id": doc["id"], "title": doc["title"], "chunk_index": i})

    print(f"{len(docs)} documents split into {len(texts)} chunks")

    print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding chunks...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    client.delete_collection(COLLECTION_NAME) if COLLECTION_NAME in [c.name for c in client.list_collections()] else None
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    print("Writing to Chroma...")
    batch = 500
    for i in tqdm(range(0, len(ids), batch)):
        collection.add(
            ids=ids[i : i + batch],
            embeddings=embeddings[i : i + batch].tolist(),
            documents=texts[i : i + batch],
            metadatas=metadatas[i : i + batch],
        )

    print(f"Indexed {collection.count()} chunks into '{CHROMA_DIR}'")


if __name__ == "__main__":
    build_index()
