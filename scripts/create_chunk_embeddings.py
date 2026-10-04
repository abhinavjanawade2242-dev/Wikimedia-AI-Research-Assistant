import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer


DATA_PATH = "data/processed/chunks_25000.jsonl"

OUTPUT_PATH = (
    "data/processed/embeddings/chunk_embeddings_25000.npy"
)

MODEL_NAME = "all-MiniLM-L6-v2"

BATCH_SIZE = 64


def load_chunks():
    """Load chunks from JSONL."""

    chunks = []

    print("Loading chunks...")

    with open(
        DATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            chunk = json.loads(line)

            chunks.append(chunk)

    print(f"Loaded {len(chunks)} chunks.")

    return chunks


def create_embeddings(chunks):
    """Create embeddings for all chunks."""

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Embedding model loaded.")

    texts = [
        chunk["search_text"]
        for chunk in chunks
    ]

    print("\nCreating chunk embeddings...")

    print(
        f"Total chunks: {len(texts)}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return embeddings


def save_embeddings(embeddings):
    """Save embeddings to NumPy file."""

    output_directory = os.path.dirname(
        OUTPUT_PATH
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    np.save(
        OUTPUT_PATH,
        embeddings
    )

    print("\nEmbeddings saved successfully.")

    print(
        f"File: {OUTPUT_PATH}"
    )

    print(
        f"Shape: {embeddings.shape}"
    )


def main():

    print(
        "========== CHUNK EMBEDDING GENERATION =========="
    )

    chunks = load_chunks()

    embeddings = create_embeddings(
        chunks
    )

    save_embeddings(
        embeddings
    )

    print(
        "\n========== COMPLETE =========="
    )


if __name__ == "__main__":

    main()