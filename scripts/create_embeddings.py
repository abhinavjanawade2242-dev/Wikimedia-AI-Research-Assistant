import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/processed/articles_25000.jsonl"
OUTPUT_PATH = "data/processed/embeddings/article_embeddings_25000.npy"

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64


def load_articles():
    """Load processed articles from JSONL file."""
    articles = []

    print("Loading processed articles...")

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            article = json.loads(line)
            articles.append(article)

    print(f"Loaded {len(articles)} articles.")

    return articles


def create_search_text(article):
    """
    Create the text that will be converted into an embedding.
    """

    title = article.get("title") or ""
    description = article.get("description") or ""
    abstract = article.get("abstract") or ""
    full_text = article.get("full_text") or ""

    return f"""
Title: {title}

Description: {description}

Abstract: {abstract}

Article:
{full_text}
""".strip()


def create_embeddings(texts):
    """Generate embeddings in batches."""

    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded.")

    print(f"\nCreating embeddings...")
    print(f"Total texts: {len(texts)}")
    print(f"Batch size: {BATCH_SIZE}")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return embeddings


def save_embeddings(embeddings):
    """Save embeddings as NumPy array."""

    output_directory = os.path.dirname(OUTPUT_PATH)

    os.makedirs(output_directory, exist_ok=True)

    np.save(OUTPUT_PATH, embeddings)

    print("\nEmbeddings saved successfully.")
    print(f"File: {OUTPUT_PATH}")
    print(f"Shape: {embeddings.shape}")


def main():

    print("========== WIKIMEDIA EMBEDDING GENERATION ==========")

    articles = load_articles()

    print("\nPreparing article text...")

    texts = []

    for article in articles:
        text = create_search_text(article)
        texts.append(text)

    print(f"Prepared {len(texts)} texts.")

    embeddings = create_embeddings(texts)

    save_embeddings(embeddings)

    print("\n========== EMBEDDING GENERATION COMPLETE ==========")


if __name__ == "__main__":
    main()