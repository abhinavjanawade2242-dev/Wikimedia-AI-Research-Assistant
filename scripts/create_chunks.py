import json
import os


DATA_PATH = "data/processed/articles_25000.jsonl"
OUTPUT_PATH = "data/processed/chunks_25000.jsonl"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def load_articles():
    """Load processed Wikimedia articles."""

    articles = []

    print("Loading articles...")

    with open(DATA_PATH, "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            article = json.loads(line)

            articles.append(article)

    print(f"Loaded {len(articles)} articles.")

    return articles


def create_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""

    if not text:
        return []

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk_words = words[start:end]

        chunk = " ".join(chunk_words)

        if chunk.strip():
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def create_article_chunks(article):
    """Create chunks for one article."""

    title = article.get("title") or ""
    description = article.get("description") or ""
    abstract = article.get("abstract") or ""
    full_text = article.get("full_text") or ""

    chunks = create_chunks(full_text)

    results = []

    for chunk_number, chunk_text in enumerate(
        chunks,
        start=1
    ):

        search_text = f"""
Title: {title}

Description: {description}

Abstract: {abstract}

Article Content:
{chunk_text}
""".strip()

        results.append({
            "article_id": article.get("id"),
            "title": title,
            "description": description,
            "abstract": abstract,
            "chunk_id": chunk_number,
            "text": chunk_text,
            "search_text": search_text,
            "url": article.get("url") or ""
        })

    return results


def save_chunks(chunks):
    """Save chunks to JSONL."""

    output_directory = os.path.dirname(OUTPUT_PATH)

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    print("\nSaving chunks...")

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        for chunk in chunks:

            json.dump(
                chunk,
                file,
                ensure_ascii=False
            )

            file.write("\n")

    print(
        f"Saved chunks to: {OUTPUT_PATH}"
    )


def main():

    print(
        "========== WIKIMEDIA CHUNK CREATION =========="
    )

    articles = load_articles()

    all_chunks = []

    print("\nCreating chunks...")

    for position, article in enumerate(
        articles,
        start=1
    ):

        article_chunks = create_article_chunks(
            article
        )

        all_chunks.extend(article_chunks)

        if position % 1000 == 0:

            print(
                f"Processed {position}/{len(articles)} articles..."
            )

    print(
        f"\nTotal chunks created: {len(all_chunks)}"
    )

    save_chunks(all_chunks)

    print(
        "\n========== CHUNKING COMPLETE =========="
    )


if __name__ == "__main__":

    main()
    