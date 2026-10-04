import json
import pandas as pd

from src.article_parser import create_article_record


# ============================================================
# CONFIGURATION
# ============================================================

FILE_PATH = "data/raw/enwiki_namespace_0_00013.parquet"

OUTPUT_PATH = "data/processed/articles_25000.jsonl"

LIMIT = 25000


# ============================================================
# PROCESS ARTICLES
# ============================================================

def process_articles(limit=None):

    print("Loading dataset...")

    df = pd.read_parquet(FILE_PATH)

    print(
        f"Total articles available: {len(df)}"
    )

    if limit is None:
        limit = len(df)

    limit = min(limit, len(df))

    print(
        f"Processing {limit} articles..."
    )

    articles = []

    for position, (_, row) in enumerate(
        df.head(limit).iterrows(),
        start=1
    ):

        article = create_article_record(row)

        articles.append(article)

        if position % 1000 == 0:

            print(
                f"Processed {position}/{limit} articles..."
            )

    print(
        f"\nSuccessfully processed: "
        f"{len(articles)} articles"
    )

    return articles


# ============================================================
# SAVE ARTICLES
# ============================================================

def save_articles(articles):

    print("\nSaving processed articles...")

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        for article in articles:

            json.dump(
                article,
                file,
                ensure_ascii=False,
                default=str
            )

            file.write("\n")

    print(
        f"Saved articles to: {OUTPUT_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    articles = process_articles(LIMIT)

    save_articles(articles)

    print(
        "\n========== PROCESSING COMPLETE =========="
    )