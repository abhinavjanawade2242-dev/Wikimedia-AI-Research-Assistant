import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

from src.research import retrieve_research_articles


# ============================================================
# CONFIGURATION
# ============================================================

QUESTIONS_PATH = "evaluation_questions.json"

CHUNKS_PATH = (
    "data/processed/chunks_25000.jsonl"
)

EMBEDDINGS_PATH = (
    "data/processed/embeddings/"
    "chunk_embeddings_25000.npy"
)

OUTPUT_PATH = (
    "data/processed/"
    "retrieval_evaluation.json"
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 5


# ============================================================
# LOAD QUESTIONS
# ============================================================

def load_questions():

    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks():

    chunks = []

    print("Loading chunks...")

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            if line.strip():

                chunks.append(
                    json.loads(line)
                )

    print(
        f"Loaded {len(chunks)} chunks."
    )

    return chunks


# ============================================================
# EVALUATE QUESTION
# ============================================================

def evaluate_question(
    question,
    expected_articles,
    chunks,
    embeddings,
    model
):

    results = retrieve_research_articles(
        question=question,
        chunks=chunks,
        embeddings=embeddings,
        model=model,
        top_k=TOP_K,
        min_articles=2
    )

    retrieved_articles = []

    for result in results:

        title = result.get(
            "title",
            ""
        )

        if title not in retrieved_articles:

            retrieved_articles.append(
                title
            )

    expected_set = set(
        expected_articles
    )

    retrieved_set = set(
        retrieved_articles
    )

    relevant_retrieved = (
        expected_set.intersection(
            retrieved_set
        )
    )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    if retrieved_set:

        precision = (
            len(relevant_retrieved)
            / len(retrieved_set)
        )

    else:

        precision = 0.0

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    if expected_set:

        recall = (
            len(relevant_retrieved)
            / len(expected_set)
        )

    else:

        recall = 0.0

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    if precision + recall > 0:

        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    else:

        f1 = 0.0

    return {
        "question": question,
        "expected_articles": expected_articles,
        "retrieved_articles": retrieved_articles,
        "relevant_articles": list(
            relevant_retrieved
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    evaluation_results,
    average_precision,
    average_recall,
    average_f1
):

    os.makedirs(
        os.path.dirname(
            OUTPUT_PATH
        ),
        exist_ok=True
    )

    output = {
        "evaluation_config": {
            "embedding_model":
                EMBEDDING_MODEL,

            "top_k":
                TOP_K,

            "questions_file":
                QUESTIONS_PATH
        },

        "results":
            evaluation_results,

        "overall_metrics": {
            "average_precision":
                average_precision,

            "average_recall":
                average_recall,

            "average_f1":
                average_f1
        }
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"\nEvaluation report saved to:"
        f"\n{OUTPUT_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STAGE 23 - RESEARCH QUALITY EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load evaluation questions
    # --------------------------------------------------------

    questions = load_questions()

    print(
        f"\nEvaluation questions: "
        f"{len(questions)}"
    )

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    chunks = load_chunks()

    # --------------------------------------------------------
    # Load embeddings
    # --------------------------------------------------------

    print(
        "\nLoading embeddings..."
    )

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    print(
        f"Embeddings shape: "
        f"{embeddings.shape}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "\nLoading embedding model..."
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print(
        "Embedding model loaded."
    )

    # --------------------------------------------------------
    # Run evaluation
    # --------------------------------------------------------

    evaluation_results = []

    print(
        "\nRunning retrieval evaluation..."
    )

    for number, item in enumerate(
        questions,
        start=1
    ):

        question = item[
            "question"
        ]

        expected_articles = item[
            "expected_articles"
        ]

        print(
            f"\n[{number}/{len(questions)}] "
            f"{question}"
        )

        result = evaluate_question(
            question=question,
            expected_articles=
                expected_articles,
            chunks=chunks,
            embeddings=embeddings,
            model=model
        )

        evaluation_results.append(
            result
        )

        print(
            "\nExpected:"
        )

        for article in expected_articles:

            print(
                f"  ✓ {article}"
            )

        print(
            "\nRetrieved:"
        )

        for article in result[
            "retrieved_articles"
        ]:

            print(
                f"  → {article}"
            )

        print(
            f"\nPrecision: "
            f"{result['precision']:.4f}"
        )

        print(
            f"Recall: "
            f"{result['recall']:.4f}"
        )

        print(
            f"F1: "
            f"{result['f1']:.4f}"
        )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    if evaluation_results:

        average_precision = sum(
            result["precision"]
            for result in evaluation_results
        ) / len(evaluation_results)

        average_recall = sum(
            result["recall"]
            for result in evaluation_results
        ) / len(evaluation_results)

        average_f1 = sum(
            result["f1"]
            for result in evaluation_results
        ) / len(evaluation_results)

    else:

        average_precision = 0.0
        average_recall = 0.0
        average_f1 = 0.0

    # --------------------------------------------------------
    # Display overall metrics
    # --------------------------------------------------------

    print("\n")

    print("=" * 70)
    print("OVERALL EVALUATION")
    print("=" * 70)

    print(
        f"\nAverage Precision: "
        f"{average_precision:.4f}"
    )

    print(
        f"Average Recall: "
        f"{average_recall:.4f}"
    )

    print(
        f"Average F1 Score: "
        f"{average_f1:.4f}"
    )

    # --------------------------------------------------------
    # Save evaluation
    # --------------------------------------------------------

    save_results(
        evaluation_results=
            evaluation_results,

        average_precision=
            average_precision,

        average_recall=
            average_recall,

        average_f1=
            average_f1
    )

    print("\n")

    print("=" * 70)
    print(
        "STAGE 23 RETRIEVAL EVALUATION COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":

    main()