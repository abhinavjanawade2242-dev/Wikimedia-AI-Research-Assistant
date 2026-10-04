import json
import sys
from pathlib import Path

# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research import retrieve_research_articles
from src.research_synthesis import synthesize_research
from src.evidence_verifier import verify_research_answer


QUESTIONS_PATH = PROJECT_ROOT / "scripts" / "evaluation_questions.json"

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks_25000.jsonl"
)

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "embeddings"
    / "chunk_embeddings_25000.npy"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "research_evaluation.json"
)


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def calculate_retrieval_recall(
    expected_articles,
    retrieved_articles
):
    if not expected_articles:
        return 0.0

    expected = {
        article.lower()
        for article in expected_articles
    }

    retrieved = {
        article.lower()
        for article in retrieved_articles
    }

    matched = expected.intersection(retrieved)

    return len(matched) / len(expected)


def parse_verification_counts(verification_text):
    text = verification_text.lower()

    supported = 0
    partially_supported = 0
    unsupported = 0

    lines = text.splitlines()

    for line in lines:
        line = line.strip()

        if line.startswith("- supported claims:"):
            try:
                supported = int(
                    line.split(":", 1)[1].strip()
                )
            except ValueError:
                supported = 0

        elif line.startswith("- partially supported claims:"):
            try:
                partially_supported = int(
                    line.split(":", 1)[1].strip()
                )
            except ValueError:
                partially_supported = 0

        elif line.startswith("- unsupported claims:"):
            try:
                unsupported = int(
                    line.split(":", 1)[1].strip()
                )
            except ValueError:
                unsupported = 0

    return (
        supported,
        partially_supported,
        unsupported
    )


def calculate_verification_score(
    supported,
    partially_supported,
    unsupported
):
    total = (
        supported
        + partially_supported
        + unsupported
    )

    if total == 0:
        return 0.0

    weighted_score = (
        supported
        + (0.5 * partially_supported)
    )

    return weighted_score / total


def main():
    print("=" * 70)
    print("WIKIMEDIA AI RESEARCH ASSISTANT")
    print("END-TO-END RESEARCH EVALUATION")
    print("=" * 70)

    print("\nLoading evaluation questions...")

    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        questions = json.load(file)

    print(f"Evaluation questions: {len(questions)}")

    print("\nLoading chunks...")

    chunks = load_jsonl(CHUNKS_PATH)

    print(f"Loaded chunks: {len(chunks):,}")

    print("\nLoading embeddings...")

    import numpy as np

    embeddings = np.load(EMBEDDINGS_PATH)

    print(f"Embedding shape: {embeddings.shape}")

    print("\nLoading Sentence Transformer...")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print("Model loaded successfully.")

    evaluation_results = []

    retrieval_recalls = []
    verification_scores = []
    overall_qualities = []

    for index, item in enumerate(
        questions,
        start=1
    ):
        question = item["question"]

        expected_articles = item.get(
            "expected_articles",
            []
        )

        print("\n" + "-" * 70)
        print(
            f"Question {index}/{len(questions)}"
        )
        print(f"Question: {question}")

        print("\nRetrieving research sources...")

        results = retrieve_research_articles(
            question=question,
            chunks=chunks,
            embeddings=embeddings,
            model=model,
            top_k=5,
            min_articles=2
        )

        retrieved_articles = [
            result.get("title", "")
            for result in results
        ]

        print("Retrieved articles:")

        for article in retrieved_articles:
            print(f"  - {article}")

        retrieval_recall = calculate_retrieval_recall(
            expected_articles,
            retrieved_articles
        )

        print(
            f"Retrieval recall: "
            f"{retrieval_recall:.2%}"
        )

        print("\nGenerating research answer...")

        research_answer = synthesize_research(
            question=question,
            results=results
        )

        print("\nResearch answer:")
        print(research_answer)

        print("\nVerifying evidence...")

        verification = verify_research_answer(
            answer=research_answer,
            results=results
        )

        print("\nVerification result:")
        print(verification)

        (
            supported,
            partially_supported,
            unsupported
        ) = parse_verification_counts(
            verification
        )

        verification_score = calculate_verification_score(
            supported,
            partially_supported,
            unsupported
        )

        overall_quality = (
            retrieval_recall * 0.5
            + verification_score * 0.5
        )

        print(
            f"\nVerification score: "
            f"{verification_score:.2%}"
        )

        print(
            f"Overall research quality: "
            f"{overall_quality:.2%}"
        )

        evaluation_results.append(
            {
                "question": question,
                "expected_articles": expected_articles,
                "retrieved_articles": retrieved_articles,
                "retrieval_recall": retrieval_recall,
                "research_answer": research_answer,
                "verification": verification,
                "supported_claims": supported,
                "partially_supported_claims": partially_supported,
                "unsupported_claims": unsupported,
                "verification_score": verification_score,
                "overall_quality": overall_quality
            }
        )

        retrieval_recalls.append(
            retrieval_recall
        )

        verification_scores.append(
            verification_score
        )

        overall_qualities.append(
            overall_quality
        )

    average_retrieval_recall = (
        sum(retrieval_recalls)
        / len(retrieval_recalls)
        if retrieval_recalls
        else 0.0
    )

    average_verification_score = (
        sum(verification_scores)
        / len(verification_scores)
        if verification_scores
        else 0.0
    )

    average_overall_quality = (
        sum(overall_qualities)
        / len(overall_qualities)
        if overall_qualities
        else 0.0
    )

    final_evaluation = {
        "total_questions": len(questions),
        "average_retrieval_recall": average_retrieval_recall,
        "average_verification_score": average_verification_score,
        "average_overall_quality": average_overall_quality,
        "questions": evaluation_results
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            final_evaluation,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    print(
        f"Average Retrieval Recall: "
        f"{average_retrieval_recall:.2%}"
    )

    print(
        f"Average Verification Score: "
        f"{average_verification_score:.2%}"
    )

    print(
        f"Average Overall Research Quality: "
        f"{average_overall_quality:.2%}"
    )

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()