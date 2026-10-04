import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# STOP WORDS
# ============================================================

STOP_WORDS = {
    "what",
    "which",
    "where",
    "when",
    "who",
    "why",
    "how",
    "is",
    "are",
    "was",
    "were",
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "from",
    "does",
    "do",
    "did",
    "has",
    "have",
    "had",
    "and",
    "or",
    "with",
    "about",
    "located",
    "there",
    "any",
    "all"
}


# ============================================================
# KEYWORD EXTRACTION
# ============================================================

def extract_keywords(question):

    words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        question.lower()
    )

    keywords = []

    for word in words:

        if word in STOP_WORDS:
            continue

        if len(word) < 2:
            continue

        if word not in keywords:
            keywords.append(word)

    return keywords


# ============================================================
# TOPIC DETECTION
# ============================================================

def detect_topic_terms(question):

    question_lower = question.lower()

    topic_terms = []

    topic_groups = {
        "airport": [
            "airport",
            "airports",
            "airstrip",
            "airstrips"
        ],

        "airline": [
            "airline",
            "airlines"
        ],

        "museum": [
            "museum",
            "museums"
        ],

        "university": [
            "university",
            "universities"
        ],

        "city": [
            "city",
            "cities"
        ],

        "river": [
            "river",
            "rivers"
        ],

        "mountain": [
            "mountain",
            "mountains"
        ],

        "lake": [
            "lake",
            "lakes"
        ],

        "country": [
            "country",
            "countries"
        ]
    }

    for terms in topic_groups.values():

        for term in terms:

            if term in question_lower:

                topic_terms.append(
                    term
                )

    return list(
        dict.fromkeys(
            topic_terms
        )
    )


# ============================================================
# LOCATION DETECTION
# ============================================================

def detect_location_terms(question):

    keywords = extract_keywords(
        question
    )

    topic_terms = detect_topic_terms(
        question
    )

    location_terms = []

    for keyword in keywords:

        if keyword in topic_terms:
            continue

        location_terms.append(
            keyword
        )

    return location_terms


# ============================================================
# SEARCHABLE TEXT
# ============================================================

def build_searchable_text(result):

    title = str(
        result.get(
            "title",
            ""
        )
    )

    text = str(
        result.get(
            "text",
            ""
        )
    )

    return (
        f"{title} {text}"
    ).lower()


# ============================================================
# KEYWORD SCORE
# ============================================================

def calculate_keyword_score(
    question,
    result
):

    keywords = extract_keywords(
        question
    )

    if not keywords:
        return 0.0

    searchable_text = build_searchable_text(
        result
    )

    matched = 0

    for keyword in keywords:

        if keyword in searchable_text:

            matched += 1

    return (
        matched / len(keywords)
    )


# ============================================================
# TOPIC SCORE
# ============================================================

def calculate_topic_score(
    question,
    result
):

    topic_terms = detect_topic_terms(
        question
    )

    if not topic_terms:
        return 0.0

    title = str(
        result.get(
            "title",
            ""
        )
    ).lower()

    matched = 0

    for term in topic_terms:

        if term in title:

            matched += 1

    return (
        matched / len(topic_terms)
    )


# ============================================================
# LOCATION SCORE
# ============================================================

def calculate_location_score(
    question,
    result
):

    location_terms = detect_location_terms(
        question
    )

    if not location_terms:
        return 0.0

    searchable_text = build_searchable_text(
        result
    )

    matched = 0

    for term in location_terms:

        if term in searchable_text:

            matched += 1

    return (
        matched / len(location_terms)
    )


# ============================================================
# RETRIEVE RESEARCH ARTICLES
# ============================================================

def retrieve_research_articles(
    question,
    chunks,
    embeddings,
    model,
    top_k=5,
    min_articles=2
):

    if not question.strip():

        return []

    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    # --------------------------------------------------------
    # Query understanding
    # --------------------------------------------------------

    topic_terms = detect_topic_terms(
        question
    )

    location_terms = detect_location_terms(
        question
    )

    # --------------------------------------------------------
    # Candidate retrieval
    # --------------------------------------------------------

    candidate_count = max(
        top_k * 20,
        200
    )

    candidate_indices = np.argsort(
        similarities
    )[::-1][:candidate_count]

    candidates = []

    for index in candidate_indices:

        chunk = chunks[index]

        title = str(
            chunk.get(
                "title",
                ""
            )
        )

        searchable_text = build_searchable_text(
            chunk
        )

        title_lower = title.lower()

        # ----------------------------------------------------
        # Topic filtering
        # ----------------------------------------------------

        if topic_terms:

            topic_found = any(
                term in title_lower
                for term in topic_terms
            )

            if not topic_found:
                continue

        # ----------------------------------------------------
        # Location filtering
        # ----------------------------------------------------

        if location_terms:

            location_found = any(
                term in searchable_text
                for term in location_terms
            )

            if not location_found:
                continue

        # ----------------------------------------------------
        # Scores
        # ----------------------------------------------------

        semantic_score = float(
            similarities[index]
        )

        keyword_score = (
            calculate_keyword_score(
                question,
                chunk
            )
        )

        topic_score = (
            calculate_topic_score(
                question,
                chunk
            )
        )

        location_score = (
            calculate_location_score(
                question,
                chunk
            )
        )

        # ----------------------------------------------------
        # Combined score
        # ----------------------------------------------------

        research_score = (
            semantic_score * 0.50
            + keyword_score * 0.10
            + topic_score * 0.15
            + location_score * 0.25
        )

        candidates.append(
            {
                "index": int(index),

                "title": title,

                "text": chunk.get(
                    "text",
                    ""
                ),

                "similarity":
                    semantic_score,

                "keyword_score":
                    keyword_score,

                "topic_score":
                    topic_score,

                "location_score":
                    location_score,

                "research_score":
                    research_score,

                "article_id":
                    chunk.get(
                        "article_id"
                    ),

                "chunk_id":
                    chunk.get(
                        "chunk_id"
                    ),

                "url":
                    chunk.get(
                        "url"
                    )
            }
        )

    # --------------------------------------------------------
    # Reranking
    # --------------------------------------------------------

    candidates.sort(
        key=lambda item:
            item["research_score"],
        reverse=True
    )

    # --------------------------------------------------------
    # Deduplicate articles
    # --------------------------------------------------------

    selected = []

    seen_articles = set()

    for candidate in candidates:

        article_id = candidate.get(
            "article_id"
        )

        if article_id in seen_articles:
            continue

        seen_articles.add(
            article_id
        )

        selected.append(
            candidate
        )

        if len(selected) >= top_k:
            break

    # --------------------------------------------------------
    # If minimum source requirement is not met,
    # return whatever relevant sources exist.
    # --------------------------------------------------------

    if len(selected) < min_articles:

        return selected

    return selected


# ============================================================
# BUILD RESEARCH CONTEXT
# ============================================================

def build_research_context(
    results
):

    context_parts = []

    for number, result in enumerate(
        results,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {number}

Article:
{result.get('title', 'Unknown Article')}

Evidence:
{result.get('text', '')}
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# RESEARCH STATISTICS
# ============================================================

def get_research_statistics(
    results
):

    if not results:

        return {
            "articles": 0,
            "best_match": 0.0,
            "average_match": 0.0
        }

    similarities = [
        result.get(
            "similarity",
            0.0
        )
        for result in results
    ]

    return {
        "articles": len(results),

        "best_match": max(
            similarities
        ),

        "average_match": (
            sum(similarities)
            / len(similarities)
        )
    }