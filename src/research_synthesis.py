import re

import ollama


LLM_MODEL = "llama3.2"


def clean_text(text):
    """Normalize whitespace and remove unwanted formatting."""
    if not text:
        return ""

    text = str(text)

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def remove_section_heading(text):
    """
    Remove common Wikimedia section headings that can
    accidentally become attached to extracted evidence.
    """

    if not text:
        return ""

    text = clean_text(text)

    headings = [
        "airlines and destinations",
        "statistics",
        "facilities",
        "history",
        "accidents and incidents",
        "references",
        "see also",
        "external links",
    ]

    for heading in headings:

        pattern = rf"^{re.escape(heading)}\s+"

        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE
        )

    return text.strip()


def split_sentences(text):
    """Split source text into simple sentences."""

    text = clean_text(text)

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    cleaned_sentences = []

    for sentence in sentences:

        sentence = remove_section_heading(
            sentence
        )

        if sentence:
            cleaned_sentences.append(
                sentence
            )

    return cleaned_sentences


def find_matching_source_title(
    question,
    results
):
    """
    Find the source article most closely related
    to the entity mentioned in the question.
    """

    question_lower = question.lower()

    # -------------------------------------------------
    # Direct title matching
    # -------------------------------------------------

    for result in results:

        title = clean_text(
            result.get("title", "")
        )

        if not title:
            continue

        if title.lower() in question_lower:
            return result

    # -------------------------------------------------
    # Entity extraction
    # -------------------------------------------------

    patterns = [
        r"what airport serves (.+?)\??$",
        r"what is the iata code of (.+?)\??$",
        r"does (.+?) have scheduled flights\??$",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            question_lower
        )

        if not match:
            continue

        entity = match.group(1).strip()

        for result in results:

            title = clean_text(
                result.get("title", "")
            )

            if entity in title.lower():
                return result

    # -------------------------------------------------
    # Title overlap fallback
    # -------------------------------------------------

    question_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            question_lower
        )
    )

    best_result = None
    best_score = 0

    for result in results:

        title = clean_text(
            result.get("title", "")
        )

        if not title:
            continue

        title_words = set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                title.lower()
            )
        )

        score = len(
            question_words.intersection(
                title_words
            )
        )

        if score > best_score:

            best_score = score
            best_result = result

    return best_result


def extract_airport_serving_answer(
    question,
    results
):
    """
    Handle questions such as:

    What airport serves Juruena?
    What airport serves Barra do Garças?
    """

    match = re.search(
        r"what airport serves (.+?)\??$",
        question.lower().strip()
    )

    if not match:
        return None

    location = match.group(1).strip()

    for result in results:

        title = clean_text(
            result.get("title", "")
        )

        text = clean_text(
            result.get("text", "")
        )

        if (
            location not in text.lower()
            and location not in title.lower()
        ):
            continue

        sentences = split_sentences(
            text
        )

        for sentence in sentences:

            sentence_lower = (
                sentence.lower()
            )

            if (
                "is the airport serving"
                in sentence_lower
                and location
                in sentence_lower
            ):

                return {
                    "answer": sentence,
                    "source": title
                }

    return None


def extract_iata_answer(
    question,
    results
):
    """
    Handle questions such as:

    What is the IATA code of Juruena Airport?
    What is the IATA code of Barra do Garças Airport?
    """

    match = re.search(
        r"what is the iata code of (.+?)\??$",
        question.lower().strip()
    )

    if not match:
        return None

    entity = match.group(1).strip()

    source = find_matching_source_title(
        question,
        results
    )

    if not source:
        return None

    title = clean_text(
        source.get("title", "")
    )

    text = clean_text(
        source.get("text", "")
    )

    if (
        entity not in title.lower()
        and entity not in text.lower()
    ):
        return None

    iata_match = re.search(
        r"\bIATA:\s*([A-Z0-9]+)",
        text,
        re.IGNORECASE
    )

    if not iata_match:
        return None

    iata_code = (
        iata_match.group(1)
        .upper()
    )

    return {
        "answer": (
            f"The IATA code of "
            f"{title} is {iata_code}."
        ),
        "source": title
    }


def extract_scheduled_flights_answer(
    question,
    results
):
    """
    Handle questions such as:

    Does Juruena Airport have scheduled flights?
    """

    match = re.search(
        r"does (.+?) have scheduled flights\??$",
        question.lower().strip()
    )

    if not match:
        return None

    entity = match.group(1).strip()

    source = find_matching_source_title(
        question,
        results
    )

    if not source:
        return None

    title = clean_text(
        source.get("title", "")
    )

    text = clean_text(
        source.get("text", "")
    )

    if (
        entity not in title.lower()
        and entity not in text.lower()
    ):
        return None

    sentences = split_sentences(
        text
    )

    for sentence in sentences:

        sentence_lower = (
            sentence.lower()
        )

        if (
            "no scheduled flights operate"
            in sentence_lower
        ):

            return {
                "answer": (
                    f"No. {sentence}"
                ),
                "source": title
            }

        if (
            "scheduled flights operate"
            in sentence_lower
        ):

            return {
                "answer": sentence,
                "source": title
            }

    return None


def extract_brazil_airports_answer(
    question,
    results
):
    """
    Handle:

    What airports are located in Brazil?
    """

    question_lower = (
        question.lower()
    )

    if (
        "what airports are located in brazil"
        not in question_lower
    ):
        return None

    airports = []

    for result in results:

        title = clean_text(
            result.get("title", "")
        )

        text = clean_text(
            result.get("text", "")
        )

        if not title or not text:
            continue

        text_lower = text.lower()

        if "airport" not in title.lower():
            continue

        if "brazil" not in text_lower:
            continue

        if (
            "is the airport serving"
            not in text_lower
        ):
            continue

        if title not in airports:
            airports.append(title)

    if not airports:
        return None

    lines = [
        "The retrieved Wikimedia sources "
        "identify these airports in Brazil:"
    ]

    for airport in airports:
        lines.append(
            f"- {airport}"
        )

    return {
        "answer": "\n".join(lines),
        "sources": airports
    }


def extract_direct_answer(
    question,
    results
):
    """
    Attempt deterministic evidence extraction
    before using the LLM.
    """

    if not question or not results:
        return None

    direct_answer = (
        extract_airport_serving_answer(
            question,
            results
        )
    )

    if direct_answer:
        return direct_answer

    direct_answer = (
        extract_iata_answer(
            question,
            results
        )
    )

    if direct_answer:
        return direct_answer

    direct_answer = (
        extract_scheduled_flights_answer(
            question,
            results
        )
    )

    if direct_answer:
        return direct_answer

    direct_answer = (
        extract_brazil_airports_answer(
            question,
            results
        )
    )

    if direct_answer:
        return direct_answer

    return None


def build_research_context(
    results
):
    """Build a clean evidence context."""

    context_parts = []

    for number, result in enumerate(
        results,
        start=1
    ):

        title = result.get(
            "title",
            "Unknown Article"
        )

        text = result.get(
            "text",
            ""
        )

        context_parts.append(
            f"""
SOURCE {number}

Article:
{title}

Evidence:
{text}
"""
        )

    return "\n".join(
        context_parts
    )


def build_research_prompt(
    question,
    context
):
    """Build the fallback LLM prompt."""

    return f"""
You are a Wikimedia research assistant.

Answer the user's question using ONLY the
provided Wikimedia source evidence.

USER QUESTION:
{question}

SOURCE EVIDENCE:
{context}

STRICT RULES:

1. Use ONLY information explicitly present
   in the source evidence.

2. Do NOT use outside knowledge.

3. Do NOT invent facts.

4. Answer the exact question asked.

5. Do not include unrelated information.

6. Preserve negative statements exactly.

7. Do not claim information is unavailable
   when it is explicitly present.

8. Keep the answer concise.

OUTPUT FORMAT:

Research Answer:
<answer>

Sources Used:
- Source X: <article title>

Now answer the user's question.
"""


def synthesize_research(
    question,
    results,
    model=LLM_MODEL
):
    """
    Generate an evidence-grounded answer.

    Deterministic extraction is preferred for
    straightforward factual questions.
    Ollama is used as a fallback for complex questions.
    """

    if not question:
        return (
            "No research question "
            "was provided."
        )

    if not results:
        return (
            "No relevant source evidence "
            "was found."
        )

    # -------------------------------------------------
    # Deterministic answer
    # -------------------------------------------------

    direct_answer = extract_direct_answer(
        question,
        results
    )

    if direct_answer:

        answer = direct_answer.get(
            "answer",
            ""
        )

        sources = []

        if direct_answer.get("source"):
            sources.append(
                direct_answer["source"]
            )

        sources.extend(
            direct_answer.get(
                "sources",
                []
            )
        )

        sources = list(
            dict.fromkeys(sources)
        )

        if sources:

            answer += (
                "\n\nSources Used:\n"
            )

            for source in sources:

                answer += (
                    f"- {source}\n"
                )

        return answer

    # -------------------------------------------------
    # LLM fallback
    # -------------------------------------------------

    context = build_research_context(
        results
    )

    prompt = build_research_prompt(
        question,
        context
    )

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response[
        "message"
    ]["content"]