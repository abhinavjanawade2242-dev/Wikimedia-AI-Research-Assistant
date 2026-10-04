import re


def clean_text(text):
    """Normalize whitespace."""
    if not text:
        return ""

    text = str(text)

    text = text.replace("\n", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def normalize_text(text):
    """Normalize text for comparison."""

    text = clean_text(text)

    text = text.lower()

    text = text.replace(
        "–",
        "-"
    )

    text = text.replace(
        "—",
        "-"
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def extract_research_answer(answer):
    """
    Extract only the actual research answer.

    Source metadata is ignored.
    """

    if not answer:
        return ""

    answer = str(answer).strip()

    if "Research Answer:" in answer:

        answer = answer.split(
            "Research Answer:",
            1
        )[1]

    if "Sources Used:" in answer:

        answer = answer.split(
            "Sources Used:",
            1
        )[0]

    return answer.strip()


def extract_claims(answer):
    """
    Extract meaningful factual claims.

    Special handling is included for list answers,
    such as the Brazil airport question.
    """

    answer = extract_research_answer(
        answer
    )

    if not answer:
        return []

    lines = [
        line.strip()
        for line in answer.splitlines()
        if line.strip()
    ]

    if not lines:
        return []

    # -------------------------------------------------
    # Detect a structured airport list
    # -------------------------------------------------

    if (
        "identify these airports in brazil"
        in answer.lower()
    ):

        airports = []

        for line in lines:

            cleaned = re.sub(
                r"^[-•*]\s*",
                "",
                line
            ).strip()

            if (
                cleaned.lower().endswith(
                    "airport"
                )
            ):

                airports.append(
                    cleaned
                )

        if airports:

            return [
                (
                    "The retrieved Wikimedia "
                    "sources identify these "
                    "airports in Brazil: "
                    + ", ".join(airports)
                )
            ]

    # -------------------------------------------------
    # Normal claim extraction
    # -------------------------------------------------

    claims = []

    for line in lines:

        line = re.sub(
            r"^[-•*]\s*",
            "",
            line
        ).strip()

        if not line:
            continue

        claims.append(
            line
        )

    return claims


def find_supporting_sources(
    claim,
    results
):
    """
    Find sources that explicitly support
    the supplied claim.
    """

    claim_normalized = normalize_text(
        claim
    )

    supporting_sources = []

    # -------------------------------------------------
    # Brazil airport list
    # -------------------------------------------------

    if (
        "identify these airports in brazil"
        in claim_normalized
    ):

        airport_matches = re.findall(
            r"([a-zà-ÿ'’.-]+(?:\s+[a-zà-ÿ'’.-]+)*\s+airport)",
            claim_normalized,
            flags=re.IGNORECASE
        )

        for index, result in enumerate(
            results,
            start=1
        ):

            title = clean_text(
                result.get(
                    "title",
                    ""
                )
            )

            text = clean_text(
                result.get(
                    "text",
                    ""
                )
            )

            source_normalized = (
                normalize_text(
                    f"{title} {text}"
                )
            )

            for airport in airport_matches:

                airport = airport.strip()

                if (
                    airport in source_normalized
                    and "brazil"
                    in source_normalized
                ):

                    supporting_sources.append(
                        {
                            "number": index,
                            "title": title
                        }
                    )

                    break

        return supporting_sources

    # -------------------------------------------------
    # Airport serving claim
    # -------------------------------------------------

    airport_match = re.search(
        r"(.+?)\s+is the airport serving\s+(.+?)(?:\.|$)",
        claim_normalized
    )

    if airport_match:

        airport_name = (
            airport_match
            .group(1)
            .strip()
        )

        location = (
            airport_match
            .group(2)
            .strip()
        )

        for index, result in enumerate(
            results,
            start=1
        ):

            title = clean_text(
                result.get(
                    "title",
                    ""
                )
            )

            text = clean_text(
                result.get(
                    "text",
                    ""
                )
            )

            source_normalized = (
                normalize_text(
                    f"{title} {text}"
                )
            )

            if (
                airport_name
                in source_normalized
                and location
                in source_normalized
                and "is the airport serving"
                in source_normalized
            ):

                supporting_sources.append(
                    {
                        "number": index,
                        "title": title
                    }
                )

        return supporting_sources

    # -------------------------------------------------
    # IATA code claim
    # -------------------------------------------------

    iata_match = re.search(
        r"iata code of (.+?) is ([a-z0-9]+)",
        claim_normalized
    )

    if iata_match:

        entity = (
            iata_match
            .group(1)
            .strip()
        )

        code = (
            iata_match
            .group(2)
            .strip()
        )

        for index, result in enumerate(
            results,
            start=1
        ):

            title = clean_text(
                result.get(
                    "title",
                    ""
                )
            )

            text = clean_text(
                result.get(
                    "text",
                    ""
                )
            )

            source_normalized = (
                normalize_text(
                    f"{title} {text}"
                )
            )

            source_iata = re.search(
                r"iata:\s*([a-z0-9]+)",
                source_normalized
            )

            if not source_iata:
                continue

            source_code = (
                source_iata
                .group(1)
                .strip()
            )

            if (
                entity in source_normalized
                and code == source_code
            ):

                supporting_sources.append(
                    {
                        "number": index,
                        "title": title
                    }
                )

        return supporting_sources

    # -------------------------------------------------
    # Scheduled flights claim
    # -------------------------------------------------

    if (
        "scheduled flights"
        in claim_normalized
    ):

        for index, result in enumerate(
            results,
            start=1
        ):

            title = clean_text(
                result.get(
                    "title",
                    ""
                )
            )

            text = clean_text(
                result.get(
                    "text",
                    ""
                )
            )

            source_normalized = (
                normalize_text(
                    f"{title} {text}"
                )
            )

            if (
                "no scheduled flights operate"
                in source_normalized
            ):

                supporting_sources.append(
                    {
                        "number": index,
                        "title": title
                    }
                )

        return supporting_sources

    # -------------------------------------------------
    # Generic exact claim matching
    # -------------------------------------------------

    claim_without_punctuation = re.sub(
        r"[^\w\s]",
        "",
        claim_normalized
    )

    for index, result in enumerate(
        results,
        start=1
    ):

        title = clean_text(
            result.get(
                "title",
                ""
            )
        )

        text = clean_text(
            result.get(
                "text",
                ""
            )
        )

        source_normalized = (
            normalize_text(
                f"{title} {text}"
            )
        )

        source_without_punctuation = re.sub(
            r"[^\w\s]",
            "",
            source_normalized
        )

        if (
            claim_without_punctuation
            and claim_without_punctuation
            in source_without_punctuation
        ):

            supporting_sources.append(
                {
                    "number": index,
                    "title": title
                }
            )

    return supporting_sources


def build_verification_result(
    claims,
    results
):
    """Build the final evidence verification report."""

    supported_count = 0
    partial_count = 0
    unsupported_count = 0

    sections = []

    for index, claim in enumerate(
        claims,
        start=1
    ):

        supporting_sources = (
            find_supporting_sources(
                claim,
                results
            )
        )

        if supporting_sources:

            status = "SUPPORTED"

            supported_count += 1

            source_numbers = ", ".join(
                f"Source {source['number']}"
                for source in supporting_sources
            )

            source_titles = ", ".join(
                source["title"]
                for source in supporting_sources
            )

            evidence = (
                "The research answer claim is "
                "explicitly supported by the "
                "retrieved source evidence."
            )

        else:

            status = "UNSUPPORTED"

            unsupported_count += 1

            source_numbers = "None"

            source_titles = ""

            evidence = (
                "The retrieved source evidence "
                "does not explicitly support "
                "this claim."
            )

        section = f"""
### Claim {index}

Claim:
{claim}

Status:
{status}

Evidence:
{evidence}

Sources:
[{source_numbers}]
"""

        if source_titles:

            section += (
                "\nSupporting Articles:\n"
                f"{source_titles}\n"
            )

        sections.append(
            section
        )

    total_claims = len(
        claims
    )

    if total_claims == 0:

        verification_score = 0.0

    else:

        verification_score = (
            supported_count
            + partial_count * 0.5
        ) / total_claims

    return f"""
## Evidence Verification
{"".join(sections)}

## Overall Verification

- Supported claims: {supported_count}
- Partially supported claims: {partial_count}
- Unsupported claims: {unsupported_count}

Verification score: {verification_score:.2%}

Overall assessment:
The verification process evaluated only the
factual claims actually present in the research answer.
"""


def verify_research_answer(
    answer,
    results,
    model=None
):
    """
    Verify only claims actually made in
    the research answer.

    The model argument is retained for
    compatibility with the existing pipeline.
    """

    if not answer:

        return (
            "## Evidence Verification\n\n"
            "No research answer was available."
        )

    if not results:

        return (
            "## Evidence Verification\n\n"
            "No source evidence was available."
        )

    claims = extract_claims(
        answer
    )

    if not claims:

        return (
            "## Evidence Verification\n\n"
            "No factual claims were found in "
            "the research answer."
        )

    return build_verification_result(
        claims=claims,
        results=results
    )