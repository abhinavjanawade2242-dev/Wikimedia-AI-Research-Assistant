import ollama


LLM_MODEL = "llama3.2"


def build_comparison_context(
    results
):
    """
    Build a structured context for
    comparing multiple research sources.
    """

    context_parts = []

    for number, result in enumerate(
        results,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {number}

Article:
{result.get("title", "Unknown Article")}

Evidence:
{result.get("text", "")}
"""
        )

    return "\n".join(
        context_parts
    )


def build_comparison_prompt(
    question,
    context
):
    """
    Build a strict prompt for source
    comparison.
    """

    return f"""
You are a research comparison assistant.

Your task is to compare information from
multiple Wikimedia sources.

RESEARCH QUESTION:
{question}

SOURCES:
{context}

STRICT RULES:

1. Use ONLY information explicitly
   present in the provided sources.

2. Do not use outside knowledge.

3. Do not invent facts.

4. Do not assume two statements are
   equivalent unless the evidence
   supports that conclusion.

5. Identify information that multiple
   sources agree on.

6. Identify meaningful differences
   between sources.

7. Identify possible contradictions
   when two sources provide conflicting
   information.

8. If there are no contradictions,
   explicitly say so.

9. Always identify the source numbers
   supporting each finding.

10. If the available evidence is
    insufficient to determine whether
    two statements conflict, say so.

11. Keep the comparison concise and
    research-focused.

FORMAT:

## Agreements

- <agreement> [Source X, Source Y]

## Differences

- <difference> [Source X, Source Y]

## Conflicts

- <conflict> [Source X, Source Y]

If there are no conflicts, write:

- No direct conflicts were identified
  in the retrieved sources.

Now compare the sources.
"""


def compare_research_sources(
    question,
    results,
    model=LLM_MODEL
):
    """
    Compare retrieved research sources
    using the local LLM.
    """

    if not results:

        return (
            "No sources were available "
            "for comparison."
        )

    context = build_comparison_context(
        results
    )

    prompt = build_comparison_prompt(
        question=question,
        context=context
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

    return (
        response["message"]["content"]
    )