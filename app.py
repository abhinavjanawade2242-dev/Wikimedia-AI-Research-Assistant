import json
import re

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ollama

from src.research import (
    retrieve_research_articles,
    get_research_statistics
)

from src.research_synthesis import synthesize_research

from src.research_report import generate_research_report

from src.research_history import (
    initialize_research_history,
    save_research,
    get_research_history,
    clear_research_history
)

from src.research_comparison import compare_research_sources

from src.evidence_verifier import verify_research_answer


# ============================================================
# CONFIGURATION
# ============================================================

CHUNKS_PATH = "data/processed/chunks_25000.jsonl"

EMBEDDINGS_PATH = (
    "data/processed/embeddings/"
    "chunk_embeddings_25000.npy"
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

LLM_MODEL = "llama3.2"

TOP_K = 5


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Wikimedia AI Research Assistant",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    .source-card {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.3);
        margin-bottom: 12px;
    }

    .confidence-high {
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #2e7d32;
    }

    .confidence-medium {
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #f9a825;
    }

    .confidence-low {
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #c62828;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_data
def load_chunks():

    chunks = []

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

    return chunks


@st.cache_data
def load_embeddings():

    return np.load(
        EMBEDDINGS_PATH
    )


@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# ============================================================
# SESSION STATE
# ============================================================

if "conversation" not in st.session_state:

    st.session_state.conversation = []


if "last_question" not in st.session_state:

    st.session_state.last_question = ""


if "last_results" not in st.session_state:

    st.session_state.last_results = []


if "summary_results" not in st.session_state:

    st.session_state.summary_results = []


initialize_research_history(
    st.session_state
)


# ============================================================
# BASIC SEARCH
# ============================================================

def retrieve_chunks(
    question,
    chunks,
    embeddings,
    model,
    top_k=TOP_K
):

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        chunk = chunks[index]

        results.append(
            {
                "title": chunk.get(
                    "title",
                    "Unknown Article"
                ),

                "text": chunk.get(
                    "text",
                    ""
                ),

                "similarity": float(
                    similarities[index]
                ),

                "article_id": chunk.get(
                    "article_id"
                ),

                "chunk_id": chunk.get(
                    "chunk_id"
                ),

                "url": chunk.get(
                    "url"
                ),

                "embedding_index": int(
                    index
                )
            }
        )

    return results


# ============================================================
# CONVERSATION CONTEXT
# ============================================================

def build_conversation_context():

    if not st.session_state.conversation:

        return ""

    context_parts = []

    for item in st.session_state.conversation[-5:]:

        context_parts.append(
            f"User: {item['question']}\n"
            f"Assistant: {item['answer']}"
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# FOLLOW-UP QUESTION
# ============================================================

def rewrite_followup_question(question):

    # IMPORTANT:
    # Do not call Ollama for a standalone first question.
    # This avoids an unnecessary LLM call.

    if not st.session_state.conversation:

        return question

    context = build_conversation_context()

    prompt = f"""
You are a question rewriting assistant.

Previous conversation:

{context}

New user question:

{question}

If the new question is a follow-up question,
rewrite it into a complete standalone question.

If it is already standalone, return it unchanged.

Return ONLY the rewritten question.
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response[
        "message"
    ][
        "content"
    ].strip()


# ============================================================
# RAG CONTEXT
# ============================================================

def build_rag_context(results):

    context_parts = []

    for number, result in enumerate(
        results,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {number}

Article:
{result['title']}

Evidence:
{result['text']}
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# DIRECT EVIDENCE EXTRACTION
# ============================================================

def clean_text(text):

    text = re.sub(
        r"\s+",
        " ",
        str(text)
    )

    return text.strip()


def split_sentences(text):

    text = clean_text(text)

    if not text:

        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def find_matching_source_title(
    requested_name,
    results
):

    requested_name = clean_text(
        requested_name
    ).lower()

    requested_name = requested_name.rstrip(
        "?. "
    )

    for number, result in enumerate(
        results,
        start=1
    ):

        title = clean_text(
            result.get(
                "title",
                ""
            )
        )

        if title.lower() == requested_name:

            return number, result

    return None, None


def extract_direct_answer(
    question,
    results
):

    """
    Try to answer simple factual questions directly
    from the retrieved evidence.

    This prevents the LLM from incorrectly denying
    information that is explicitly present.
    """

    if not results:

        return None

    question_clean = clean_text(
        question
    )

    question_lower = question_clean.lower()

    # --------------------------------------------------------
    # Pattern 1:
    # What airport serves Juruena?
    # --------------------------------------------------------

    airport_match = re.search(
        r"what airport serves (.+?)\??$",
        question_lower
    )

    if airport_match:

        location = airport_match.group(
            1
        ).strip()

        location = location.rstrip(
            "?."
        )

        for number, result in enumerate(
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

            pattern = (
                rf"{re.escape(title)}"
                rf".*?is the airport serving "
                rf"{re.escape(location)}"
            )

            if re.search(
                pattern,
                text,
                flags=re.IGNORECASE
            ):

                sentence_matches = [
                    sentence
                    for sentence in split_sentences(text)
                    if (
                        "airport serving" in
                        sentence.lower()
                        and
                        location.lower() in
                        sentence.lower()
                    )
                ]

                if sentence_matches:

                    return (
                        f"Answer:\n"
                        f"{sentence_matches[0]}\n\n"
                        f"Source:\n"
                        f"[Source {number}]"
                    )

    # --------------------------------------------------------
    # Pattern 2:
    # What is the IATA code of Juruena Airport?
    # --------------------------------------------------------

    iata_match = re.search(
        r"what is the iata code of (.+?)\??$",
        question_lower
    )

    if iata_match:

        article_name = iata_match.group(
            1
        ).strip()

        article_name = article_name.rstrip(
            "?."
        )

        source_number, result = (
            find_matching_source_title(
                article_name,
                results
            )
        )

        if result:

            text = clean_text(
                result.get(
                    "text",
                    ""
                )
            )

            code_match = re.search(
                r"IATA:\s*([A-Z0-9-]+)",
                text,
                flags=re.IGNORECASE
            )

            if code_match:

                code = code_match.group(
                    1
                ).upper()

                return (
                    f"Answer:\n"
                    f"The IATA code of "
                    f"{result['title']} is "
                    f"{code}.\n\n"
                    f"Source:\n"
                    f"[Source {source_number}]"
                )

    # --------------------------------------------------------
    # Pattern 3:
    # Does Juruena Airport have scheduled flights?
    # --------------------------------------------------------

    scheduled_match = re.search(
        r"does (.+?) have scheduled flights\??$",
        question_lower
    )

    if scheduled_match:

        article_name = scheduled_match.group(
            1
        ).strip()

        article_name = article_name.rstrip(
            "?."
        )

        source_number, result = (
            find_matching_source_title(
                article_name,
                results
            )
        )

        if result:

            text = clean_text(
                result.get(
                    "text",
                    ""
                )
            )

            sentences = split_sentences(
                text
            )

            for sentence in sentences:

                sentence_lower = sentence.lower()

                if (
                    "scheduled flights" in
                    sentence_lower
                ):

                    return (
                        f"Answer:\n"
                        f"{sentence}\n\n"
                        f"Source:\n"
                        f"[Source {source_number}]"
                    )

    # --------------------------------------------------------
    # Generic exact sentence match
    # --------------------------------------------------------

    question_keywords = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        question_lower
    )

    stop_words = {
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

    meaningful_keywords = [
        word
        for word in question_keywords
        if (
            word not in stop_words
            and len(word) >= 3
        )
    ]

    if meaningful_keywords:

        best_sentence = None
        best_source = None
        best_score = 0

        for number, result in enumerate(
            results,
            start=1
        ):

            sentences = split_sentences(
                result.get(
                    "text",
                    ""
                )
            )

            for sentence in sentences:

                sentence_lower = (
                    sentence.lower()
                )

                matched = sum(
                    1
                    for keyword
                    in meaningful_keywords
                    if keyword in sentence_lower
                )

                score = (
                    matched /
                    len(meaningful_keywords)
                )

                if (
                    score >= 0.70
                    and score > best_score
                ):

                    best_score = score
                    best_sentence = sentence
                    best_source = number

        if best_sentence:

            return (
                f"Answer:\n"
                f"{best_sentence}\n\n"
                f"Source:\n"
                f"[Source {best_source}]"
            )

    return None


# ============================================================
# GROUNDED ANSWER
# ============================================================

def generate_grounded_answer(
    question,
    results
):

    if not results:

        return (
            "I could not find enough relevant "
            "information in the Wikimedia knowledge base."
        )

    # --------------------------------------------------------
    # STEP 1:
    # Try deterministic evidence extraction first.
    # --------------------------------------------------------

    direct_answer = extract_direct_answer(
        question,
        results
    )

    if direct_answer:

        return direct_answer

    # --------------------------------------------------------
    # STEP 2:
    # Use Llama for questions that require synthesis.
    # --------------------------------------------------------

    context = build_rag_context(
        results
    )

    prompt = f"""
You are a factual research assistant.

You must answer the question using ONLY
the Wikimedia evidence provided below.

QUESTION:
{question}

SOURCE EVIDENCE:
{context}

IMPORTANT INSTRUCTIONS:

1. The answer MUST come from the evidence.

2. Read every source carefully before answering.

3. If the evidence explicitly contains the answer,
   state that answer directly.

4. Never claim that information is missing if
   a source explicitly contains it.

5. Do not use outside knowledge.

6. Do not guess.

7. Do not invent facts.

8. Do not contradict explicit statements in
   the source evidence.

9. For a specific fact such as a name, code,
   number, location, date, or status, copy the
   factual information accurately from the source.

10. If multiple sources support the answer,
    combine only explicitly supported information.

11. If the evidence truly does not answer the
    question, say that the answer cannot be found
    in the provided evidence.

12. Keep the answer short and factual.

RESPONSE FORMAT:

Answer:
<direct factual answer>

Source:
[Source X]

Return ONLY the answer and source.
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict evidence-grounded "
                    "Wikimedia research assistant. "
                    "Never contradict provided evidence."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response[
        "message"
    ][
        "content"
    ].strip()

    return answer


# ============================================================
# CONFIDENCE
# ============================================================

def calculate_confidence(results):

    if not results:

        return {
            "level": "Low",
            "reason": "No relevant sources were retrieved."
        }

    similarities = [
        result["similarity"]
        for result in results
    ]

    best_match = max(
        similarities
    )

    average_match = (
        sum(similarities)
        / len(similarities)
    )

    strong_sources = sum(
        1
        for score in similarities
        if score >= 0.60
    )

    if (
        best_match >= 0.75
        and average_match >= 0.60
        and strong_sources >= 2
    ):

        level = "High"

        reason = (
            "Multiple strong sources were retrieved "
            "with high semantic similarity."
        )

    elif (
        best_match >= 0.60
        and average_match >= 0.45
    ):

        level = "Moderate"

        reason = (
            "Relevant sources were found, but "
            "the evidence strength is moderate."
        )

    else:

        level = "Low"

        reason = (
            "The retrieved evidence has relatively "
            "low similarity to the question."
        )

    return {
        "level": level,
        "reason": reason
    }


# ============================================================
# DISPLAY SEARCH RESULTS
# ============================================================

def display_search_results(results):

    st.subheader(
        "🔎 Search Results"
    )

    for number, result in enumerate(
        results,
        start=1
    ):

        st.markdown(
            f"""
            ### {number}. {result['title']}

            **Similarity:** `{result['similarity']:.4f}`

            {result['text'][:1000]}
            """
        )

        st.divider()


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_sources(results):

    st.subheader(
        "📖 Sources Used"
    )

    for number, result in enumerate(
        results,
        start=1
    ):

        st.markdown(
            f"""
            ### Source {number}: {result['title']}

            **Similarity:** `{result.get('similarity', 0):.4f}`

            **Research Score:** `{result.get('research_score', 0):.4f}`

            **Evidence**

            {result.get('text', '')}
            """
        )

        if result.get("url"):

            st.markdown(
                f"🔗 {result['url']}"
            )

        st.divider()


# ============================================================
# EVIDENCE VIEW
# ============================================================

def display_evidence_view(results):

    st.subheader(
        "🧾 Evidence View"
    )

    for number, result in enumerate(
        results,
        start=1
    ):

        with st.expander(
            f"Source {number}: {result['title']}"
        ):

            st.write(
                result.get(
                    "text",
                    ""
                )
            )


# ============================================================
# CONFIDENCE DISPLAY
# ============================================================

def display_confidence(confidence):

    level = confidence.get(
        "level",
        "Unknown"
    )

    reason = confidence.get(
        "reason",
        ""
    )

    if level == "High":

        css_class = "confidence-high"

    elif level == "Moderate":

        css_class = "confidence-medium"

    else:

        css_class = "confidence-low"

    st.markdown(
        f"""
        <div class="{css_class}">

        <h3>Research Confidence: {level}</h3>

        <p>{reason}</p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# RELATED ARTICLES
# ============================================================

def find_related_articles(
    selected_article,
    chunks,
    embeddings,
    model,
    top_k=5
):

    article_id = selected_article.get(
        "article_id"
    )

    embedding_index = selected_article.get(
        "embedding_index"
    )

    if embedding_index is None:

        return []

    selected_embedding = embeddings[
        embedding_index
    ]

    similarities = cosine_similarity(
        [selected_embedding],
        embeddings
    )[0]

    indices = np.argsort(
        similarities
    )[::-1]

    related = []

    seen = set()

    for index in indices:

        chunk = chunks[index]

        current_article_id = chunk.get(
            "article_id"
        )

        if current_article_id == article_id:

            continue

        if current_article_id in seen:

            continue

        seen.add(
            current_article_id
        )

        related.append(
            {
                "title": chunk.get(
                    "title",
                    "Unknown"
                ),

                "text": chunk.get(
                    "text",
                    ""
                ),

                "similarity": float(
                    similarities[index]
                ),

                "article_id":
                    current_article_id,

                "embedding_index":
                    int(index)
            }
        )

        if len(related) >= top_k:

            break

    return related


def display_related_articles(
    article,
    chunks,
    embeddings,
    model
):

    st.subheader(
        "🔗 Related Articles"
    )

    if "embedding_index" not in article:

        st.info(
            "Related article information "
            "is not available for this selection."
        )

        return

    related = find_related_articles(
        article,
        chunks,
        embeddings,
        model
    )

    for number, item in enumerate(
        related,
        start=1
    ):

        st.markdown(
            f"""
            **{number}. {item['title']}**

            Similarity: `{item['similarity']:.4f}`
            """
        )


# ============================================================
# ARTICLE SUMMARY
# ============================================================

def generate_summary(article):

    title = article.get(
        "title",
        "Unknown Article"
    )

    text = article.get(
        "abstract"
    )

    if not text:

        text = article.get(
            "text",
            ""
        )

    prompt = f"""
Summarize the following Wikimedia article.

Title:
{title}

Article:
{text}

Rules:

- Use only the provided information.
- Do not invent facts.
- Keep the summary concise.
- Mention the main topic and important facts.
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response[
        "message"
    ][
        "content"
    ]


# ============================================================
# KNOWLEDGE MAP
# ============================================================

def create_knowledge_map(results):

    if not results:

        st.info(
            "No research sources available "
            "for the knowledge map."
        )

        return

    titles = [
        result["title"]
        for result in results
    ]

    similarities = [
        result["similarity"]
        for result in results
    ]

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=titles,
            y=similarities
        )
    )

    figure.update_layout(
        title="Research Source Similarity",
        xaxis_title="Article",
        yaxis_title="Similarity",
        height=500
    )

    st.plotly_chart(
        figure,
        use_container_width=True
    )


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">'
    '📚 Wikimedia AI Research Assistant'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered research over structured Wikimedia data'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🧭 Navigation"
)

mode = st.sidebar.radio(
    "Choose a mode",
    [
        "Research Assistant",
        "Multi-Document Research",
        "Research History",
        "Knowledge Explorer",
        "Article Summarizer",
        "Related Articles",
        "Knowledge Map"
    ]
)


# ============================================================
# LOAD DATA
# ============================================================

chunks = load_chunks()

embeddings = load_embeddings()

embedding_model = load_embedding_model()


# ============================================================
# RESEARCH ASSISTANT
# ============================================================

if mode == "Research Assistant":

    st.header(
        "🤖 Research Assistant"
    )

    question = st.text_input(
        "Ask a question about the Wikimedia dataset",
        placeholder=(
            "Example: What airport serves Juruena?"
        )
    )

    if st.button(
        "🔍 Search",
        type="primary"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching Wikimedia knowledge base..."
            ):

                standalone_question = (
                    rewrite_followup_question(
                        question
                    )
                )

                results = retrieve_chunks(
                    question=standalone_question,
                    chunks=chunks,
                    embeddings=embeddings,
                    model=embedding_model,
                    top_k=TOP_K
                )

            with st.spinner(
                "Generating grounded answer..."
            ):

                answer = generate_grounded_answer(
                    question=standalone_question,
                    results=results
                )

            st.session_state.last_question = (
                standalone_question
            )

            st.session_state.last_results = (
                results
            )

            st.session_state.conversation.append(
                {
                    "question":
                        standalone_question,

                    "answer":
                        answer
                }
            )

            st.subheader(
                "🧠 Answer"
            )

            st.write(
                answer
            )

            display_search_results(
                results
            )

            display_evidence_view(
                results
            )


# ============================================================
# MULTI-DOCUMENT RESEARCH
# ============================================================

elif mode == "Multi-Document Research":

    st.header(
        "🔬 Multi-Document Research"
    )

    st.write(
        "Retrieve multiple sources, synthesize "
        "the evidence, compare sources, verify "
        "claims, and export a research report."
    )

    question = st.text_input(
        "Research Question",
        placeholder=(
            "Example: What airports are located in Brazil?"
        )
    )

    if st.button(
        "🚀 Start Research",
        type="primary"
    ):

        if not question.strip():

            st.warning(
                "Please enter a research question."
            )

        else:

            # ------------------------------------------------
            # STEP 1: RETRIEVE SOURCES
            # ------------------------------------------------

            with st.spinner(
                "Retrieving research sources..."
            ):

                results = retrieve_research_articles(
                    question=question,
                    chunks=chunks,
                    embeddings=embeddings,
                    model=embedding_model,
                    top_k=TOP_K,
                    min_articles=2
                )

            st.subheader(
                "📚 Retrieved Research Sources"
            )

            if not results:

                st.error(
                    "No relevant research sources "
                    "were found."
                )

                st.stop()

            for number, result in enumerate(
                results,
                start=1
            ):

                st.markdown(
                    f"""
                    ### Source {number}: {result['title']}

                    **Similarity:** `{result['similarity']:.4f}`

                    **Research Score:** `{result.get('research_score', 0):.4f}`

                    {result['text']}
                    """
                )

                st.divider()

            # ------------------------------------------------
            # STEP 2: SYNTHESIS
            # ------------------------------------------------

            with st.spinner(
                "Synthesizing research answer..."
            ):

                answer = synthesize_research(
                    question=question,
                    results=results
                )

            st.subheader(
                "🧠 Research Answer"
            )

            st.write(
                answer
            )

            # ------------------------------------------------
            # STEP 3: STATISTICS
            # ------------------------------------------------

            statistics = get_research_statistics(
                results
            )

            # ------------------------------------------------
            # STEP 4: CONFIDENCE
            # ------------------------------------------------

            confidence = calculate_confidence(
                results
            )

            st.subheader(
                "📊 Research Confidence"
            )

            display_confidence(
                confidence
            )

            st.write(
                f"**Articles found:** "
                f"{statistics.get('articles', 0)}"
            )

            st.write(
                f"**Best similarity:** "
                f"{statistics.get('best_match', 0):.4f}"
            )

            st.write(
                f"**Average similarity:** "
                f"{statistics.get('average_match', 0):.4f}"
            )

            # ------------------------------------------------
            # STEP 5: SOURCES
            # ------------------------------------------------

            display_sources(
                results
            )

            # ------------------------------------------------
            # STEP 6: SOURCE COMPARISON
            # ------------------------------------------------

            st.subheader(
                "⚖️ Source Comparison"
            )

            with st.spinner(
                "Comparing retrieved sources..."
            ):

                comparison = compare_research_sources(
                    question=question,
                    results=results
                )

            st.markdown(
                comparison
            )

            # ------------------------------------------------
            # STEP 7: EVIDENCE VERIFICATION
            # ------------------------------------------------

            st.subheader(
                "🔎 Evidence Verification"
            )

            with st.spinner(
                "Verifying research claims..."
            ):

                verification = verify_research_answer(
                    answer=answer,
                    results=results
                )

            st.markdown(
                verification
            )

            # ------------------------------------------------
            # STEP 8: GENERATE VERIFIED REPORT
            # ------------------------------------------------

            st.subheader(
                "📄 Export Research Report"
            )

            report = generate_research_report(
                question=question,
                answer=answer,
                results=results,
                statistics=statistics,
                confidence=confidence,
                comparison=comparison,
                verification=verification
            )

            st.download_button(
                label="⬇️ Download Verified Research Report",
                data=report,
                file_name="wikimedia_verified_research_report.md",
                mime="text/markdown"
            )

            # ------------------------------------------------
            # STEP 9: SAVE RESEARCH HISTORY
            # ------------------------------------------------

            save_research(
                session_state=st.session_state,
                question=question,
                answer=answer,
                results=results,
                statistics=statistics,
                confidence=confidence,
                report=report
            )

            st.success(
                "Research completed and saved to history."
            )


# ============================================================
# RESEARCH HISTORY
# ============================================================

elif mode == "Research History":

    st.header(
        "🕘 Research History"
    )

    history = get_research_history(
        st.session_state
    )

    if not history:

        st.info(
            "No research history available yet."
        )

    else:

        if st.button(
            "🗑️ Clear Research History"
        ):

            clear_research_history(
                st.session_state
            )

            st.success(
                "Research history cleared."
            )

            st.rerun()

        for number, item in enumerate(
            reversed(history),
            start=1
        ):

            with st.expander(
                f"{number}. {item['question']}"
            ):

                st.write(
                    f"**Time:** {item['timestamp']}"
                )

                st.write(
                    f"**Confidence:** "
                    f"{item['confidence']['level']}"
                )

                st.markdown(
                    "### Research Answer"
                )

                st.write(
                    item["answer"]
                )

                st.markdown(
                    "### Statistics"
                )

                statistics = item[
                    "statistics"
                ]

                st.write(
                    f"Articles: "
                    f"{statistics.get('articles', 0)}"
                )

                st.write(
                    f"Best similarity: "
                    f"{statistics.get('best_match', 0):.4f}"
                )

                st.write(
                    f"Average similarity: "
                    f"{statistics.get('average_match', 0):.4f}"
                )

                st.download_button(
                    label="⬇️ Download Report",
                    data=item["report"],
                    file_name=(
                        "wikimedia_research_history_report.md"
                    ),
                    mime="text/markdown",
                    key=f"history_download_{number}"
                )


# ============================================================
# KNOWLEDGE EXPLORER
# ============================================================

elif mode == "Knowledge Explorer":

    st.header(
        "🗺️ Knowledge Explorer"
    )

    query = st.text_input(
        "Search the Wikimedia knowledge base",
        placeholder="Enter an article or topic"
    )

    if st.button(
        "🔎 Explore"
    ):

        if not query.strip():

            st.warning(
                "Please enter a topic."
            )

        else:

            results = retrieve_chunks(
                question=query,
                chunks=chunks,
                embeddings=embeddings,
                model=embedding_model,
                top_k=10
            )

            display_search_results(
                results
            )


# ============================================================
# ARTICLE SUMMARIZER
# ============================================================

elif mode == "Article Summarizer":

    st.header(
        "📝 Article Summarizer"
    )

    query = st.text_input(
        "Find an article to summarize",
        placeholder="Example: Juruena Airport"
    )

    if st.button(
        "🔎 Find Article"
    ):

        if not query.strip():

            st.warning(
                "Please enter an article name."
            )

        else:

            results = retrieve_chunks(
                question=query,
                chunks=chunks,
                embeddings=embeddings,
                model=embedding_model,
                top_k=5
            )

            st.session_state.summary_results = (
                results
            )

    results = st.session_state.summary_results

    if results:

        selected_titles = [
            result["title"]
            for result in results
        ]

        selected_title = st.selectbox(
            "Select an article",
            selected_titles
        )

        selected = next(
            result
            for result in results
            if result["title"] == selected_title
        )

        if st.button(
            "✨ Generate Summary"
        ):

            with st.spinner(
                "Generating summary..."
            ):

                summary = generate_summary(
                    selected
                )

            st.subheader(
                "📋 Summary"
            )

            st.write(
                summary
            )

            st.subheader(
                "📖 Source Text"
            )

            st.write(
                selected["text"]
            )


# ============================================================
# RELATED ARTICLES
# ============================================================

elif mode == "Related Articles":

    st.header(
        "🔗 Related Articles"
    )

    query = st.text_input(
        "Enter an article or topic"
    )

    if st.button(
        "Find Related Articles"
    ):

        if not query.strip():

            st.warning(
                "Please enter a topic."
            )

        else:

            results = retrieve_chunks(
                question=query,
                chunks=chunks,
                embeddings=embeddings,
                model=embedding_model,
                top_k=1
            )

            if not results:

                st.warning(
                    "No article found."
                )

            else:

                article = results[0]

                st.subheader(
                    article["title"]
                )

                st.write(
                    article["text"]
                )

                st.subheader(
                    "🔗 Related Articles"
                )

                query_embedding = (
                    embedding_model.encode(
                        [query],
                        normalize_embeddings=True
                    )
                )

                similarities = cosine_similarity(
                    query_embedding,
                    embeddings
                )[0]

                indices = np.argsort(
                    similarities
                )[::-1]

                seen = {
                    article.get(
                        "article_id"
                    )
                }

                related = []

                for index in indices:

                    candidate = chunks[index]

                    candidate_id = candidate.get(
                        "article_id"
                    )

                    if candidate_id in seen:

                        continue

                    seen.add(
                        candidate_id
                    )

                    related.append(
                        {
                            "title":
                                candidate.get(
                                    "title",
                                    "Unknown"
                                ),

                            "similarity":
                                float(
                                    similarities[index]
                                )
                        }
                    )

                    if len(related) >= 10:

                        break

                for number, item in enumerate(
                    related,
                    start=1
                ):

                    st.write(
                        f"{number}. "
                        f"{item['title']} "
                        f"— similarity "
                        f"{item['similarity']:.4f}"
                    )


# ============================================================
# KNOWLEDGE MAP
# ============================================================

elif mode == "Knowledge Map":

    st.header(
        "🕸️ Interactive Knowledge Map"
    )

    query = st.text_input(
        "Enter a topic for the knowledge map",
        placeholder="Example: airports in Brazil"
    )

    if st.button(
        "🗺️ Generate Knowledge Map"
    ):

        if not query.strip():

            st.warning(
                "Please enter a topic."
            )

        else:

            results = retrieve_research_articles(
                question=query,
                chunks=chunks,
                embeddings=embeddings,
                model=embedding_model,
                top_k=10,
                min_articles=2
            )

            if not results:

                st.warning(
                    "No relevant sources found."
                )

            else:

                create_knowledge_map(
                    results
                )

                display_sources(
                    results
                )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.markdown(
    "---"
)

st.sidebar.caption(
    "Wikimedia AI Research Assistant"
)

st.sidebar.caption(
    "Powered by Wikimedia data, "
    "Sentence Transformers and Ollama"
)