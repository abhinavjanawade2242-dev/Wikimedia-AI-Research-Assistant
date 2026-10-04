import json
from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Research Evaluation",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------------
# PATH
# ---------------------------------------------------------

EVALUATION_PATH = Path(
    "data/processed/research_evaluation.json"
)


# ---------------------------------------------------------
# LOAD EVALUATION RESULTS
# ---------------------------------------------------------

@st.cache_data
def load_evaluation():
    if not EVALUATION_PATH.exists():
        return None

    with open(EVALUATION_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


evaluation = load_evaluation()


# ---------------------------------------------------------
# PAGE TITLE
# ---------------------------------------------------------

st.title("📊 Research Quality Dashboard")

st.markdown(
    """
    This dashboard evaluates the quality of the Wikimedia
    AI Research Assistant using retrieval performance and
    evidence verification results.
    """
)


# ---------------------------------------------------------
# CHECK EVALUATION FILE
# ---------------------------------------------------------

if evaluation is None:

    st.warning(
        "Research evaluation results were not found."
    )

    st.info(
        "Run the following command from the project root:"
    )

    st.code(
        "python scripts/evaluate_research.py",
        language="powershell"
    )

    st.stop()


# ---------------------------------------------------------
# OVERALL METRICS
# ---------------------------------------------------------

retrieval_recall = evaluation.get(
    "average_retrieval_recall",
    0.0
)

verification_score = evaluation.get(
    "average_verification_score",
    0.0
)

overall_quality = evaluation.get(
    "average_overall_quality",
    0.0
)


st.subheader("Overall Performance")


col1, col2, col3 = st.columns(3)


with col1:
    st.metric(
        "Retrieval Recall",
        f"{retrieval_recall:.2%}"
    )


with col2:
    st.metric(
        "Verification Score",
        f"{verification_score:.2%}"
    )


with col3:
    st.metric(
        "Overall Research Quality",
        f"{overall_quality:.2%}"
    )


# ---------------------------------------------------------
# QUALITY ASSESSMENT
# ---------------------------------------------------------

st.subheader("Quality Assessment")


if overall_quality >= 0.80:

    st.success(
        "Excellent research quality. "
        "The system retrieves relevant sources and "
        "produces well-supported answers."
    )

elif overall_quality >= 0.60:

    st.info(
        "Good research quality, but there is room "
        "for improvement in retrieval or evidence verification."
    )

elif overall_quality >= 0.40:

    st.warning(
        "Moderate research quality. "
        "Some research questions may require better retrieval "
        "or stronger evidence grounding."
    )

else:

    st.error(
        "Low research quality. "
        "The retrieval and/or verification pipeline needs improvement."
    )


# ---------------------------------------------------------
# QUESTION-LEVEL RESULTS
# ---------------------------------------------------------

st.subheader("Question-Level Evaluation")


questions = evaluation.get(
    "questions",
    evaluation.get(
        "results",
        []
    )
)


if questions:

    table_data = []

    for item in questions:

        table_data.append(
            {
                "Question": item.get(
                    "question",
                    ""
                ),
                "Retrieval Recall": item.get(
                    "retrieval_recall",
                    0.0
                ),
                "Verification Score": item.get(
                    "verification_score",
                    0.0
                ),
                "Overall Quality": item.get(
                    "overall_quality",
                    0.0
                )
            }
        )

    dataframe = pd.DataFrame(table_data)

    if not dataframe.empty:

        for column in [
            "Retrieval Recall",
            "Verification Score",
            "Overall Quality"
        ]:

            if column in dataframe.columns:

                dataframe[column] = dataframe[column].apply(
                    lambda value: f"{value:.2%}"
                    if isinstance(value, (int, float))
                    else value
                )

        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True
        )

else:

    st.info(
        "No question-level evaluation results were found."
    )


# ---------------------------------------------------------
# DETAILED RESULTS
# ---------------------------------------------------------

st.subheader("Detailed Evaluation")


if questions:

    for index, item in enumerate(questions, start=1):

        question = item.get(
            "question",
            f"Question {index}"
        )

        with st.expander(
            f"Question {index}: {question}"
        ):

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Retrieval Recall",
                    f"{item.get('retrieval_recall', 0.0):.2%}"
                )

            with col2:
                st.metric(
                    "Verification Score",
                    f"{item.get('verification_score', 0.0):.2%}"
                )

            with col3:
                st.metric(
                    "Overall Quality",
                    f"{item.get('overall_quality', 0.0):.2%}"
                )

            expected_articles = item.get(
                "expected_articles",
                []
            )

            retrieved_articles = item.get(
                "retrieved_articles",
                []
            )

            if expected_articles:

                st.markdown("**Expected Articles**")

                for article in expected_articles:
                    st.write(f"• {article}")

            if retrieved_articles:

                st.markdown("**Retrieved Articles**")

                for article in retrieved_articles:
                    st.write(f"• {article}")

            supported = item.get(
                "supported_claims",
                0
            )

            partial = item.get(
                "partially_supported_claims",
                0
            )

            unsupported = item.get(
                "unsupported_claims",
                0
            )

            st.markdown("**Evidence Verification**")

            verification_columns = st.columns(3)

            with verification_columns[0]:
                st.write(
                    f"✅ Supported: {supported}"
                )

            with verification_columns[1]:
                st.write(
                    f"🟡 Partially Supported: {partial}"
                )

            with verification_columns[2]:
                st.write(
                    f"❌ Unsupported: {unsupported}"
                )


# ---------------------------------------------------------
# STRONG AND WEAK QUESTIONS
# ---------------------------------------------------------

st.subheader("Research Performance Analysis")


if questions:

    scored_questions = []

    for item in questions:

        scored_questions.append(
            (
                item.get("question", ""),
                item.get("overall_quality", 0.0)
            )
        )

    scored_questions.sort(
        key=lambda item: item[1],
        reverse=True
    )

    strong_questions = scored_questions[:3]

    weak_questions = scored_questions[-3:]

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🟢 Strongest Queries")

        for question, score in strong_questions:

            st.write(
                f"**{question}** — {score:.2%}"
            )

    with col2:

        st.markdown("### 🔴 Queries Needing Improvement")

        for question, score in weak_questions:

            st.write(
                f"**{question}** — {score:.2%}"
            )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Wikimedia AI Research Assistant • "
    "Research Quality Evaluation"
)