def initialize_research_history(session_state):
    """
    Initialize research history in Streamlit session state.
    """

    if "research_history" not in session_state:
        session_state.research_history = []


def save_research(
    session_state,
    question,
    answer,
    results,
    statistics,
    confidence,
    report
):
    """
    Save one completed research session.
    """

    research_item = {
        "question": question,
        "answer": answer,
        "results": results,
        "statistics": statistics,
        "confidence": confidence,
        "report": report
    }

    session_state.research_history.append(
        research_item
    )


def get_research_history(session_state):
    """
    Return all saved research sessions.
    """

    return session_state.get(
        "research_history",
        []
    )


def clear_research_history(session_state):
    """
    Remove all saved research sessions.
    """

    session_state.research_history = []