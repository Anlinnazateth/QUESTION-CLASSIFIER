"""Question Classifier — Streamlit application.

Classifies questions by topic and difficulty using Naive Bayes models.
"""

import streamlit as st
from topic_model import TopicModel
from difficulty_model import DifficultyModel

st.set_page_config(page_title="Question Classifier", page_icon="🧠", layout="centered")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🧠 Question Classifier")
st.sidebar.markdown(
    """
    Classifies questions into:
    - **Topic**: Math, Science, Geography, Literature, Biology,
      History, Chemistry, Physics, Art, Computer Science
    - **Difficulty**: Easy, Medium, Hard

    Built with Multinomial Naive Bayes and CountVectorizer.
    """
)

EXAMPLE_QUESTIONS = [
    "What is the capital of France?",
    "Explain quantum mechanics.",
    "What is the derivative of x^2?",
    "Who painted the Mona Lisa?",
    "Describe the process of mitosis.",
    "What is a black hole?",
    "Define an algorithm.",
]

st.sidebar.markdown("### Try an example")
example = st.sidebar.selectbox("Example questions:", [""] + EXAMPLE_QUESTIONS)

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------


@st.cache_resource
def load_models():
    """Load topic and difficulty models (cached)."""
    return TopicModel(), DifficultyModel()


try:
    topic_model, difficulty_model = load_models()
except FileNotFoundError as e:
    st.error(f"Could not load training data: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("Question Classifier")
st.markdown("Enter a question to classify its **topic** and **difficulty level**.")

user_question = st.text_input("Enter your question:", value=example)

if user_question:
    topic, topic_conf = topic_model.predict_with_confidence(user_question)
    difficulty, diff_conf = difficulty_model.predict_with_confidence(user_question)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Topic")
        st.info(f"**{topic}**")
        st.caption(f"Confidence: {topic_conf:.0%}")

    with col2:
        st.markdown("### Difficulty")
        color_map = {"Easy": "success", "Medium": "warning", "Hard": "error", "Unknown": "info"}
        getattr(st, color_map.get(difficulty, "info"))(f"**{difficulty}**")
        st.caption(f"Confidence: {diff_conf:.0%}")

# ---------------------------------------------------------------------------
# Batch classification
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Batch Classification")
batch_text = st.text_area(
    "Enter multiple questions (one per line):",
    height=150,
    placeholder="What is 2 + 2?\nWho wrote Hamlet?\nExplain photosynthesis.",
)

if batch_text.strip():
    questions = [q.strip() for q in batch_text.strip().split("\n") if q.strip()]
    results = []
    for q in questions:
        t, tc = topic_model.predict_with_confidence(q)
        d, dc = difficulty_model.predict_with_confidence(q)
        results.append({
            "Question": q,
            "Topic": t,
            "Topic Confidence": f"{tc:.0%}",
            "Difficulty": d,
            "Difficulty Confidence": f"{dc:.0%}",
        })
    st.dataframe(results, use_container_width=True)
