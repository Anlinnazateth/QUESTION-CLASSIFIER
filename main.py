import streamlit as st
from topic_model import TopicModel
from difficulty_model import DifficultyModel

# Load models
topic_model = TopicModel()
difficulty_model = DifficultyModel()

st.title("Question Classifier")

user_question = st.text_input("Enter your question:")

if user_question:
    topic = topic_model.predict(user_question)
    difficulty = difficulty_model.predict(user_question)

    st.write("### Prediction Results")
    st.write(f"**Topic:** {topic}")
    st.write(f"**Difficulty:** {difficulty}")

    # Optional terminal output
    print(f"Question: {user_question}")
    print(f"Topic: {topic}")
    print(f"Difficulty: {difficulty}")
