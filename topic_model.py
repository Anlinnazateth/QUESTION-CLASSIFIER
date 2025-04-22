"""Topic classification model using Multinomial Naive Bayes."""

import os
from typing import Optional, Tuple

import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


class TopicModel:
    """Classifies questions into topic categories using Naive Bayes.

    Topics include: Math, Science, Geography, Literature, Biology,
    History, Chemistry, Physics, Art, Computer Science.
    """

    def __init__(self, csv_path: str = "questions.csv"):
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()
        self._train(csv_path)

    def _train(self, csv_path: str) -> None:
        """Train the model from CSV data."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Training data not found: {csv_path}")

        data = pd.read_csv(csv_path)
        data["question"] = data["question"].str.lower().str.strip()

        X = self.vectorizer.fit_transform(data["question"])
        self.model.fit(X, data["topic"])

    def predict(self, question: str) -> str:
        """Predict the topic of a question."""
        question = question.lower().strip()
        x = self.vectorizer.transform([question])

        if x.nnz == 0:
            return "Unknown"

        prob = self.model.predict_proba(x).max()
        return self.model.predict(x)[0] if prob > 0.2 else "Unknown"

    def predict_with_confidence(self, question: str) -> Tuple[str, float]:
        """Predict topic with confidence score."""
        question = question.lower().strip()
        x = self.vectorizer.transform([question])

        if x.nnz == 0:
            return "Unknown", 0.0

        prob = self.model.predict_proba(x).max()
        label = self.model.predict(x)[0] if prob > 0.2 else "Unknown"
        return label, float(prob)

    def save(self, path: str = "models/topic_model.joblib") -> None:
        """Save model and vectorizer to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "vectorizer": self.vectorizer}, path)

    @classmethod
    def load(cls, path: str = "models/topic_model.joblib") -> "TopicModel":
        """Load a saved model from disk."""
        data = joblib.load(path)
        instance = cls.__new__(cls)
        instance.model = data["model"]
        instance.vectorizer = data["vectorizer"]
        return instance
