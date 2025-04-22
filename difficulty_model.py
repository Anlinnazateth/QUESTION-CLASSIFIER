"""Difficulty classification model using Multinomial Naive Bayes."""

import os
from typing import Optional, Tuple

import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


class DifficultyModel:
    """Classifies questions into difficulty levels: Easy, Medium, Hard."""

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
        self.model.fit(X, data["difficulty"])

    def predict(self, question: str) -> str:
        """Predict the difficulty of a question."""
        question = question.lower().strip()
        x = self.vectorizer.transform([question])

        if x.nnz == 0:
            return "Unknown"

        prob = self.model.predict_proba(x).max()
        return self.model.predict(x)[0] if prob > 0.2 else "Unknown"

    def predict_with_confidence(self, question: str) -> Tuple[str, float]:
        """Predict difficulty with confidence score."""
        question = question.lower().strip()
        x = self.vectorizer.transform([question])

        if x.nnz == 0:
            return "Unknown", 0.0

        prob = self.model.predict_proba(x).max()
        label = self.model.predict(x)[0] if prob > 0.2 else "Unknown"
        return label, float(prob)

    def save(self, path: str = "models/difficulty_model.joblib") -> None:
        """Save model and vectorizer to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "vectorizer": self.vectorizer}, path)

    @classmethod
    def load(cls, path: str = "models/difficulty_model.joblib") -> "DifficultyModel":
        """Load a saved model from disk."""
        data = joblib.load(path)
        instance = cls.__new__(cls)
        instance.model = data["model"]
        instance.vectorizer = data["vectorizer"]
        return instance
