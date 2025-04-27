"""Tests for the question classifier models."""

import os
import sys
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from topic_model import TopicModel
from difficulty_model import DifficultyModel

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "questions.csv")


@pytest.fixture
def topic_model():
    return TopicModel(csv_path=CSV_PATH)


@pytest.fixture
def difficulty_model():
    return DifficultyModel(csv_path=CSV_PATH)


class TestTopicModel:
    def test_initialization(self, topic_model):
        assert topic_model.model is not None
        assert topic_model.vectorizer is not None

    def test_predict_returns_string(self, topic_model):
        result = topic_model.predict("What is 2 + 2?")
        assert isinstance(result, str)

    def test_predict_known_topic(self, topic_model):
        result = topic_model.predict("What is the capital of France?")
        assert result in ["Geography", "History", "Unknown"]

    def test_predict_math(self, topic_model):
        result = topic_model.predict("What is the derivative of x squared?")
        assert result in ["Math", "Unknown"]

    def test_predict_with_confidence(self, topic_model):
        label, confidence = topic_model.predict_with_confidence("What is gravity?")
        assert isinstance(label, str)
        assert 0.0 <= confidence <= 1.0

    def test_empty_input(self, topic_model):
        result = topic_model.predict("")
        assert result == "Unknown"

    def test_unknown_input(self, topic_model):
        label, conf = topic_model.predict_with_confidence("xyzzy foobar baz")
        assert isinstance(label, str)


class TestDifficultyModel:
    def test_initialization(self, difficulty_model):
        assert difficulty_model.model is not None
        assert difficulty_model.vectorizer is not None

    def test_predict_returns_string(self, difficulty_model):
        result = difficulty_model.predict("What is 2 + 2?")
        assert isinstance(result, str)

    def test_predict_valid_difficulty(self, difficulty_model):
        result = difficulty_model.predict("What is the capital of France?")
        assert result in ["Easy", "Medium", "Hard", "Unknown"]

    def test_predict_with_confidence(self, difficulty_model):
        label, confidence = difficulty_model.predict_with_confidence("Explain quantum mechanics")
        assert isinstance(label, str)
        assert 0.0 <= confidence <= 1.0

    def test_empty_input(self, difficulty_model):
        result = difficulty_model.predict("")
        assert result == "Unknown"

    def test_text_normalization(self, difficulty_model):
        r1 = difficulty_model.predict("WHAT IS THE CAPITAL OF FRANCE?")
        r2 = difficulty_model.predict("what is the capital of france?")
        assert r1 == r2


class TestModelPersistence:
    def test_topic_model_save_load(self, topic_model, tmp_path):
        save_path = str(tmp_path / "topic.joblib")
        topic_model.save(save_path)
        loaded = TopicModel.load(save_path)
        original = topic_model.predict("What is gravity?")
        restored = loaded.predict("What is gravity?")
        assert original == restored

    def test_difficulty_model_save_load(self, difficulty_model, tmp_path):
        save_path = str(tmp_path / "diff.joblib")
        difficulty_model.save(save_path)
        loaded = DifficultyModel.load(save_path)
        original = difficulty_model.predict("What is 2+2?")
        restored = loaded.predict("What is 2+2?")
        assert original == restored


class TestMissingData:
    def test_topic_model_missing_csv(self):
        with pytest.raises(FileNotFoundError):
            TopicModel(csv_path="nonexistent.csv")

    def test_difficulty_model_missing_csv(self):
        with pytest.raises(FileNotFoundError):
            DifficultyModel(csv_path="nonexistent.csv")
