import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

class DifficultyModel:
    def __init__(self, csv_path='questions.csv'):
        data = pd.read_csv(csv_path)
        data['question'] = data['question'].str.lower().str.strip()

        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()
        self.X = self.vectorizer.fit_transform(data['question'])
        self.y = data['difficulty']
        self.model.fit(self.X, self.y)

    def predict(self, question):
        question = question.lower().strip()
        x = self.vectorizer.transform([question])

        # If no known words are found, the vector will be all zeros
        if x.nnz == 0:  # nnz = number of non-zero elements
            return "Unknown"

        prob = self.model.predict_proba(x).max()
        return self.model.predict(x)[0] if prob > 0.2 else "Unknown"
