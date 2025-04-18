import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

class TopicModel:
    def __init__(self, csv_path='questions.csv'):
        data = pd.read_csv(csv_path)
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()
        self.X = self.vectorizer.fit_transform(data['question'])
        self.y = data['topic']
        self.model.fit(self.X, self.y)

    def predict(self, question):
        x = self.vectorizer.transform([question])
        pred = self.model.predict(x)
        prob = self.model.predict_proba(x).max()
        return pred[0] if prob > 0.2 else "Unknown"
