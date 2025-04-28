# Question Classifier

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Classify questions by **topic** and **difficulty** using machine learning. Built with Multinomial Naive Bayes, CountVectorizer, and Streamlit.

## Demo

| Input | Topic | Difficulty |
|-------|-------|------------|
| "What is the capital of France?" | Geography | Easy |
| "Explain quantum mechanics." | Science | Hard |
| "What is the derivative of x^2?" | Math | Medium |

<details>
<summary>Screenshots</summary>

See the `outputs/` folder for screenshots of the Streamlit interface and command line output.
</details>

## Features

- Single question classification with confidence scores
- Batch classification (multiple questions at once)
- 10 topic categories: Math, Science, Geography, Literature, Biology, History, Chemistry, Physics, Art, Computer Science
- 3 difficulty levels: Easy, Medium, Hard
- Model persistence with joblib (save/load trained models)
- Example questions in the sidebar

## How It Works

1. **Text Vectorization** — Questions are converted to numerical feature vectors using `CountVectorizer` (bag-of-words)
2. **Classification** — Two separate `MultinomialNB` (Naive Bayes) models predict topic and difficulty independently
3. **Confidence Thresholding** — Predictions below 20% confidence are labeled "Unknown"

## Installation

```bash
git clone https://github.com/Anlinnazateth/QUESTION-CLASSIFIER.git
cd QUESTION-CLASSIFIER
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Dataset

The `questions.csv` file contains 100+ labeled questions:

```csv
question,topic,difficulty
"What is the capital of France?",Geography,Easy
"Explain quantum mechanics.",Science,Hard
```

### Adding Training Data

Add new rows to `questions.csv` following the same format. The models retrain on startup.

**Supported topics:** Math, Science, Geography, Literature, Biology, History, Chemistry, Physics, Art, Computer Science

**Supported difficulties:** Easy, Medium, Hard

## Project Structure

```
QUESTION-CLASSIFIER/
├── app.py                  # Streamlit UI
├── topic_model.py          # Topic classification model
├── difficulty_model.py     # Difficulty classification model
├── questions.csv           # Training dataset (100+ questions)
├── requirements.txt        # Python dependencies
├── LICENSE
├── .gitignore
├── outputs/                # Demo screenshots
│   ├── command prompt.png
│   ├── streamlit interface(known).png
│   └── streamlit interface(unknown).png
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI
└── tests/
    └── test_models.py      # Unit tests
```

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes
4. Push and open a Pull Request

## License

MIT License. See [LICENSE](LICENSE) for details.
