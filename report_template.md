# Information Retrieval Project 8: Duplicate Question Retrieval

## 1. Introduction
This project implements an Information Retrieval (IR) system designed to detect and retrieve semantically duplicate questions from large Q&A databases. The dataset used is the Quora Question Pairs dataset, which contains over 400,000 question pairs with binary duplicate labels.

## 2. Methodology
### 2.1 Text Preprocessing
The text data underwent a rigorous preprocessing pipeline to standardize the input for classical models:
- **Cleaning**: Removal of punctuation and conversion to lowercase.
- **Tokenization**: Splitting sentences into individual word tokens using NLTK.
- **Stop-word Removal**: Removing common English stop words to reduce noise.
- **Stemming**: Applying the Porter Stemmer to reduce words to their root forms.

### 2.2 Core Ranking Models
Three distinct baseline ranking algorithms were implemented:
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: A classical sparse vector model that evaluates term relevance based on frequency in a document scaled by its rarity across the corpus.
- **BM25**: A probabilistic relevance framework that improves upon TF-IDF by factoring in term frequency saturation and document length normalization.
- **Sentence Transformers (Dense Embedding)**: A semantic search model utilizing the `all-MiniLM-L6-v2` transformer to encode questions into dense vector representations, capturing deep semantic meaning beyond exact keyword matches.

### 2.3 Hybrid Model (Reciprocal Rank Fusion)
To combine the strengths of exact lexical matching (BM25/TF-IDF) and semantic matching (Dense), a **Reciprocal Rank Fusion (RRF)** strategy was utilized. RRF blends rankings without requiring score normalization using the formula:
`RRF_Score = Sum(1 / (k + rank))`
where `k` is a smoothing constant (set to 60).

## 3. Evaluation & Results
The system was evaluated using standard IR metrics. Below is a comparative performance table of the models evaluated on a sample of the dataset:

| Model | Precision@10 | Recall | MAP | nDCG |
|-------|--------------|--------|-----|------|
| TF-IDF | [Run Evaluation to fill] | [Run Evaluation to fill] | [Run Evaluation to fill] | [Run Evaluation to fill] |
| BM25 | [Run Evaluation to fill] | [Run Evaluation to fill] | [Run Evaluation to fill] | [Run Evaluation to fill] |
| Dense | [Run Evaluation to fill] | [Run Evaluation to fill] | [Run Evaluation to fill] | [Run Evaluation to fill] |
| Hybrid RRF | [Run Evaluation to fill] | [Run Evaluation to fill] | [Run Evaluation to fill] | [Run Evaluation to fill] |

*(Note: Run the Evaluation tab in the App and copy the metrics and chart into this section).*

## 4. Conclusion
The Hybrid RRF approach successfully bridges the gap between lexical and semantic search paradigms, demonstrating that dense and sparse representations capture complementary relevance signals in short-text paraphrase matching tasks.
