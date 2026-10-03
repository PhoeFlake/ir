import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk
import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Set nltk data path to local project to avoid global permission issues
nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download required NLTK resources silently
nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    tokens = word_tokenize(text)
    cleaned_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return " ".join(cleaned_tokens)

def load_and_prepare_data(num_samples=10000):
    """
    Loads Quora Question Pairs from local questions.csv.
    Extracts unique questions to form a corpus and identifies relevant pairs for queries.
    """
    # Load from local CSV
    df = pd.read_csv("questions.csv")
    
    questions_dict = {}
    qrels = {}
    
    count = 0
    for _, row in df.iterrows():
        if count >= num_samples:
            break
            
        try:
            is_dup = int(row['is_duplicate']) if pd.notna(row['is_duplicate']) else 0
            q1_id = int(row['qid1'])
            q1_text = str(row['question1'])
            q2_id = int(row['qid2'])
            q2_text = str(row['question2'])
        except Exception:
            continue
            
        # Add to corpus
        questions_dict[q1_id] = q1_text
        questions_dict[q2_id] = q2_text
        
        # Build relevance judgments (qrels)
        if is_dup == 1:
            if q1_id not in qrels:
                qrels[q1_id] = set()
            qrels[q1_id].add(q2_id)
            
            if q2_id not in qrels:
                qrels[q2_id] = set()
            qrels[q2_id].add(q1_id)
            
        count += 1
        
    corpus_ids = list(questions_dict.keys())
    corpus_texts = list(questions_dict.values())
    
    return corpus_ids, corpus_texts, qrels

def build_tfidf(corpus_processed):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus_processed)
    return vectorizer, tfidf_matrix

def build_bm25(corpus_processed):
    tokenized_corpus = [doc.split() for doc in corpus_processed]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

def build_dense(corpus_texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(corpus_texts, show_progress_bar=False)
    return model, embeddings

def reciprocal_rank_fusion(rankings_list, k=60):
    """
    Reciprocal Rank Fusion formula: 1 / (k + rank)
    """
    rrf_scores = {}
    for ranking in rankings_list:
        for rank, doc_id in enumerate(ranking):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
    
    # Sort documents by their accumulated RRF score in descending order
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, score in sorted_rrf]
    
def calculate_metrics(retrieved, relevant, k=10):
    if not relevant:
        return 0.0, 0.0, 0.0, 0.0
        
    retrieved_k = retrieved[:k]
    
    # Precision@10
    relevant_retrieved = len(set(retrieved_k).intersection(relevant))
    p_10 = relevant_retrieved / k
    
    # Recall
    if len(relevant) == 0:
        recall = 0.0
    else:
        recall = len(set(retrieved).intersection(relevant)) / len(relevant)
    
    # MAP (Mean Average Precision)
    ap = 0.0
    hits = 0
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            hits += 1
            ap += hits / (i + 1)
    ap = ap / len(relevant) if relevant else 0
    
    # nDCG (Normalized Discounted Cumulative Gain)
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_k):
        if doc_id in relevant:
            dcg += 1.0 / np.log2(i + 2) # i starts at 0, so rank is i+1 -> log2(rank+1) -> log2(i+2)
            
    idcg = 0.0
    for i in range(min(len(relevant), k)):
        idcg += 1.0 / np.log2(i + 2)
        
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    return p_10, recall, ap, ndcg
