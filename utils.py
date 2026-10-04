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

nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

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
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return " ".join(cleaned_tokens)

def load_and_prepare_data(num_samples=None):
    """Load rows from the local zipped CSV."""
    df = pd.read_csv("questions.csv.zip")

    questions_dict = {}
    qrels = {}
    count = 0

    for _, row in df.iterrows():
        if num_samples is not None and count >= num_samples:
            break
        try:
            is_dup = int(row['is_duplicate']) if pd.notna(row['is_duplicate']) else 0
            q1_id  = int(row['qid1'])
            q1_text = str(row['question1'])
            q2_id  = int(row['qid2'])
            q2_text = str(row['question2'])
        except Exception:
            continue

        questions_dict[q1_id] = q1_text
        questions_dict[q2_id] = q2_text

        if is_dup == 1:
            qrels.setdefault(q1_id, set()).add(q2_id)
            qrels.setdefault(q2_id, set()).add(q1_id)
            
        count += 1

    corpus_ids   = list(questions_dict.keys())
    corpus_texts = list(questions_dict.values())
    return corpus_ids, corpus_texts, qrels

def build_tfidf(corpus_processed):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus_processed)
    return vectorizer, tfidf_matrix

def build_bm25(corpus_processed):
    tokenized_corpus = [doc.split() for doc in corpus_processed]
    return BM25Okapi(tokenized_corpus)

def build_dense(corpus_texts, batch_size=512):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(
        corpus_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return model, embeddings

def reciprocal_rank_fusion(rankings_list, k=60):
    rrf_scores = {}
    for ranking in rankings_list:
        for rank, doc_id in enumerate(ranking):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return [doc_id for doc_id, _ in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)]

def calculate_metrics(retrieved, relevant, k=10):
    if not relevant:
        return 0.0, 0.0, 0.0, 0.0

    retrieved_k = retrieved[:k]

    # Precision@k
    p_k = len(set(retrieved_k).intersection(relevant)) / k

    # Recall
    recall = len(set(retrieved).intersection(relevant)) / len(relevant)

    # MAP
    ap, hits = 0.0, 0
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            hits += 1
            ap += hits / (i + 1)
    ap /= len(relevant)

    # nDCG
    dcg  = sum(1.0 / np.log2(i + 2) for i, d in enumerate(retrieved_k) if d in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return p_k, recall, ap, ndcg


class FastBM25:
    def __init__(self, meta_path, tf_path):
        import pickle
        from scipy import sparse
        import numpy as np
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.vocabulary = meta["vocabulary"]
        self.idf_array = meta["idf_array"]
        self.doc_len = meta["doc_len"]
        self.avgdl = meta["avgdl"]
        self.tf_matrix = sparse.load_npz(tf_path)
        self.k1 = 1.5
        self.b = 0.75
        self.doc_len_norm = self.doc_len / self.avgdl
        
    def get_scores(self, query_tokens):
        import numpy as np
        scores = np.zeros(len(self.doc_len), dtype=np.float32)
        for term in query_tokens:
            if term in self.vocabulary:
                term_idx = self.vocabulary[term]
                col = self.tf_matrix[:, term_idx]
                doc_indices = col.indices
                freqs = col.data
                
                idf = self.idf_array[term_idx]
                
                num = freqs * (self.k1 + 1)
                den = freqs + self.k1 * (1 - self.b + self.b * self.doc_len_norm[doc_indices])
                
                scores[doc_indices] += idf * (num / den)
        return scores
