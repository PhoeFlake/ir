import streamlit as st

st.header("About the System")
st.markdown("""
This project implements an Information Retrieval pipeline for Duplicate Question Detection using the Quora Question Pairs dataset.

### Retrieval Pipeline

1. **User Query**  
   The system accepts a raw natural language question.

2. **Preprocessing**  
   Text is lowercased, stripped of punctuation, tokenized, and stemmed using Porter Stemmer. Stop words are removed to reduce noise.

3. **Query Representation**  
   The query is encoded in three parallel streams:
   - Sparse lexical vector (TF-IDF)
   - Probabilistic matching terms (BM25)
   - Dense semantic embeddings (Sentence-BERT `all-MiniLM-L6-v2`)

4. **Candidate Retrieval**  
   Each model independently scores and retrieves the top 100 matching questions from the 404,351 question corpus.

5. **Similarity Calculation**  
   - TF-IDF and Dense embeddings use Cosine Similarity.
   - BM25 uses its native probabilistic scoring.

6. **Ranking (Hybrid RRF)**  
   The individual rankings are combined using Reciprocal Rank Fusion (RRF). RRF penalizes documents that only do well in one model but rewards documents that consistently rank highly across semantic and lexical searches.

7. **Duplicate Detection**  
   The final ordered list represents the highest probability duplicates.

### Dataset Overview
- **Source**: Quora Question Pairs
- **Size**: 404,351 question pairs
- **Labels**: Binary (`is_duplicate`) indicating if the two questions carry the exact same intent.

### Limitations
- **Latency**: Brute-force cosine similarity over 404k dense vectors takes ~0.5s. In production, this would be indexed using FAISS or HNSW for sub-millisecond latency.
- **Asymmetry**: True duplicate detection often involves symmetric relationships, while this system frames it as a directed retrieval task.
""")
