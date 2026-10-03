import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import utils
import time

st.set_page_config(layout="wide", page_title="Duplicate Question Retrieval")

st.title("Duplicate Question Retrieval")
st.markdown("Information Retrieval Pipeline with TF-IDF, BM25, Sentence Transformers, and Hybrid RRF.")

@st.cache_resource
def load_pipeline():
    with st.spinner("Loading Quora dataset... (Using 10,000 samples for rapid demo)"):
        corpus_ids, corpus_texts, qrels = utils.load_and_prepare_data(num_samples=10000)
        
    with st.spinner("Preprocessing text (Cleaning, Tokenization, Stemming)..."):
        corpus_processed = [utils.preprocess_text(text) for text in corpus_texts]
        
    with st.spinner("Building TF-IDF Index..."):
        tfidf_vec, tfidf_mat = utils.build_tfidf(corpus_processed)
        
    with st.spinner("Building BM25 Index..."):
        bm25_model = utils.build_bm25(corpus_processed)
        
    with st.spinner("Building Dense Embeddings (SentenceTransformer)..."):
        dense_model, dense_mat = utils.build_dense(corpus_texts)
        
    return {
        "corpus_ids": corpus_ids,
        "corpus_texts": corpus_texts,
        "corpus_processed": corpus_processed,
        "qrels": qrels,
        "tfidf_vec": tfidf_vec,
        "tfidf_mat": tfidf_mat,
        "bm25_model": bm25_model,
        "dense_model": dense_model,
        "dense_mat": dense_mat
    }

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Error loading pipeline: {e}")
    st.stop()

corpus_ids = pipeline["corpus_ids"]
corpus_texts = pipeline["corpus_texts"]
id_to_text = {cid: txt for cid, txt in zip(corpus_ids, corpus_texts)}

tab1, tab2 = st.tabs(["Search Interface", "Evaluation & Metrics"])

with tab1:
    st.subheader("Interactive Search")
    st.write("Find duplicate questions using 4 different ranking paradigms.")
    
    query = st.text_input("Enter a question to find duplicates:", "How can I lose weight fast?")
    
    if st.button("Search"):
        start_time = time.time()
        
        # Preprocess query
        processed_query = utils.preprocess_text(query)
        
        # 1. TF-IDF
        query_tfidf = pipeline["tfidf_vec"].transform([processed_query])
        tfidf_scores = cosine_similarity(query_tfidf, pipeline["tfidf_mat"]).flatten()
        tfidf_ranked_ids = [corpus_ids[i] for i in tfidf_scores.argsort()[::-1][:100]]
        
        # 2. BM25
        tokenized_query = processed_query.split()
        bm25_scores = pipeline["bm25_model"].get_scores(tokenized_query)
        bm25_ranked_ids = [corpus_ids[i] for i in np.argsort(bm25_scores)[::-1][:100]]
        
        # 3. Dense
        query_dense = pipeline["dense_model"].encode([query], show_progress_bar=False)
        dense_scores = cosine_similarity(query_dense, pipeline["dense_mat"]).flatten()
        dense_ranked_ids = [corpus_ids[i] for i in dense_scores.argsort()[::-1][:100]]
        
        # 4. RRF
        rrf_ranked_ids = utils.reciprocal_rank_fusion([tfidf_ranked_ids, bm25_ranked_ids, dense_ranked_ids])[:100]
        
        st.success(f"Search completed in {time.time() - start_time:.2f} seconds.")
        
        col1, col2, col3, col4 = st.columns(4)
        
        def display_results(col, title, ranked_ids, top_k=10):
            col.markdown(f"### {title}")
            for rank, doc_id in enumerate(ranked_ids[:top_k]):
                col.info(f"**{rank+1}.** {id_to_text[doc_id]}")
                
        display_results(col1, "TF-IDF", tfidf_ranked_ids)
        display_results(col2, "BM25", bm25_ranked_ids)
        display_results(col3, "Dense (Sentence-BERT)", dense_ranked_ids)
        display_results(col4, "Hybrid RRF", rrf_ranked_ids)

with tab2:
    st.subheader("System Evaluation (Precision@10, Recall, MAP, nDCG)")
    st.write("Evaluates the retrieval systems on queries that have known duplicates.")
    
    if st.button("Run Evaluation on Sample Queries"):
        with st.spinner("Evaluating models... this may take a minute."):
            valid_queries = [qid for qid, rel in pipeline["qrels"].items() if len(rel) > 0]
            sample_queries = valid_queries[:50]
            
            results = {"TF-IDF": [], "BM25": [], "Dense": [], "Hybrid RRF": []}
            
            progress_bar = st.progress(0)
            for idx, qid in enumerate(sample_queries):
                q_text = id_to_text[qid]
                proc_q = utils.preprocess_text(q_text)
                relevant = pipeline["qrels"][qid]
                
                # TF-IDF
                q_tfidf = pipeline["tfidf_vec"].transform([proc_q])
                tfidf_scores = cosine_similarity(q_tfidf, pipeline["tfidf_mat"]).flatten()
                tfidf_ranked = [corpus_ids[i] for i in tfidf_scores.argsort()[::-1][:100]]
                
                # BM25
                bm25_scores = pipeline["bm25_model"].get_scores(proc_q.split())
                bm25_ranked = [corpus_ids[i] for i in np.argsort(bm25_scores)[::-1][:100]]
                
                # Dense
                q_dense = pipeline["dense_model"].encode([q_text], show_progress_bar=False)
                dense_scores = cosine_similarity(q_dense, pipeline["dense_mat"]).flatten()
                dense_ranked = [corpus_ids[i] for i in dense_scores.argsort()[::-1][:100]]
                
                # RRF
                rrf_ranked = utils.reciprocal_rank_fusion([tfidf_ranked, bm25_ranked, dense_ranked])[:100]
                
                results["TF-IDF"].append(utils.calculate_metrics(tfidf_ranked, relevant))
                results["BM25"].append(utils.calculate_metrics(bm25_ranked, relevant))
                results["Dense"].append(utils.calculate_metrics(dense_ranked, relevant))
                results["Hybrid RRF"].append(utils.calculate_metrics(rrf_ranked, relevant))
                
                progress_bar.progress((idx + 1) / len(sample_queries))
                
            summary = []
            for model_name, metrics in results.items():
                avg_p10 = np.mean([m[0] for m in metrics])
                avg_rec = np.mean([m[1] for m in metrics])
                avg_map = np.mean([m[2] for m in metrics])
                avg_ndcg = np.mean([m[3] for m in metrics])
                summary.append({
                    "Model": model_name,
                    "Precision@10": avg_p10,
                    "Recall": avg_rec,
                    "MAP": avg_map,
                    "nDCG": avg_ndcg
                })
                
            df_summary = pd.DataFrame(summary).set_index("Model")
            st.dataframe(df_summary.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
            
            st.subheader("Performance Graph")
            fig, ax = plt.subplots(figsize=(10, 6))
            df_summary.plot(kind='bar', ax=ax, width=0.8)
            plt.title("IR Models Performance Comparison")
            plt.ylabel("Score")
            plt.xticks(rotation=45)
            plt.legend(loc='upper right')
            plt.tight_layout()
            st.pyplot(fig)
