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
    
    query = st.text_input("Enter a question to find duplicates:", "top marvel movies")
    
    if st.button("Search"):
        start_time = time.time()
        
        # Preprocess query
        processed_query = utils.preprocess_text(query)
        
        # 1. TF-IDF
        query_tfidf = pipeline["tfidf_vec"].transform([processed_query])
        tfidf_scores_all = cosine_similarity(query_tfidf, pipeline["tfidf_mat"]).flatten()
        tfidf_sorted_idx = tfidf_scores_all.argsort()[::-1][:100]
        tfidf_ranked_ids = [corpus_ids[i] for i in tfidf_sorted_idx]
        tfidf_score_map = {corpus_ids[i]: float(tfidf_scores_all[i]) for i in tfidf_sorted_idx}
        
        # 2. BM25
        tokenized_query = processed_query.split()
        bm25_scores_all = pipeline["bm25_model"].get_scores(tokenized_query)
        bm25_sorted_idx = np.argsort(bm25_scores_all)[::-1][:100]
        bm25_ranked_ids = [corpus_ids[i] for i in bm25_sorted_idx]
        bm25_max = float(bm25_scores_all[bm25_sorted_idx[0]]) if bm25_scores_all[bm25_sorted_idx[0]] > 0 else 1.0
        bm25_score_map = {corpus_ids[i]: float(bm25_scores_all[i]) / bm25_max for i in bm25_sorted_idx}
        
        # 3. Dense
        query_dense = pipeline["dense_model"].encode([query], show_progress_bar=False)
        dense_scores_all = cosine_similarity(query_dense, pipeline["dense_mat"]).flatten()
        dense_sorted_idx = dense_scores_all.argsort()[::-1][:100]
        dense_ranked_ids = [corpus_ids[i] for i in dense_sorted_idx]
        dense_score_map = {corpus_ids[i]: float(dense_scores_all[i]) for i in dense_sorted_idx}
        
        # 4. RRF
        rrf_ranked_ids = utils.reciprocal_rank_fusion([tfidf_ranked_ids, bm25_ranked_ids, dense_ranked_ids])[:100]
        # RRF score: average of normalized scores from all models
        rrf_score_map = {}
        for doc_id in rrf_ranked_ids:
            scores = [
                tfidf_score_map.get(doc_id, 0),
                bm25_score_map.get(doc_id, 0),
                dense_score_map.get(doc_id, 0)
            ]
            rrf_score_map[doc_id] = float(np.mean(scores))
        
        elapsed = time.time() - start_time
        st.success(f"Search completed in {elapsed:.2f} seconds.")
        
        col1, col2, col3, col4 = st.columns(4)
        
        def score_color(score):
            """Return green for high scores, yellow for mid, red for low."""
            if score >= 0.6:
                return "🟢"
            elif score >= 0.3:
                return "🟡"
            else:
                return "🔴"

        def display_results(col, title, ranked_ids, score_map, top_k=10):
            col.markdown(f"### {title}")
            for rank, doc_id in enumerate(ranked_ids[:top_k]):
                score = score_map.get(doc_id, 0.0)
                emoji = score_color(score)
                col.info(f"**{rank+1}.** {id_to_text[doc_id]}\n\n{emoji} Score: `{score:.3f}`")
                
        display_results(col1, "TF-IDF", tfidf_ranked_ids, tfidf_score_map)
        display_results(col2, "BM25", bm25_ranked_ids, bm25_score_map)
        display_results(col3, "Dense (Sentence-BERT)", dense_ranked_ids, dense_score_map)
        display_results(col4, "Hybrid RRF", rrf_ranked_ids, rrf_score_map)

with tab2:
    st.subheader("System Evaluation (Precision@10, Recall, MAP, nDCG)")
    st.write("Evaluates the retrieval systems on queries that have known duplicates.")
    
    if st.button("Run Evaluation on Sample Queries"):
        with st.spinner("Evaluating models... this may take a minute."):
            valid_queries = [qid for qid, rel in pipeline["qrels"].items() if len(rel) > 0]
            sample_queries = valid_queries[:50]
            
            results = {"TF-IDF": [], "BM25": [], "Dense": [], "Hybrid RRF": []}
            # Store per-query precision at each recall level for PR curves
            pr_data = {"TF-IDF": [], "BM25": [], "Dense": [], "Hybrid RRF": []}
            
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
                
                # Collect precision-recall curve data per query
                for model_name, ranked in [("TF-IDF", tfidf_ranked), ("BM25", bm25_ranked),
                                            ("Dense", dense_ranked), ("Hybrid RRF", rrf_ranked)]:
                    prec_pts, rec_pts = [], []
                    hits = 0
                    for k, doc_id in enumerate(ranked):
                        if doc_id in relevant:
                            hits += 1
                        prec_pts.append(hits / (k + 1))
                        rec_pts.append(hits / len(relevant) if relevant else 0)
                    pr_data[model_name].append((rec_pts, prec_pts))
                
                progress_bar.progress((idx + 1) / len(sample_queries))
                
            # Summary table
            summary = []
            for model_name, metrics in results.items():
                summary.append({
                    "Model": model_name,
                    "Precision@10": round(np.mean([m[0] for m in metrics]), 4),
                    "Recall": round(np.mean([m[1] for m in metrics]), 4),
                    "MAP": round(np.mean([m[2] for m in metrics]), 4),
                    "nDCG": round(np.mean([m[3] for m in metrics]), 4)
                })
                
            df_summary = pd.DataFrame(summary).set_index("Model")
            st.dataframe(df_summary.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
            
            # Bar chart
            st.subheader("Performance Comparison")
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            df_summary.plot(kind='bar', ax=ax1, width=0.75, colormap='Set2')
            ax1.set_title("IR Models Performance Comparison")
            ax1.set_ylabel("Score")
            ax1.set_ylim(0, 1)
            ax1.set_xticklabels(df_summary.index, rotation=20, ha='right')
            ax1.legend(loc='upper right')
            plt.tight_layout()
            st.pyplot(fig1)
            
            # Precision-Recall Curves
            st.subheader("Precision-Recall Curves")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            colors = {"TF-IDF": "#e74c3c", "BM25": "#3498db", "Dense": "#2ecc71", "Hybrid RRF": "#9b59b6"}
            
            recall_thresholds = np.linspace(0, 1, 50)
            for model_name, query_prs in pr_data.items():
                interp_precisions = []
                for rec_pts, prec_pts in query_prs:
                    if not rec_pts:
                        continue
                    # Interpolate precision at fixed recall thresholds
                    interp = []
                    for r in recall_thresholds:
                        above = [p for rec, p in zip(rec_pts, prec_pts) if rec >= r]
                        interp.append(max(above) if above else 0.0)
                    interp_precisions.append(interp)
                
                if interp_precisions:
                    mean_prec = np.mean(interp_precisions, axis=0)
                    ax2.plot(recall_thresholds, mean_prec, label=model_name, color=colors[model_name], linewidth=2.5)
            
            ax2.set_xlabel("Recall", fontsize=12)
            ax2.set_ylabel("Precision", fontsize=12)
            ax2.set_title("Precision-Recall Curves (averaged over 50 queries)", fontsize=13)
            ax2.legend(fontsize=11)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2)
