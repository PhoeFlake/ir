import streamlit as st
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import utils

db = st.session_state.db
def get_text(qid):
    return db.execute("SELECT text FROM questions WHERE qid=?", (int(qid),)).fetchone()[0]

pipeline = st.session_state.pipeline
corpus_ids = pipeline["corpus_ids"]

st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h1>Find questions that mean the same thing.</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown('<div class="search-pill-container">', unsafe_allow_html=True)
query = st.text_input(
    "Search Query",
    placeholder="How do I learn Python by myself?",
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    search_clicked = st.button("Search", use_container_width=True, type="primary")

st.markdown("<br>", unsafe_allow_html=True)

if search_clicked and query.strip():
    t0 = time.time()
    proc_q = utils.preprocess_text(query)

    # Retrieval
    q_tfidf = pipeline["tfidf_vec"].transform([proc_q])
    tfidf_raw = cosine_similarity(q_tfidf, pipeline["tfidf_mat"]).flatten()
    tfidf_idx = tfidf_raw.argsort()[::-1][:100]
    tfidf_ids = [corpus_ids[i] for i in tfidf_idx]
    tfidf_smap = {corpus_ids[i]: float(tfidf_raw[i]) for i in tfidf_idx}

    bm25_raw = pipeline["bm25_model"].get_scores(proc_q.split())
    bm25_idx = np.argsort(bm25_raw)[::-1][:100]
    bm25_ids = [corpus_ids[i] for i in bm25_idx]
    bm25_top = float(bm25_raw[bm25_idx[0]]) if bm25_raw[bm25_idx[0]] > 0 else 1.0
    bm25_smap = {corpus_ids[i]: float(bm25_raw[i]) / bm25_top for i in bm25_idx}

    q_dense = pipeline["dense_model"].encode([query], show_progress_bar=False)
    dense_raw = np.dot(q_dense, pipeline["dense_mat"].T).flatten()
    dense_idx = dense_raw.argsort()[::-1][:100]
    dense_ids = [corpus_ids[i] for i in dense_idx]
    dense_smap = {corpus_ids[i]: float(dense_raw[i]) for i in dense_idx}

    # Hybrid RRF (Primary ranking used for display)
    rrf_ids = utils.reciprocal_rank_fusion([tfidf_ids, bm25_ids, dense_ids])[:20]
    
    elapsed = time.time() - t0

    st.markdown(f"<p class='secondary-text'>Retrieved from 404,351 candidates in {elapsed:.2f}s using Hybrid RRF</p>", unsafe_allow_html=True)
    st.divider()

    for rank, doc_id in enumerate(rrf_ids):
        # Calculate scores
        dense_score = dense_smap.get(doc_id, 0.0)
        tf_score = tfidf_smap.get(doc_id, 0.0)
        bm_score = bm25_smap.get(doc_id, 0.0)
        
        doc_text = get_text(doc_id)
        # Determine overlap
        doc_tokens = set(utils.preprocess_text(doc_text).split())
        query_tokens = set(proc_q.split())
        overlap = len(doc_tokens.intersection(query_tokens)) / max(len(query_tokens), 1)

        st.markdown(f"""
            <div class='result-card'>
                <div class='result-card-header'>
                    <span class='result-card-score'>{dense_score*100:.1f}% semantic similarity</span>
                    <span class='result-card-sub'>Rank #{rank+1}</span>
                </div>
                <div class='result-card-question'>{doc_text}</div>
                <div class='result-card-sub'>This question is semantically similar to your query.</div>
            </div>
        """, unsafe_allow_html=True)
        
        with st.expander("Why this result?"):
            st.markdown(f"""
            - **Semantic similarity (Dense)**: {dense_score:.3f}
            - **Lexical overlap (TF-IDF)**: {tf_score:.3f}
            - **BM25 Score (Normalized)**: {bm_score:.3f}
            - **Token overlap fraction**: {overlap:.2f}
            - **Internal Document ID**: `{doc_id}`
            """)
        st.markdown("<br>", unsafe_allow_html=True)

elif search_clicked:
    st.warning("Please enter a question.")
