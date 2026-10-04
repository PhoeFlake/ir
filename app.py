import streamlit as st
import utils
import base64

# Must be the first Streamlit command
st.set_page_config(
    page_title="Quora Duplicate Question Retrieval",
    page_icon="assets/favicon.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load Custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# --- Pipeline Loading ---
@st.cache_resource(show_spinner=False)
def get_pipeline():
    import pickle
    import numpy as np
    from scipy import sparse
    import os
    from sentence_transformers import SentenceTransformer
    
    # Check LFS
    size = os.path.getsize("indexes/dense_embeddings.npy")
    if size < 1000:
        st.error(f"Git LFS pointer detected! File size is {size} bytes. Streamlit didn't pull LFS files properly.")
        st.stop()
        
    corpus_ids = np.load("indexes/corpus_ids.npy")
    
    with open("indexes/tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vec = pickle.load(f)
    tfidf_mat = sparse.load_npz("indexes/tfidf_matrix.npz")
    
    bm25_model = utils.FastBM25("indexes/fast_bm25_meta.pkl", "indexes/fast_bm25_tf.npz")
    
    dense_model = SentenceTransformer('all-MiniLM-L6-v2')
    dense_mat = np.load("indexes/dense_embeddings.npy", mmap_mode='r')
    
    return {
        "corpus_ids": corpus_ids,
        "tfidf_vec": tfidf_vec,
        "tfidf_mat": tfidf_mat,
        "bm25_model": bm25_model,
        "dense_model": dense_model,
        "dense_mat": dense_mat
    }

def get_db():
    import sqlite3
    return sqlite3.connect("indexes/metadata.db", check_same_thread=False)

if "pipeline" not in st.session_state:
    with st.spinner("Initializing IR Pipeline (Loading disk-backed indexes)..."):
        try:
            st.session_state.pipeline = get_pipeline()
            st.session_state.db = get_db()
        except Exception as e:
            st.error(f"Failed to load pipeline: {e}")
            st.stop()

# --- Navigation ---
pages = {
    "Search": [
        st.Page("pages/1_Search.py", title="Search", default=True)
    ],
    "Analysis": [
        st.Page("pages/2_Explore.py", title="Explore Dataset"),
        st.Page("pages/3_Evaluation.py", title="Evaluation")
    ],
    "Report": [
        st.Page("pages/4_About.py", title="Report")
    ]
}

pg = st.navigation(pages)

st.logo("assets/quora_logo.png")

pg.run()
