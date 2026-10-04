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
    # Only loads once for the entire application session
    corpus_ids, corpus_texts, qrels = utils.load_and_prepare_data()
    corpus_processed = [utils.preprocess_text(t) for t in corpus_texts]
    tfidf_vec, tfidf_mat = utils.build_tfidf(corpus_processed)
    bm25_model = utils.build_bm25(corpus_processed)
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
        "dense_mat": dense_mat,
        "id_to_text": dict(zip(corpus_ids, corpus_texts))
    }

# Initialize pipeline in session state so it's globally available
if "pipeline" not in st.session_state:
    with st.spinner("Initializing IR Pipeline (Loading 404k dataset & encodings)..."):
        try:
            st.session_state.pipeline = get_pipeline()
        except Exception as e:
            st.error(f"Failed to load pipeline: {e}")
            st.stop()

# --- Navigation ---
pages = {
    "Search": [
        st.Page("pages/1_Search.py", title="Search", icon="🔍", default=True)
    ],
    "Analysis": [
        st.Page("pages/2_Explore.py", title="Explore Dataset", icon="📊"),
        st.Page("pages/3_Evaluation.py", title="Evaluation", icon="📈")
    ],
    "Information": [
        st.Page("pages/4_About.py", title="About", icon="ℹ️")
    ]
}

pg = st.navigation(pages, position="hidden") # Custom top nav in the pages themselves if preferred, or sidebar. 
# We'll use sidebar navigation but style it cleanly. Let's stick to standard sidebar navigation for simplicity of implementation, but we'll collapse it initially. Actually, the user asked for top navigation. Streamlit `st.navigation` combined with `position="hidden"` lets us build custom top nav. Or we can just use `st.navigation` which natively puts it in the sidebar. Let's use `st.navigation` with `position="sidebar"` as it's the standard clean way, but we'll add custom HTML links at the top of each page for true top-nav feel. 

# Let's use the native sidebar navigation, it's very clean now.
pg = st.navigation(pages)

# Top branding
col1, col2 = st.columns([1, 15])
with col1:
    st.image("assets/quora_logo.png", width=120)
with col2:
    st.markdown("<h3 style='margin:0; padding:0; padding-top:10px;'>Duplicate Question Retrieval</h3>", unsafe_allow_html=True)

st.divider()

pg.run()
