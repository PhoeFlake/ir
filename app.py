import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import utils
import time

st.set_page_config(layout="wide", page_title="Duplicate Question Retrieval", page_icon="🔍")

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Page background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: rgba(255,255,255,0.05); }

/* Main title */
h1 { 
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.6rem !important;
    font-weight: 800 !important;
}

/* Subtitle */
.subtitle {
    color: #94a3b8;
    font-size: 1rem;
    margin-top: -10px;
    margin-bottom: 20px;
}

/* Metric pill badges */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 6px;
    margin-bottom: 8px;
}
.badge-tfidf   { background: #1e3a5f; color: #60a5fa; border: 1px solid #60a5fa44; }
.badge-bm25    { background: #1a3a2a; color: #34d399; border: 1px solid #34d39944; }
.badge-dense   { background: #3b1f5e; color: #c084fc; border: 1px solid #c084fc44; }
.badge-rrf     { background: #3b2a10; color: #fb923c; border: 1px solid #fb923c44; }

/* Result card */
.result-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}
.result-card:hover { border-color: rgba(167,139,250,0.5); }

.rank-badge {
    display: inline-block;
    background: rgba(167,139,250,0.2);
    color: #a78bfa;
    border-radius: 6px;
    padding: 1px 8px;
    font-size: 0.75rem;
    font-weight: 700;
    margin-right: 8px;
}

.result-text {
    color: #e2e8f0;
    font-size: 0.88rem;
    line-height: 1.5;
}

.score-bar-wrap {
    margin-top: 10px;
}
.score-label {
    font-size: 0.72rem;
    color: #94a3b8;
    margin-bottom: 3px;
}
.score-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 999px;
    height: 6px;
    width: 100%;
}
.score-bar-fill {
    height: 6px;
    border-radius: 999px;
}

/* Model header */
.model-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}
.model-dot {
    width: 12px; height: 12px;
    border-radius: 50%;
}
.model-name {
    font-weight: 700;
    font-size: 1rem;
    color: #f1f5f9;
}

/* Stat strip */
.stat-strip {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.stat-value { font-size: 1.6rem; font-weight: 800; color: #a78bfa; }
.stat-label { font-size: 0.75rem; color: #64748b; margin-top: 2px; }

/* Search bar */
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(167,139,250,0.4) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    font-size: 1rem !important;
}

/* Search button */
.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 28px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    width: 100%;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px);
}

/* Tab styling */
[data-testid="stTabs"] [role="tab"] {
    color: #94a3b8 !important;
    font-weight: 600;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #a78bfa !important;
    border-bottom-color: #a78bfa !important;
}

/* Spinner */
[data-testid="stSpinner"] { color: #a78bfa !important; }

/* Success/info boxes */
[data-testid="stAlert"] {
    background: rgba(52,211,153,0.1) !important;
    border: 1px solid rgba(52,211,153,0.3) !important;
    border-radius: 10px !important;
}

/* DataFrames */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("<h1>🔍 Duplicate Question Retrieval</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Information Retrieval Pipeline · TF-IDF · BM25 · Sentence Transformers · Hybrid RRF</p>', unsafe_allow_html=True)

MODEL_COLORS = {
    "TF-IDF": {"dot": "#60a5fa", "bar": "#60a5fa", "badge": "badge-tfidf"},
    "BM25": {"dot": "#34d399", "bar": "#34d399", "badge": "badge-bm25"},
    "Dense (Sentence-BERT)": {"dot": "#c084fc", "bar": "#c084fc", "badge": "badge-dense"},
    "Hybrid RRF": {"dot": "#fb923c", "bar": "#fb923c", "badge": "badge-rrf"},
}

# ── Pipeline loading ─────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    with st.spinner("Loading dataset..."):
        corpus_ids, corpus_texts, qrels = utils.load_and_prepare_data(num_samples=10000)
    with st.spinner("Preprocessing text..."):
        corpus_processed = [utils.preprocess_text(text) for text in corpus_texts]
    with st.spinner("Building TF-IDF index..."):
        tfidf_vec, tfidf_mat = utils.build_tfidf(corpus_processed)
    with st.spinner("Building BM25 index..."):
        bm25_model = utils.build_bm25(corpus_processed)
    with st.spinner("Building dense embeddings..."):
        dense_model, dense_mat = utils.build_dense(corpus_texts)
    return {
        "corpus_ids": corpus_ids, "corpus_texts": corpus_texts,
        "corpus_processed": corpus_processed, "qrels": qrels,
        "tfidf_vec": tfidf_vec, "tfidf_mat": tfidf_mat,
        "bm25_model": bm25_model, "dense_model": dense_model, "dense_mat": dense_mat
    }

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Error loading pipeline: {e}")
    st.stop()

corpus_ids = pipeline["corpus_ids"]
corpus_texts = pipeline["corpus_texts"]
id_to_text = {cid: txt for cid, txt in zip(corpus_ids, corpus_texts)}

# ── Stat strip ───────────────────────────────────────────────────────────────
dup_count = sum(len(v) for v in pipeline["qrels"].values())
c1, c2, c3, c4 = st.columns(4)
for col, val, label in [
    (c1, f"{len(corpus_ids):,}", "Unique Questions"),
    (c2, "4", "IR Models"),
    (c3, f"{dup_count:,}", "Duplicate Pairs"),
    (c4, "10K", "Samples Loaded"),
]:
    col.markdown(f"""
    <div class="stat-strip">
        <div class="stat-value">{val}</div>
        <div class="stat-label">{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔎 Search Interface", "📊 Evaluation & Metrics"])

# ── Search Tab ───────────────────────────────────────────────────────────────
with tab1:
    qcol, bcol = st.columns([5, 1])
    with qcol:
        query = st.text_input("", placeholder="Enter a question to find duplicates...", value="top marvel movies", label_visibility="collapsed")
    with bcol:
        search_clicked = st.button("Search", use_container_width=True)

    if search_clicked:
        start_time = time.time()
        processed_query = utils.preprocess_text(query)

        # TF-IDF
        q_tfidf = pipeline["tfidf_vec"].transform([processed_query])
        tfidf_scores_all = cosine_similarity(q_tfidf, pipeline["tfidf_mat"]).flatten()
        tfidf_idx = tfidf_scores_all.argsort()[::-1][:100]
        tfidf_ranked = [corpus_ids[i] for i in tfidf_idx]
        tfidf_score_map = {corpus_ids[i]: float(tfidf_scores_all[i]) for i in tfidf_idx}

        # BM25
        bm25_scores_all = pipeline["bm25_model"].get_scores(processed_query.split())
        bm25_idx = np.argsort(bm25_scores_all)[::-1][:100]
        bm25_ranked = [corpus_ids[i] for i in bm25_idx]
        bm25_max = float(bm25_scores_all[bm25_idx[0]]) if bm25_scores_all[bm25_idx[0]] > 0 else 1.0
        bm25_score_map = {corpus_ids[i]: float(bm25_scores_all[i]) / bm25_max for i in bm25_idx}

        # Dense
        q_dense = pipeline["dense_model"].encode([query], show_progress_bar=False)
        dense_scores_all = cosine_similarity(q_dense, pipeline["dense_mat"]).flatten()
        dense_idx = dense_scores_all.argsort()[::-1][:100]
        dense_ranked = [corpus_ids[i] for i in dense_idx]
        dense_score_map = {corpus_ids[i]: float(dense_scores_all[i]) for i in dense_idx}

        # RRF
        rrf_ranked = utils.reciprocal_rank_fusion([tfidf_ranked, bm25_ranked, dense_ranked])[:100]
        rrf_score_map = {
            doc_id: float(np.mean([
                tfidf_score_map.get(doc_id, 0),
                bm25_score_map.get(doc_id, 0),
                dense_score_map.get(doc_id, 0)
            ])) for doc_id in rrf_ranked
        }

        elapsed = time.time() - start_time
        st.success(f"✅ Search completed in **{elapsed:.2f}s** across {len(corpus_ids):,} questions")

        models = [
            ("TF-IDF", tfidf_ranked, tfidf_score_map),
            ("BM25", bm25_ranked, bm25_score_map),
            ("Dense (Sentence-BERT)", dense_ranked, dense_score_map),
            ("Hybrid RRF", rrf_ranked, rrf_score_map),
        ]

        cols = st.columns(4)
        for col, (model_name, ranked_ids, score_map) in zip(cols, models):
            color = MODEL_COLORS[model_name]
            with col:
                st.markdown(f"""
                <div class="model-header">
                    <div class="model-dot" style="background:{color['dot']}"></div>
                    <span class="model-name">{model_name}</span>
                </div>""", unsafe_allow_html=True)

                for rank, doc_id in enumerate(ranked_ids[:10]):
                    score = score_map.get(doc_id, 0.0)
                    bar_width = int(score * 100)
                    st.markdown(f"""
                    <div class="result-card">
                        <span class="rank-badge">#{rank+1}</span>
                        <span class="result-text">{id_to_text[doc_id]}</span>
                        <div class="score-bar-wrap">
                            <div class="score-label">Similarity score: <b style="color:{color['bar']}">{score:.3f}</b></div>
                            <div class="score-bar-bg">
                                <div class="score-bar-fill" style="width:{bar_width}%; background:{color['bar']}"></div>
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)

# ── Evaluation Tab ────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### System Evaluation")
    st.markdown('<p class="subtitle">Measures Precision@10, Recall, MAP, and nDCG across 50 sample queries with known duplicates.</p>', unsafe_allow_html=True)

    if st.button("▶ Run Evaluation", use_container_width=False):
        with st.spinner("Evaluating all models... this may take a minute."):
            valid_queries = [qid for qid, rel in pipeline["qrels"].items() if len(rel) > 0]
            sample_queries = valid_queries[:50]
            results = {"TF-IDF": [], "BM25": [], "Dense": [], "Hybrid RRF": []}
            pr_data = {"TF-IDF": [], "BM25": [], "Dense": [], "Hybrid RRF": []}

            progress_bar = st.progress(0)
            for idx, qid in enumerate(sample_queries):
                q_text = id_to_text[qid]
                proc_q = utils.preprocess_text(q_text)
                relevant = pipeline["qrels"][qid]

                q_tfidf = pipeline["tfidf_vec"].transform([proc_q])
                tfidf_sc = cosine_similarity(q_tfidf, pipeline["tfidf_mat"]).flatten()
                tfidf_r = [corpus_ids[i] for i in tfidf_sc.argsort()[::-1][:100]]

                bm25_sc = pipeline["bm25_model"].get_scores(proc_q.split())
                bm25_r = [corpus_ids[i] for i in np.argsort(bm25_sc)[::-1][:100]]

                q_dense = pipeline["dense_model"].encode([q_text], show_progress_bar=False)
                dense_sc = cosine_similarity(q_dense, pipeline["dense_mat"]).flatten()
                dense_r = [corpus_ids[i] for i in dense_sc.argsort()[::-1][:100]]

                rrf_r = utils.reciprocal_rank_fusion([tfidf_r, bm25_r, dense_r])[:100]

                results["TF-IDF"].append(utils.calculate_metrics(tfidf_r, relevant))
                results["BM25"].append(utils.calculate_metrics(bm25_r, relevant))
                results["Dense"].append(utils.calculate_metrics(dense_r, relevant))
                results["Hybrid RRF"].append(utils.calculate_metrics(rrf_r, relevant))

                for mname, ranked in [("TF-IDF", tfidf_r), ("BM25", bm25_r), ("Dense", dense_r), ("Hybrid RRF", rrf_r)]:
                    prec_pts, rec_pts = [], []
                    hits = 0
                    for k, doc_id in enumerate(ranked):
                        if doc_id in relevant:
                            hits += 1
                        prec_pts.append(hits / (k + 1))
                        rec_pts.append(hits / len(relevant) if relevant else 0)
                    pr_data[mname].append((rec_pts, prec_pts))

                progress_bar.progress((idx + 1) / len(sample_queries))

        # Summary table
        summary = []
        for model_name, metrics in results.items():
            summary.append({
                "Model": model_name,
                "Precision@10": round(np.mean([m[0] for m in metrics]), 4),
                "Recall": round(np.mean([m[1] for m in metrics]), 4),
                "MAP": round(np.mean([m[2] for m in metrics]), 4),
                "nDCG": round(np.mean([m[3] for m in metrics]), 4),
            })

        df_summary = pd.DataFrame(summary).set_index("Model")
        st.dataframe(df_summary.style.highlight_max(axis=0, color="#1a3a2a").format("{:.4f}"), use_container_width=True)

        chart_col, pr_col = st.columns(2)

        with chart_col:
            st.markdown("#### Performance Comparison")
            fig1, ax1 = plt.subplots(figsize=(7, 4))
            fig1.patch.set_facecolor("#0f0c29")
            ax1.set_facecolor("#1a1a2e")
            bar_colors = ["#60a5fa", "#34d399", "#c084fc", "#fb923c"]
            x = np.arange(len(df_summary.columns))
            width = 0.18
            for i, (idx_label, row) in enumerate(df_summary.iterrows()):
                ax1.bar(x + i * width, row.values, width, label=idx_label, color=bar_colors[i], alpha=0.9)
            ax1.set_xticks(x + width * 1.5)
            ax1.set_xticklabels(df_summary.columns, color="#94a3b8", fontsize=9)
            ax1.set_ylim(0, 1)
            ax1.tick_params(colors="#94a3b8")
            ax1.spines[:].set_color("#2d2d44")
            ax1.legend(fontsize=8, labelcolor="white", facecolor="#1a1a2e")
            ax1.set_ylabel("Score", color="#94a3b8")
            plt.tight_layout()
            st.pyplot(fig1)

        with pr_col:
            st.markdown("#### Precision-Recall Curves")
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            fig2.patch.set_facecolor("#0f0c29")
            ax2.set_facecolor("#1a1a2e")
            recall_thresholds = np.linspace(0, 1, 50)
            pr_colors = {"TF-IDF": "#60a5fa", "BM25": "#34d399", "Dense": "#c084fc", "Hybrid RRF": "#fb923c"}
            for mname, query_prs in pr_data.items():
                interp_precs = []
                for rec_pts, prec_pts in query_prs:
                    if not rec_pts:
                        continue
                    interp = [max((p for rec, p in zip(rec_pts, prec_pts) if rec >= r), default=0.0) for r in recall_thresholds]
                    interp_precs.append(interp)
                if interp_precs:
                    mean_prec = np.mean(interp_precs, axis=0)
                    ax2.plot(recall_thresholds, mean_prec, label=mname, color=pr_colors[mname], linewidth=2.5)
            ax2.set_xlabel("Recall", color="#94a3b8")
            ax2.set_ylabel("Precision", color="#94a3b8")
            ax2.tick_params(colors="#94a3b8")
            ax2.spines[:].set_color("#2d2d44")
            ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.15, color="white")
            ax2.legend(fontsize=8, labelcolor="white", facecolor="#1a1a2e")
            plt.tight_layout()
            st.pyplot(fig2)
