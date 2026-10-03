import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import utils
import time

st.set_page_config(layout="wide", page_title="Duplicate Question Retrieval")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Duplicate Question Retrieval")
st.caption("Information Retrieval Pipeline  ·  TF-IDF  ·  BM25  ·  Sentence Transformers  ·  Hybrid RRF")
st.divider()

# ── Load pipeline ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    steps = [
        ("Loading full Quora dataset (404k pairs)…",
         lambda: utils.load_and_prepare_data()),
        ("Preprocessing text…",
         None),
        ("Building TF-IDF index…",
         None),
        ("Building BM25 index…",
         None),
        ("Encoding dense embeddings (this takes a while on first run)…",
         None),
    ]

    with st.status("Initialising pipeline — please wait…", expanded=True) as status:
        st.write(steps[0][0])
        corpus_ids, corpus_texts, qrels = utils.load_and_prepare_data()

        st.write("Preprocessing text…")
        corpus_processed = [utils.preprocess_text(t) for t in corpus_texts]

        st.write("Building TF-IDF index…")
        tfidf_vec, tfidf_mat = utils.build_tfidf(corpus_processed)

        st.write("Building BM25 index…")
        bm25_model = utils.build_bm25(corpus_processed)

        st.write("Encoding dense embeddings (this takes a while on first run)…")
        dense_model, dense_mat = utils.build_dense(corpus_texts)

        status.update(label="Pipeline ready!", state="complete", expanded=False)

    return dict(
        corpus_ids=corpus_ids,
        corpus_texts=corpus_texts,
        corpus_processed=corpus_processed,
        qrels=qrels,
        tfidf_vec=tfidf_vec,
        tfidf_mat=tfidf_mat,
        bm25_model=bm25_model,
        dense_model=dense_model,
        dense_mat=dense_mat,
    )

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Error loading pipeline: {e}")
    st.stop()

corpus_ids   = pipeline["corpus_ids"]
corpus_texts = pipeline["corpus_texts"]
id_to_text   = dict(zip(corpus_ids, corpus_texts))

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Search", "Evaluation & Metrics"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    query = st.text_input(
        "Enter a question to find duplicates:",
        placeholder="e.g. How do I start investing in stocks?",
    )
    top_k = st.slider("Results per model", min_value=5, max_value=20, value=10, step=5)

    if st.button("Search", type="primary", disabled=not query.strip()):
        if not query.strip():
            st.warning("Please enter a question first.")
        else:
            t0 = time.time()
            proc_q = utils.preprocess_text(query)

            # TF-IDF
            q_tfidf       = pipeline["tfidf_vec"].transform([proc_q])
            tfidf_raw     = cosine_similarity(q_tfidf, pipeline["tfidf_mat"]).flatten()
            tfidf_idx     = tfidf_raw.argsort()[::-1][:100]
            tfidf_ids     = [corpus_ids[i] for i in tfidf_idx]
            tfidf_smap    = {corpus_ids[i]: float(tfidf_raw[i]) for i in tfidf_idx}

            # BM25
            bm25_raw      = pipeline["bm25_model"].get_scores(proc_q.split())
            bm25_idx      = np.argsort(bm25_raw)[::-1][:100]
            bm25_ids      = [corpus_ids[i] for i in bm25_idx]
            bm25_top      = float(bm25_raw[bm25_idx[0]]) if bm25_raw[bm25_idx[0]] > 0 else 1.0
            bm25_smap     = {corpus_ids[i]: float(bm25_raw[i]) / bm25_top for i in bm25_idx}

            # Dense
            q_dense       = pipeline["dense_model"].encode([query], show_progress_bar=False)
            dense_raw     = cosine_similarity(q_dense, pipeline["dense_mat"]).flatten()
            dense_idx     = dense_raw.argsort()[::-1][:100]
            dense_ids     = [corpus_ids[i] for i in dense_idx]
            dense_smap    = {corpus_ids[i]: float(dense_raw[i]) for i in dense_idx}

            # Hybrid RRF
            rrf_ids       = utils.reciprocal_rank_fusion([tfidf_ids, bm25_ids, dense_ids])[:100]
            rrf_smap      = {d: float(np.mean([tfidf_smap.get(d, 0),
                                                bm25_smap.get(d, 0),
                                                dense_smap.get(d, 0)])) for d in rrf_ids}

            elapsed = time.time() - t0
            st.success(f"Completed in **{elapsed:.2f}s**  ·  searching across **{len(corpus_ids):,}** unique questions")

            def score_badge(score):
                if score >= 0.6:   return "🟢"
                elif score >= 0.3: return "🟡"
                else:              return "🔴"

            def render_col(col, title, ids, smap, k):
                col.markdown(f"#### {title}")
                col.divider()
                for rank, doc_id in enumerate(ids[:k]):
                    score = smap.get(doc_id, 0.0)
                    col.markdown(
                        f"**{rank+1}.** {id_to_text[doc_id]}  \n"
                        f"{score_badge(score)} `{score:.3f}`"
                    )
                    col.divider()

            c1, c2, c3, c4 = st.columns(4)
            render_col(c1, "TF-IDF",            tfidf_ids, tfidf_smap, top_k)
            render_col(c2, "BM25",              bm25_ids,  bm25_smap,  top_k)
            render_col(c3, "Dense (SBERT)",     dense_ids, dense_smap, top_k)
            render_col(c4, "Hybrid RRF",        rrf_ids,   rrf_smap,   top_k)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Evaluation")
    st.write("Runs all 4 models over a sample of queries that have known duplicate labels and reports IR metrics.")

    n_eval = st.slider("Number of evaluation queries", 20, 200, 50, step=10)

    if st.button("Run Evaluation", type="primary"):
        with st.spinner("Evaluating…"):
            valid_qids   = [q for q, r in pipeline["qrels"].items() if r]
            sample_qids  = valid_qids[:n_eval]

            res    = {m: [] for m in ["TF-IDF", "BM25", "Dense", "Hybrid RRF"]}
            pr_raw = {m: [] for m in ["TF-IDF", "BM25", "Dense", "Hybrid RRF"]}
            bar    = st.progress(0)

            for idx, qid in enumerate(sample_qids):
                q_txt  = id_to_text[qid]
                proc_q = utils.preprocess_text(q_txt)
                rel    = pipeline["qrels"][qid]

                q_tf   = pipeline["tfidf_vec"].transform([proc_q])
                tf_s   = cosine_similarity(q_tf, pipeline["tfidf_mat"]).flatten()
                tf_r   = [corpus_ids[i] for i in tf_s.argsort()[::-1][:100]]

                bm_s   = pipeline["bm25_model"].get_scores(proc_q.split())
                bm_r   = [corpus_ids[i] for i in np.argsort(bm_s)[::-1][:100]]

                q_de   = pipeline["dense_model"].encode([q_txt], show_progress_bar=False)
                de_s   = cosine_similarity(q_de, pipeline["dense_mat"]).flatten()
                de_r   = [corpus_ids[i] for i in de_s.argsort()[::-1][:100]]

                rr_r   = utils.reciprocal_rank_fusion([tf_r, bm_r, de_r])[:100]

                for name, ranked in [("TF-IDF", tf_r), ("BM25", bm_r),
                                     ("Dense", de_r), ("Hybrid RRF", rr_r)]:
                    res[name].append(utils.calculate_metrics(ranked, rel))

                    # PR curve data
                    hits, prec_pts, rec_pts = 0, [], []
                    for k, d in enumerate(ranked):
                        if d in rel:
                            hits += 1
                        prec_pts.append(hits / (k + 1))
                        rec_pts.append(hits / len(rel) if rel else 0)
                    pr_raw[name].append((rec_pts, prec_pts))

                bar.progress((idx + 1) / len(sample_qids))

        # ── Summary table ────────────────────────────────────────────────────
        summary = []
        for name, metrics in res.items():
            summary.append({
                "Model":        name,
                "Precision@10": round(np.mean([m[0] for m in metrics]), 4),
                "Recall":       round(np.mean([m[1] for m in metrics]), 4),
                "MAP":          round(np.mean([m[2] for m in metrics]), 4),
                "nDCG":         round(np.mean([m[3] for m in metrics]), 4),
            })
        df = pd.DataFrame(summary).set_index("Model")
        st.dataframe(df.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)

        # ── Charts side-by-side ──────────────────────────────────────────────
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Performance Comparison**")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            df.plot(kind="bar", ax=ax1, width=0.7, colormap="tab10", legend=True)
            ax1.set_ylim(0, 1)
            ax1.set_ylabel("Score")
            ax1.set_xticklabels(df.index, rotation=20, ha="right")
            ax1.legend(fontsize=8, loc="upper right")
            ax1.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig1)

        with col_b:
            st.markdown("**Precision-Recall Curves**")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            colors = {"TF-IDF": "#e74c3c", "BM25": "#3498db",
                      "Dense": "#27ae60", "Hybrid RRF": "#f39c12"}
            recall_pts = np.linspace(0, 1, 50)

            for name, qpr in pr_raw.items():
                interps = []
                for rec_pts, prec_pts in qpr:
                    if not rec_pts:
                        continue
                    row = [max((p for r, p in zip(rec_pts, prec_pts) if r >= thr), default=0.0)
                           for thr in recall_pts]
                    interps.append(row)
                if interps:
                    ax2.plot(recall_pts, np.mean(interps, axis=0),
                             label=name, color=colors[name], linewidth=2.2)

            ax2.set_xlabel("Recall")
            ax2.set_ylabel("Precision")
            ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
            ax2.legend(fontsize=8)
            ax2.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2)
