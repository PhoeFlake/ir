import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import utils

pipeline = st.session_state.pipeline
corpus_ids = pipeline["corpus_ids"]
db = st.session_state.db
def get_text(qid):
    return db.execute("SELECT text FROM questions WHERE qid=?", (int(qid),)).fetchone()[0]



st.header("Model Evaluation")
st.markdown("Academic evaluation of retrieval performance on queries with known duplicates.")

n_eval = st.slider("Number of evaluation queries", 20, 200, 50, step=10)

if st.button("Run Evaluation", type="primary"):
    with st.spinner("Evaluating models..."):
        valid_qids = [r[0] for r in db.execute("SELECT DISTINCT qid FROM qrels").fetchall()]
        sample_qids = valid_qids[:n_eval]

        res = {m: [] for m in ["TF-IDF", "BM25", "Dense", "Hybrid RRF"]}
        pr_raw = {m: [] for m in ["TF-IDF", "BM25", "Dense", "Hybrid RRF"]}
        
        # We will also calculate @5 and @10
        retrieval_perf = {m: {"P@5": [], "P@10": [], "R@5": [], "R@10": []} for m in res.keys()}
        
        bar = st.progress(0)

        for idx, qid in enumerate(sample_qids):
            q_txt = get_text(qid)
            proc_q = utils.preprocess_text(q_txt)
            rel = set([r[0] for r in db.execute("SELECT duplicate_id FROM qrels WHERE qid=?", (int(qid),)).fetchall()])

            q_tf = pipeline["tfidf_vec"].transform([proc_q])
            tf_s = pipeline["tfidf_mat"].dot(q_tf.T).toarray().flatten()
            tf_r = [corpus_ids[i] for i in tf_s.argsort()[::-1][:100]]

            bm_s = pipeline["bm25_model"].get_scores(proc_q.split())
            bm_r = [corpus_ids[i] for i in np.argsort(bm_s)[::-1][:100]]

            q_de = pipeline["dense_model"].encode([q_txt], show_progress_bar=False)
            # CHUNKED DOT PRODUCT to avoid OOM
            de_s = np.zeros(len(pipeline["dense_mat"]), dtype=np.float32)
            chunk_size = 20000
            for i in range(0, len(pipeline["dense_mat"]), chunk_size):
                chunk = pipeline["dense_mat"][i:i+chunk_size]
                de_s[i:i+chunk_size] = np.dot(q_de, chunk.astype(np.float32).T).flatten()
            
            de_r = [corpus_ids[i] for i in de_s.argsort()[::-1][:100]]

            rr_r = utils.reciprocal_rank_fusion([tf_r, bm_r, de_r])[:100]

            for name, ranked in [("TF-IDF", tf_r), ("BM25", bm_r), ("Dense", de_r), ("Hybrid RRF", rr_r)]:
                # Base metrics (P@10, Recall, MAP, nDCG)
                p_10, recall, map_score, ndcg = utils.calculate_metrics(ranked, rel, k=10)
                # F1 calculation approximation
                f1 = 2 * (p_10 * recall) / (p_10 + recall) if (p_10 + recall) > 0 else 0
                
                # MRR calculation
                mrr = 0
                for rank, doc_id in enumerate(ranked):
                    if doc_id in rel:
                        mrr = 1.0 / (rank + 1)
                        break
                        
                res[name].append([p_10, recall, f1, map_score, mrr])
                
                # P@5, R@5
                p_5 = len(set(ranked[:5]).intersection(rel)) / 5
                r_5 = len(set(ranked[:5]).intersection(rel)) / len(rel) if rel else 0
                r_10 = len(set(ranked[:10]).intersection(rel)) / len(rel) if rel else 0
                
                retrieval_perf[name]["P@5"].append(p_5)
                retrieval_perf[name]["P@10"].append(p_10)
                retrieval_perf[name]["R@5"].append(r_5)
                retrieval_perf[name]["R@10"].append(r_10)

                # PR curve data
                hits, prec_pts, rec_pts = 0, [], []
                for k, d in enumerate(ranked):
                    if d in rel:
                        hits += 1
                    prec_pts.append(hits / (k + 1))
                    rec_pts.append(hits / len(rel) if rel else 0)
                pr_raw[name].append((rec_pts, prec_pts))

            bar.progress((idx + 1) / len(sample_qids))

    # Summary table
    summary = []
    for name, metrics in res.items():
        summary.append({
            "Model": name,
            "Precision": round(np.mean([m[0] for m in metrics]), 4),
            "Recall": round(np.mean([m[1] for m in metrics]), 4),
            "F1": round(np.mean([m[2] for m in metrics]), 4),
            "MAP": round(np.mean([m[3] for m in metrics]), 4),
            "MRR": round(np.mean([m[4] for m in metrics]), 4),
        })
    df = pd.DataFrame(summary).set_index("Model")
    st.dataframe(df.style.highlight_max(axis=0, color="#f7f7f5"), use_container_width=True)
    
    st.markdown("### Retrieval performance")
    perf_summary = []
    for name, metrics in retrieval_perf.items():
        perf_summary.append({
            "Model": name,
            "Precision@5": round(np.mean(metrics["P@5"]), 4),
            "Precision@10": round(np.mean(metrics["P@10"]), 4),
            "Recall@5": round(np.mean(metrics["R@5"]), 4),
            "Recall@10": round(np.mean(metrics["R@10"]), 4),
        })
    df_perf = pd.DataFrame(perf_summary).set_index("Model")
    st.dataframe(df_perf.style.highlight_max(axis=0, color="#f7f7f5"), use_container_width=True)

    # Charts side-by-side
    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Performance Comparison (MAP & MRR)**")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        df[["MAP", "MRR"]].plot(kind="bar", ax=ax1, width=0.6, color=["#171717", "#B92B27"])
        ax1.set_ylim(0, 1)
        ax1.set_ylabel("Score")
        ax1.set_xticklabels(df.index, rotation=0)
        ax1.legend(fontsize=9, loc="upper right")
        ax1.grid(axis="y", alpha=0.3)
        # Apply restrained styling to plot
        fig1.patch.set_facecolor('#F7F7F5')
        ax1.set_facecolor('#F7F7F5')
        plt.tight_layout()
        st.pyplot(fig1)

    with col_b:
        st.markdown("**Precision-Recall Curves**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        colors = {"TF-IDF": "#6B6B6B", "BM25": "#171717", "Dense": "#D9D9D4", "Hybrid RRF": "#B92B27"}
        recall_pts = np.linspace(0, 1, 50)

        for name, qpr in pr_raw.items():
            interps = []
            for rec_pts, prec_pts in qpr:
                if not rec_pts:
                    continue
                row = [max((p for r, p in zip(rec_pts, prec_pts) if r >= thr), default=0.0) for thr in recall_pts]
                interps.append(row)
            if interps:
                ax2.plot(recall_pts, np.mean(interps, axis=0), label=name, color=colors[name], linewidth=2.5)

        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)
        fig2.patch.set_facecolor('#F7F7F5')
        ax2.set_facecolor('#F7F7F5')
        plt.tight_layout()
        st.pyplot(fig2)
