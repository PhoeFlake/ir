import streamlit as st
import pandas as pd
import random

pipeline = st.session_state.pipeline
corpus_ids = pipeline["corpus_ids"]
db = st.session_state.db
def get_text(qid):
    r=db.execute("SELECT text FROM questions WHERE qid=?", (int(qid),)).fetchone()
    return r[0] if r else ""
def get_dups(qid):
    return [r[0] for r in db.execute("SELECT duplicate_id FROM qrels WHERE qid=?", (int(qid),)).fetchall()]

st.header("Dataset Explorer")
st.markdown("Explore the 404,351 questions in the Quora Duplicate Questions dataset.")

search_corpus = st.text_input("Search the corpus (exact match)", placeholder="e.g. machine learning")

if search_corpus:
    results = []
    for qid in corpus_ids:
        text = get_text(qid)
        if search_corpus.lower() in text.lower():
            pairs = get_dups(qid)
            pair_text = get_text(pairs[0]) if pairs else "No duplicate pair"
            results.append({"ID": qid, "Question": text, "Duplicate Pair ID": pairs[0] if pairs else "-", "Duplicate Pair Text": pair_text})
            if len(results) >= 50:
                break
    
    if results:
        st.markdown(f"Showing top {len(results)} matches:")
        st.dataframe(pd.DataFrame(results), use_container_width=True)
    else:
        st.info("No matches found.")
else:
    st.markdown("### Random Sample")
    sample_ids = random.sample(corpus_ids, 20)
    results = []
    for qid in sample_ids:
        pairs = get_dups(qid)
        pair_text = get_text(pairs[0]) if pairs else "No duplicate pair"
        results.append({"ID": qid, "Question": get_text(qid), "Duplicate Pair ID": pairs[0] if pairs else "-", "Duplicate Pair Text": pair_text})
    
    st.dataframe(pd.DataFrame(results), use_container_width=True)
