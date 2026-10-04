import streamlit as st
import pandas as pd
import random

db = st.session_state.db
def get_text(qid):
    r=db.execute("SELECT text FROM questions WHERE qid=?", (int(qid),)).fetchone()
    return r[0] if r else ""
def get_dups(qid):
    return [r[0] for r in db.execute("SELECT duplicate_id FROM qrels WHERE qid=?", (int(qid),)).fetchall()]

pipeline = st.session_state.pipeline
corpus_ids = pipeline["corpus_ids"]

st.header("Dataset Explorer")
st.markdown("Explore the 404,351 questions in the Quora Duplicate Questions dataset.")

search_corpus = st.text_input("Search the corpus (exact match)", placeholder="e.g. machine learning")

if search_corpus:
    results = []
    # Note: Sequential scan on 400k can be slow. We limit matches.
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
    st.markdown("### Random Sample (Technology & Pop Culture)")
    # Instead of truly random from 404k, query the SQLite db for interesting topics
    interesting_keywords = ['movie', 'marvel', 'programming', 'python', 'tech', 'software', 'google', 'apple', 'computer']
    query_str = " OR ".join([f"text LIKE '%{kw}%'" for kw in interesting_keywords])
    
    # Fetch 50 and randomly sample 20 to keep it fresh but on-topic
    rows = db.execute(f"SELECT qid, text FROM questions WHERE {query_str} LIMIT 200").fetchall()
    
    if rows:
        sample_rows = random.sample(rows, min(20, len(rows)))
        results = []
        for qid, text in sample_rows:
            pairs = get_dups(qid)
            pair_text = get_text(pairs[0]) if pairs else "No duplicate pair"
            results.append({
                "ID": qid, 
                "Question": text, 
                "Duplicate Pair ID": pairs[0] if pairs else "-", 
                "Duplicate Pair Text": pair_text
            })
        
        st.dataframe(pd.DataFrame(results), use_container_width=True)
