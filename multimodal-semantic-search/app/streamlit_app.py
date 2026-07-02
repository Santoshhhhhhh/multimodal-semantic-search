"""Streamlit UI: natural-language photo gallery search.

Usage:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.search.search_engine import SemanticSearchEngine

st.set_page_config(page_title="Multimodal Semantic Search", page_icon="🔍", layout="wide")

st.title("🔍 Multimodal Semantic Search")
st.caption("DINOv2 + SBERT (LoRA-tuned) → shared embedding space → FAISS HNSW retrieval")

with st.sidebar:
    st.header("Setup")
    checkpoint_path = st.text_input("Checkpoint path", value="checkpoints/best.pt")
    index_path = st.text_input("FAISS index path", value="index/gallery.hnsw")
    top_k = st.slider("Results (k)", min_value=1, max_value=50, value=10)
    load_btn = st.button("Load model + index")

if "engine" not in st.session_state:
    st.session_state.engine = None

if load_btn:
    with st.spinner("Loading model and index..."):
        try:
            st.session_state.engine = SemanticSearchEngine(
                checkpoint_path=checkpoint_path, index_path=index_path
            )
            n = len(st.session_state.engine.index.id_to_path) if st.session_state.engine.index else 0
            st.sidebar.success(f"Loaded. Gallery size: {n} images.")
        except Exception as e:
            st.sidebar.error(f"Failed to load: {e}")

query = st.text_input("Describe the photo you're looking for", placeholder="a dog running on a beach at sunset")
search_clicked = st.button("Search", type="primary")

if search_clicked:
    if st.session_state.engine is None:
        st.warning("Load a model + index from the sidebar first.")
    elif not query.strip():
        st.warning("Enter a search query.")
    else:
        with st.spinner("Searching..."):
            results = st.session_state.engine.search(query, k=top_k)

        if not results:
            st.info("No results found.")
        else:
            cols = st.columns(4)
            for i, (path, score) in enumerate(results):
                with cols[i % 4]:
                    st.image(path, use_column_width=True)
                    st.caption(f"score: {score:.3f}")
