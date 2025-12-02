# streamlit_app.py
import os
import tempfile
import time
from typing import List

import streamlit as st

from summarizer import (
    extract_text_from_pdf,
    chunk_text,
    summarize_chunk,
    aggregate_summaries,
    file_hash_bytes,
)

# --------- Page config ----------
st.set_page_config(
    page_title="Generative PDF Summarizer",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("📄 Generative PDF Summarizer")
st.caption("Upload a PDF → it extracts text → summarizes with Gemini → download summary")

# Sidebar controls
st.sidebar.header("Settings")
target_words = st.sidebar.slider("Final summary length (words)", 100, 800, 300, step=50)
per_chunk_words = st.sidebar.slider("Per-chunk target (words)", 80, 400, 150, step=10)
max_chars = st.sidebar.number_input("Chunk size (chars)", value=3000, min_value=1000, max_value=20000)

# Simple local cache in session state to avoid re-calling API for same file
if "summary_cache" not in st.session_state:
    st.session_state.summary_cache = {}  # maps file_hash -> summary text

# Upload widget
uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])

def summarize_uploaded_pdf(file_bytes: bytes) -> str:
    """
    Full pipeline: save uploaded bytes -> extract -> chunk -> summarize each chunk -> aggregate.
    Uses summarizer functions.
    Returns final summary string.
    """
    # compute hash for caching
    fh = file_hash_bytes(file_bytes)
    if fh in st.session_state.summary_cache:
        return st.session_state.summary_cache[fh]

    # write to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # 1) Extract text
        text = extract_text_from_pdf(tmp_path)
        if not text or not text.strip():
            raise RuntimeError("No extractable text found in PDF (probably scanned image).")

        # 2) Chunk text
        chunks: List[str] = chunk_text(text, max_chars=int(max_chars))

        # 3) Summarize each chunk with progress
        summaries = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i, chunk in enumerate(chunks, start=1):
            status_text.text(f"Summarizing chunk {i}/{len(chunks)}...")
            # call summarize_chunk (wrap in try to surface errors)
            try:
                s = summarize_chunk(chunk, target_words=int(per_chunk_words))
            except Exception as e:
                raise RuntimeError(f"Error while summarizing chunk {i}: {e}")
            summaries.append(s.strip())
            progress_bar.progress(i / len(chunks))
            # small sleep to make UI responsive if many tiny chunks (optional)
            time.sleep(0.1)

        status_text.text("Aggregating summaries...")
        final = aggregate_summaries(summaries, target_words=int(target_words))
        status_text.text("Done.")
        progress_bar.empty()

        # cache and return
        st.session_state.summary_cache[fh] = final
        return final

    finally:
        # cleanup temp file
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# Main UI
if uploaded is None:
    st.info("Upload a PDF to start summarization.")
else:
    # read bytes
    file_bytes = uploaded.getvalue()

    # simple file size check
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > 20:
        st.warning(f"Large file ({size_mb:.1f} MB). Summarization may be slow or costly. Consider smaller PDFs.")
    st.markdown(f"**File:** {uploaded.name} — **Size:** {size_mb:.2f} MB")

    try:
        with st.spinner("Running PDF → summarization pipeline (this may use API tokens)..."):
            final_summary = summarize_uploaded_pdf(file_bytes)

        # show result
        st.subheader("Final Summary")
        st.write(final_summary)

        # allow download
        st.download_button(
            label="Download summary (.txt)",
            data=final_summary,
            file_name=f"{os.path.splitext(uploaded.name)[0]}_summary.txt",
            mime="text/plain",
        )

        # show cache info
        st.write("")
        st.caption("Tip: repeated uploads of the same file are cached to avoid extra API calls.")

    except Exception as e:
        st.error(f"Summarization failed: {e}")
        st.stop()

# Footer / help
st.markdown("---")
st.write(
    "Notes: This app uses a remote generative model (Gemini). API calls consume quota—watch your billing dashboard. "
    "If your PDF is scanned (images), the extractor may return no text; use OCR (Tesseract / Vision) first."
)
