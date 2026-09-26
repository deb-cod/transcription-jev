from __future__ import annotations

import os

import httpx
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="OpenJev Call Intelligence", layout="wide")
st.title("OpenJev Call Intelligence")
st.caption("Structured local classification — transcripts are not stored by this UI.")

transcript = st.text_area("Paste transcript", height=360, placeholder="Agent: ...\nCaller: ...")
col1, col2 = st.columns(2)
col1.metric("Characters", f"{len(transcript):,}")
col2.caption("Chunk count is returned after preprocessing and speaker-aware chunking.")

if st.button("CLASSIFY", type="primary", disabled=not transcript.strip(), use_container_width=True):
    try:
        with st.spinner("Classifying locally with OpenJev..."):
            response = httpx.post(
                f"{API_URL}/classify", json={"transcript": transcript}, timeout=600
            )
            response.raise_for_status()
            result = response.json()
        task_names = [
            key
            for key, value in result.items()
            if isinstance(value, dict) and "label" in value and "probabilities" in value
        ]
        columns = st.columns(min(3, len(task_names)))
        for index, task in enumerate(task_names):
            value = result[task]
            with columns[index % len(columns)]:
                st.subheader(task.replace("_", " ").title())
                st.metric(value["label"].replace("_", " ").title(), f"{value['confidence']:.1%}")
                st.bar_chart(value["probabilities"], horizontal=True)
        processing = result["transcript_processing"]
        st.subheader("Transcript")
        st.write(
            f"{processing['characters']:,} characters · {processing['chunks']} chunks · "
            f"{'truncated' if processing['truncated'] else 'no truncation'}"
        )
        st.caption(
            f"Model: {result['model']['name']} · backend: {result['model']['backend']} · "
            f"latency: {result['model']['latency_ms']:.1f} ms"
        )
    except httpx.HTTPStatusError as exc:
        st.error(f"Classification failed ({exc.response.status_code}): {exc.response.text}")
    except httpx.HTTPError as exc:
        st.error(f"Cannot reach the local API at {API_URL}: {exc}")

with st.expander("Current Labels", expanded=False):
    try:
        active = httpx.get(f"{API_URL}/config", timeout=5).json()
        for task, labels in active.items():
            st.write(f"**{task.replace('_', ' ').title()} — {len(labels)} enabled labels**")
            st.caption(", ".join(label.replace("_", " ") for label in labels))
    except httpx.HTTPError:
        st.info("Start the FastAPI service to view active labels.")

