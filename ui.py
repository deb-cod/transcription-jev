from __future__ import annotations

import os

import httpx
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="OpenJev Call Intelligence", layout="wide")
st.title("OpenJev Call Intelligence")
st.caption("Structured local classification - transcripts are not stored by this UI.")


@st.cache_data(ttl=10)
def load_models() -> dict:
    response = httpx.get(f"{API_URL}/models", timeout=10)
    response.raise_for_status()
    return response.json()


backend = "openjev"
model = None
try:
    model_catalog = load_models()
    available_backends = [
        name
        for name, details in model_catalog.get("backends", {}).items()
        if details.get("models")
    ]
    if available_backends:
        configured_default = model_catalog.get("default_backend", available_backends[0])
        default_index = (
            available_backends.index(configured_default)
            if configured_default in available_backends
            else 0
        )
        selector_col1, selector_col2 = st.columns(2)
        backend_labels = {
            "openjev": "OpenJev native (fast)",
            "openjev_gemma": "OpenJev native Gemma (experimental)",
            "openjev_ollama": "OpenJev with Ollama",
            "ollama": "Ollama direct",
            "hybrid": "Smart hybrid (fast)",
        }
        backend = selector_col1.selectbox(
            "Inference backend",
            available_backends,
            index=default_index,
            format_func=lambda value: backend_labels.get(value, value),
        )
        backend_models = model_catalog["backends"][backend]["models"]
        model_names = [item["name"] for item in backend_models]
        model = selector_col2.selectbox("Local model", model_names)
        selected_details = next(item for item in backend_models if item["name"] == model)
        metadata = " | ".join(
            str(value)
            for value in (
                selected_details.get("family"),
                selected_details.get("parameter_size"),
                selected_details.get("quantization"),
            )
            if value
        )
        if metadata:
            st.caption(metadata)
    else:
        st.warning("No available local classification model was found.")
except httpx.HTTPError:
    st.warning(f"Start the FastAPI service at {API_URL} to discover local models.")

transcript = st.text_area("Paste transcript", height=360, placeholder="Agent: ...\nCaller: ...")
col1, col2 = st.columns(2)
col1.metric("Characters", f"{len(transcript):,}")
col2.caption("Chunk count is returned after preprocessing and speaker-aware chunking.")

if st.button("CLASSIFY", type="primary", disabled=not transcript.strip(), use_container_width=True):
    try:
        with st.spinner(f"Classifying locally with {backend} / {model or 'default model'}..."):
            response = httpx.post(
                f"{API_URL}/classify",
                json={"transcript": transcript, "backend": backend, "model": model},
                timeout=1800,
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
                confidence = value.get("confidence")
                st.metric(
                    value["label"].replace("_", " ").title(),
                    f"{confidence:.1%}" if confidence is not None else "Confidence unavailable",
                )
                if value.get("raw_label") != value.get("label"):
                    st.caption(
                        "Raw prediction: "
                        + value["raw_label"].replace("_", " ").title()
                        + f"; below threshold {value['threshold']:.0%}"
                    )
                probabilities = value.get("probabilities")
                if probabilities:
                    st.bar_chart(probabilities, horizontal=True)
                elif value.get("vote_counts"):
                    st.caption("Chunk votes (counts, not probabilities)")
                    st.bar_chart(value["vote_counts"], horizontal=True)
                else:
                    st.caption("This backend does not expose label probabilities.")
        processing = result["transcript_processing"]
        st.subheader("Transcript")
        st.write(
            f"{processing['characters']:,} characters | {processing['chunks']} chunks | "
            f"{'truncated' if processing['truncated'] else 'no truncation'}"
        )
        st.caption(
            f"Model: {result['model']['name']} | backend: {result['model']['backend']} | "
            f"latency: {result['model']['latency_ms']:.1f} ms"
        )
        routing = result["model"].get("routing")
        if routing:
            if routing.get("fallback_used"):
                uncertain_tasks = []
                for detail in routing.get("fallback_tasks", []):
                    task_name = str(detail.get("task", "task")).replace("_", " ").title()
                    reason = detail.get("reason")
                    confidence = detail.get("confidence")
                    if reason == "unknown":
                        raw_label = detail.get("raw_label")
                        raw_text = (
                            f", raw choice {str(raw_label).replace('_', ' ').title()}"
                            if raw_label and raw_label != "unknown"
                            else ""
                        )
                        confidence_text = (
                            f", confidence {confidence:.1%}" if confidence is not None else ""
                        )
                        uncertain_tasks.append(
                            f"{task_name}: returned Unknown{raw_text}{confidence_text}"
                        )
                    elif reason == "low_confidence":
                        required = detail.get("hybrid_threshold")
                        uncertain_tasks.append(
                            f"{task_name}: {confidence:.1%} confidence"
                            + (f" < {required:.1%} required" if required is not None else "")
                        )
                    else:
                        uncertain_tasks.append(f"{task_name}: confidence unavailable")
                uncertainty_text = (
                    " Uncertain task(s): " + "; ".join(uncertain_tasks) + "."
                    if uncertain_tasks
                    else ""
                )
                st.info(
                    "Native OpenJev was uncertain, so the request fell back to Ollama."
                    + uncertainty_text
                    + " "
                    f"Native: {routing.get('native_latency_ms') or 0:.1f} ms | "
                    f"Ollama: {routing.get('ollama_latency_ms') or 0:.1f} ms"
                )
            else:
                st.success(
                    "Native OpenJev was confident, so Ollama was skipped "
                    f"({routing.get('native_latency_ms') or 0:.1f} ms)."
                )
    except httpx.HTTPStatusError as exc:
        st.error(f"Classification failed ({exc.response.status_code}): {exc.response.text}")
    except httpx.HTTPError as exc:
        st.error(f"Cannot reach the local API at {API_URL}: {exc}")

with st.expander("Current Labels", expanded=False):
    try:
        active = httpx.get(f"{API_URL}/config", timeout=5).json()
        for task, labels in active.items():
            st.write(f"**{task.replace('_', ' ').title()} - {len(labels)} enabled labels**")
            st.caption(", ".join(label.replace("_", " ") for label in labels))
    except httpx.HTTPError:
        st.info("Start the FastAPI service to view active labels.")
