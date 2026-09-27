# Project Architecture

This document describes the current call-classification application and all five inference choices. OpenJev's internal implementation is covered separately in [openjev-architecture.md](openjev-architecture.md), and measured four-chunk execution is documented in [README_INFERENCE_BENCHMARK.md](README_INFERENCE_BENCHMARK.md).

## System context

```mermaid
flowchart TB
    User[User] --> Browser[Streamlit web UI :8501]
    Client[API client] --> API[FastAPI :8000]
    Browser --> API

    subgraph Application
        API --> Loader[Configuration loader]
        API --> Classifier[Generic decision classifier]
        Classifier --> Prep[Transcript preprocessor]
        Prep --> Chunker[Token-aware chunker]
        Chunker --> Router{Requested backend}
        Router --> OJC[OpenJev client]
        Router --> OLC[Direct Ollama client]
        Router --> Hybrid[Native-first hybrid router]
        Hybrid --> OJC
        Hybrid -->|uncertain only| OLC
        OJC --> AggP[Probability aggregation]
        OLC --> AggV[Categorical vote aggregation]
        AggP --> Response[Classification response]
        AggV --> Response
    end

    OJC -->|native HTTP :8090| OpenJev[OpenJev native]
    OJC -->|Ollama HTTP :8091| OpenJevOllama[OpenJev with Ollama]
    OJC -->|Gemma direct HTTP :8092| OpenJevGemma[OpenJev native Gemma]
    OpenJevOllama -->|native API :11434| Ollama
    OpenJev -->|OpenAI-compatible :18080| Llama[llama.cpp server]
    OpenJevGemma -->|OpenAI-compatible :18081| LlamaGemma[llama.cpp Gemma server]
    Llama --> GGUF[Configured GGUF model]

    OLC -->|HTTP :11434| Ollama[Ollama]
    Ollama --> LocalModel[Selected local completion model]
```

All transcript inference stays on the local machine when the configured URLs point to localhost.

## Component responsibilities

| Component | Main files | Responsibility |
|---|---|---|
| Web UI | `ui.py` | Collect transcript, select backend/model, call the API, and visualize results |
| API layer | `app/main.py`, `app/api/routes.py` | Validate requests and expose health, configuration, model discovery, and classification endpoints |
| Configuration | `app/config/loader.py`, `config/classification.yaml` | Load and validate tasks, labels, thresholds, backend settings, and reloadable configuration |
| Backend router | `app/classifier/decision_client.py`, `app/classifier/generic_classifier.py` | Select a backend, perform hybrid confidence routing when requested, and normalize responses |
| OpenJev adapter | `app/classifier/openjev_client.py` | Convert configured tasks into OpenJev questions and read validated choice distributions |
| Ollama adapter | `app/classifier/ollama_client.py` | Discover completion models, request structured label JSON, and validate returned labels |
| Transcript pipeline | `app/transcript/preprocessor.py`, `app/transcript/chunker.py` | Normalize transcript text and divide long input into overlapping chunks |
| Aggregation | `app/transcript/evidence_aggregator.py` and classifier logic | Combine per-chunk probabilities or categorical votes |
| Operations | `run.ps1`, `run-ui.ps1`, `Dockerfile`, `docker-compose.yml` | Start local services and package the Python application |
| Validation tools | `tests/`, `scripts/evaluate.py`, `scripts/benchmark_requests.py`, `scripts/benchmark_all_backends.py`, `scripts/test_model.py` | Unit tests, labeled evaluation, request benchmarking, and model checks |

## Classification request flow

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit/UI client
    participant API as FastAPI
    participant C as Generic classifier
    participant B as Selected backend
    participant A as Aggregator

    User->>UI: Submit transcript, backend, optional model
    UI->>API: POST /classify
    API->>C: Validated request
    C->>C: Normalize and chunk transcript
    loop Every transcript chunk
        C->>B: Classify all configured tasks
        B-->>C: Labels plus probabilities, or labels only
    end
    C->>A: Per-chunk results
    A-->>C: Final task results
    C-->>API: Normalized response
    API-->>UI: JSON
    UI-->>User: Labels, confidence, charts, or vote counts
```

The three tasks are configured data, not hard-coded UI labels. A configuration reload can update task labels and descriptions without changing the frontend.

## Backend abstraction

The request selects `hybrid`, `openjev`, `openjev_gemma`, `openjev_ollama`, or `ollama`. The generic
classifier owns the shared transcript pipeline and presents a stable response
shape, while each adapter implements backend-specific inference.

At deployment time, `INFERENCE_BACKENDS` can restrict which adapter clients are
created and published by `/health` and `/models`. For example,
`INFERENCE_BACKENDS=openjev_gemma` creates only the native Gemma client. The
separate `INFERENCE_BACKEND=openjev_gemma` value makes it the request default.
`run-gemma-only.ps1` sets both values and starts only ports 18081, 8092, and
8000, so the UI cannot select a backend whose model was intentionally omitted.

### Execution expansion by backend

Let `C` be the number of transcript chunks and `T` the number of enabled tasks.
The current configuration has `T = 3`.

| Backend | Service-level requests | Model-level decisions | Evidence returned |
|---|---:|---:|---|
| `openjev` | `C` calls to `:8090` | `C × T` direct decisions on MiniCPM `:18080` | Token-logprob distributions |
| `openjev_gemma` | `C` calls to `:8092` | `C × T` direct decisions on Gemma `:18081` | Token-logprob distributions |
| `openjev_ollama` | `C` calls to `:8091` | `C × T` Ollama generations on `:11434` | Generated normalized weights |
| `ollama` | `C` calls to Ollama | `C` structured generations containing all tasks | Categorical labels |
| `hybrid`, accepted | Same as `openjev` | Same as `openjev` | Native distributions |
| `hybrid`, fallback | Native pass plus same work as `ollama` | `C × T` native decisions plus `C` Ollama generations | Replacement categorical labels |

OpenJev receives all tasks in one HTTP request per chunk, but internally places
each task on its worker queue. Worker count controls scheduling capacity; the
underlying model runtime, GPU memory, and configured model slots determine how
much work truly executes concurrently. Increasing workers does not guarantee
lower latency on a single-GPU laptop.

### Hybrid path

1. Classify the complete transcript with native OpenJev.
2. Inspect every configured task after chunk aggregation and thresholding.
3. Return immediately when every label is known and confidence meets the
   configured hybrid fallback threshold.
4. Otherwise classify the complete transcript with the selected direct Ollama
   model and return that categorical result.
5. Record the chosen path, reason, and per-backend timing in `model.routing`.

This is latency routing, not faster Ollama inference. Confident calls incur only
the native path; uncertain calls incur native plus Ollama latency. Routing the
whole transcript avoids mixing probabilistic native evidence with categorical
Ollama evidence inside one aggregation pass.

### OpenJev path

1. Build one OpenJev `choice` question per configured task.
2. Submit all questions for a chunk in one `/v1/systemone` request.
3. OpenJev routes each question to its configured `ollama` or `llama.cpp` engine.
4. Read the selected label and normalized probability vector.
5. Aggregate probability evidence over chunks and apply thresholds.

With the default Ollama engine, option weights are generated by the model,
normalized by the adapter, and strictly validated by OpenJev. With the legacy
`llama.cpp` direct engine, the distribution comes from constrained label-token
logprobs. Both paths return numeric confidence, but their probability sources
are not equivalent.

### Ollama path

1. Discover locally installed models from Ollama.
2. Exclude models that cannot generate text.
3. Build a task prompt containing only the labels and their descriptions.
4. Request structured JSON constrained to a configured label.
5. Validate the returned label exactly.
6. Aggregate chunk labels using task-aware categorical voting.

This path does not invent probability values. Confidence and probabilities remain `null`, while `vote_counts` preserves aggregation evidence.

## Long-transcript processing

```mermaid
flowchart LR
    T[Raw transcript] --> N[Normalize whitespace/text]
    N --> S{Within chunk limit?}
    S -->|Yes| C1[One chunk]
    S -->|No| SP[Split with overlap]
    C1 --> I[Backend inference]
    SP --> I
    I --> R{Result type}
    R -->|Probabilities| PA[Evidence aggregation]
    R -->|Labels only| VA[Categorical vote aggregation]
    PA --> F[Final result]
    VA --> F
```

Chunk size and overlap are configured in `config/classification.yaml`. Overlap reduces the risk of losing context at a boundary but can repeat evidence, so aggregation must be consistent across all chunks.

### Probability aggregation

The OpenJev path supports the configured aggregation policies:

- healthcare detection favors the strongest positive evidence across chunks;
- multiclass caller type and intent combine chunk distributions using confidence-weighted averaging;
- minimum-confidence rules can mark a decision uncertain.

### Categorical aggregation

The Ollama path has no distribution to average:

- healthcare uses positive-evidence behavior so a relevant chunk can make the whole call healthcare-related;
- non-`unknown` votes take precedence when substantive evidence exists;
- caller introductions can help break caller-type ties;
- `vote_counts` is returned for transparency.

## HTTP interface

| Method and path | Purpose |
|---|---|
| `POST /classify` | Classify a transcript with an optional backend and model override |
| `GET /health` | Report API/backend availability |
| `GET /models` | List supported backends, hybrid fallback choices, and dynamically discovered Ollama models |
| `GET /config` | Return the active public classification configuration |
| `POST /config/reload` | Reload configuration from disk |

Representative request:

```json
{
  "transcript": "I am calling from a clinic to verify insurance eligibility.",
  "backend": "ollama",
  "model": "gemma4:e4b"
}
```

The response identifies the backend and model and returns one result for every configured task.

## Processes and ports

| Process | Default port | Needed in Ollama-only mode | Needed in OpenJev mode |
|---|---:|---:|---:|
| Streamlit | 8501 | Yes, for browser UI | Yes, for browser UI |
| FastAPI/Uvicorn | 8000 | Yes | Yes |
| Ollama | 11434 | Yes | Yes for the default OpenJev engine |
| OpenJev native | 8090 | No | Yes |
| OpenJev with Ollama | 8091 | No | Yes when selected or in dual mode |
| OpenJev native Gemma | 8092 | No | Only in `all` or `gemma.cpp` mode |
| `llama.cpp` server | 18080 | No | Only for the legacy direct engine |
| Gemma `llama.cpp` server | 18081 | No | Only for experimental native Gemma |

The API can be used without Streamlit. The backend model processes remain persistent so model weights are not reloaded for every classification.

## Configuration boundaries

`config/classification.yaml` is the main policy boundary. It contains task definitions, label vocabulary, prompt guidance, transcript chunking, backend endpoints, thresholds, and inference options.

Environment variables are best suited to deployment-specific values such as backend selection or service URLs. Per-request fields are best suited to interactive backend/model selection.

`INFERENCE_BACKEND` chooses the default route; `INFERENCE_BACKENDS` is a
comma-separated allow-list of instantiated routes. Keeping those two values
consistent prevents a default from pointing to a disabled client. The native
Gemma-only profile and full installation procedure are documented in
[README_NATIVE_GEMMA_ONLY.md](README_NATIVE_GEMMA_ONLY.md).

Configuration precedence is conceptually:

```text
request override → environment/runtime override → classification.yaml default
```

## Failure behavior

- Invalid input is rejected at the API boundary.
- An unknown backend or Ollama model produces a clear client error.
- Backend connection or timeout failures are translated to API errors rather than fabricated classifications.
- A backend label outside the configured vocabulary is rejected.
- Ollama probability fields remain `null` instead of presenting generated numbers as measured confidence.
- OpenJev health depends on both its decision service and the underlying model server being ready.

## Source layout

```text
app/
  api/                 HTTP routes and schemas
  classifier/          backend abstraction and adapters
  config/              configuration loading and validation
  transcript/          preprocessing, chunking, aggregation
config/
  classification.yaml  classification policy and backend settings
vendor/openjev/         pinned OpenJev source checkout
patches/                local OpenJev compatibility patch
samples/                sample/evaluation transcripts
scripts/                setup, evaluation, benchmark utilities
tests/                  automated tests
ui.py                   Streamlit application
run.ps1                 backend and API launcher
run-ui.ps1              UI launcher
```

## Adding another backend

A new backend should implement the same conceptual operations as the existing adapters:

1. availability/health checking;
2. optional model discovery;
3. classification of every configured task for one transcript chunk;
4. strict validation against configured labels;
5. an explicit declaration of whether real probability distributions are available.

Register it in the backend router and `/models` response, then choose probability or categorical aggregation based on its actual output. Do not synthesize confidence merely to match the OpenJev shape.

## Privacy and deployment notes

Local URLs keep inference on the laptop, but transcripts still appear in application memory and may appear in logs if verbose logging is enabled. Healthcare or other sensitive data should use restricted host binding, access controls, encrypted storage where applicable, minimal logging, and an organizational retention policy before production use.

The current application is a local proof of concept. Authentication, authorization, TLS termination, audit logging, rate limiting, and production observability are deployment responsibilities rather than properties of the local launcher.
