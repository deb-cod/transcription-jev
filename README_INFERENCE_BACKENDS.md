# Inference Backend Guide

For measured latency, storage, memory pressure, and electricity-cost formulas,
see [README_INFERENCE_COSTS.md](README_INFERENCE_COSTS.md).
For the exact four-chunk benchmark transcript, method, raw samples, and current
results, see [README_INFERENCE_BENCHMARK.md](README_INFERENCE_BENCHMARK.md).

The application exposes five inference choices in the Streamlit UI. They use
different paths even when the same Ollama model name is displayed.

## Quick comparison

| UI option | Request path | Result evidence | Typical use |
|---|---|---|---|
| **Smart hybrid (fast)** | Native OpenJev first; direct Ollama only when uncertain | Native probabilities, or categorical Ollama labels after fallback | Recommended general-purpose option |
| **OpenJev native (fast)** | FastAPI → OpenJev → `llama.cpp` | Token-logprob probability distribution | Lowest and most predictable latency |
| **OpenJev native Gemma (experimental)** | FastAPI → OpenJev → `llama.cpp` | Gemma 4 E4B token-logprob distribution | Compare Gemma using OpenJev's direct method |
| **OpenJev with Ollama** | FastAPI → OpenJev → Ollama | Model-generated option weights normalized by OpenJev | Testing Ollama through the OpenJev engine interface |
| **Ollama direct** | FastAPI → Ollama | Structured categorical labels without probabilities | Fastest way to use any installed Ollama completion model |

Latency depends on the transcript, model, hardware, model warm state, and
current system load. Measurements below are examples from this laptop, not
guarantees.

## The shared pipeline, in plain and technical terms

In plain language, the application first cuts a long call into pieces that fit
inside the models. It asks the selected backend the same three questions about
every piece, then combines the evidence into one answer for the whole call.
The backend choice changes who answers those questions and what evidence is
available; it does not change the task definitions.

Technically, the request goes through these stages:

```text
Streamlit
  -> FastAPI POST /classify
  -> configuration snapshot
  -> safe transcript preprocessing
  -> speaker-aware chunks with overlap
  -> selected DecisionClient for each chunk
  -> probability aggregation OR categorical vote aggregation
  -> task thresholding
  -> structured JSON response
```

With `C` chunks and the current `T = 3` tasks, model work expands as follows:

| Backend | Calls made for `C` chunks | Work for this four-chunk benchmark |
|---|---|---|
| Smart hybrid, native accepted | `C` OpenJev calls; OpenJev schedules `C × T` direct decisions | 4 OpenJev calls and 12 one-token decisions |
| Smart hybrid, fallback | Native work plus `C` direct Ollama generations | Previous row plus 4 Ollama generations |
| OpenJev native MiniCPM | `C` OpenJev calls and `C × T` direct decisions | 4 OpenJev calls and 12 one-token decisions |
| OpenJev native Gemma | Same call count as MiniCPM, using the larger Gemma runtime | 4 OpenJev calls and 12 one-token decisions |
| Ollama direct | `C` Ollama generations, each answering all tasks in one JSON object | 4 structured generations |
| OpenJev with Ollama | `C` OpenJev calls and `C × T` verbose weight generations | 4 OpenJev calls and 12 generated distributions |

“One-token decision” does not mean the transcript costs one token. The model
must still ingest the entire chunk and its labels; only the answer is reduced
to one constrained letter. That is why native latency grows with prompt length
and chunk count. OpenJev with Ollama does more output work because every task
generates all option weights separately.

### What the confidence numbers mean

- Native MiniCPM and Native Gemma expose constrained label-token logprobs.
  OpenJev applies softmax across allowed labels; Python aggregates those
  distributions and applies the configured `0.60` task threshold.
- OpenJev with Ollama asks the model to write relative option weights. OpenJev
  validates and normalizes them, but they are not observed token probabilities
  and should not be compared numerically with native confidence.
- Ollama direct returns only valid categorical labels. The application reports
  `confidence: null` and `probabilities: null` rather than inventing evidence.
- Hybrid returns the evidence type of the route it selected: native
  probabilities on the fast path or categorical votes after fallback.

## 1. Smart hybrid (fast)

This is the recommended option when you want native OpenJev speed for clear
calls and Ollama as a fallback for ambiguous calls.

```text
Transcript
    ↓
Native OpenJev
    ↓
Are all task labels known and confidence >= 0.65?
    ├── Yes → return the native OpenJev result
    └── No  → run the selected direct Ollama model and return its result
```

Important behavior:

- Native OpenJev always runs first.
- Ollama is skipped if every task is sufficiently confident and not `unknown`.
- If any task is below the fallback threshold or is `unknown`, the complete
  transcript is classified again using the selected Ollama model.
- Results from the two engines are not mixed. The response contains either the
  accepted native result or the replacement Ollama result.
- The model selected beside this backend is the **Ollama fallback model**. It
  does not replace the native model used for the first attempt.
- `model.routing` in the API response identifies the selected path, whether a
  fallback occurred, the reason, and each backend's latency.

Example measured behavior:

- Historical clear one-chunk call: approximately **230 ms**, with Ollama skipped.
- Current four-chunk benchmark: **1.873 seconds mean**, with Ollama skipped.
- A fallback always costs the complete native pass plus a complete direct
  Ollama pass; the long test did not fall back because all three native results
  exceeded `0.65`.

Configure the routing policy in `config/classification.yaml`:

```yaml
hybrid:
  fallback_confidence: 0.65
  fallback_on_unknown: true
```

A lower confidence value reduces Ollama usage and average latency. A higher
value sends more calls to Ollama. This route does not make Ollama itself run in
300 ms; it achieves native-like latency when Ollama is unnecessary.

## 2. OpenJev native (fast)

For a fresh MiniCPM-only installation and an interface that exposes only this
backend, follow
[README_NATIVE_OPENJEV_ONLY.md](README_NATIVE_OPENJEV_ONLY.md). Use
`setup-openjev-native.ps1` once, then `run-openjev-native-only.ps1` whenever you
want to run it.

Request path:

```text
Streamlit → FastAPI → OpenJev on port 8090 → llama.cpp on port 18080
```

This backend uses the configured MiniCPM GGUF model through `llama.cpp` and
OpenJev's direct decision method. It reads constrained option-token logprobs and
returns a probability distribution for every classification task.

Use it when:

- minimum latency is the priority;
- predictable performance matters;
- probability charts and confidence thresholding are required; or
- you do not need a larger Ollama model to reconsider uncertain results.

On this laptop, a short one-chunk transcript normally completes in roughly
200–300 ms after the model is loaded. The current 14,705-character,
four-chunk benchmark averaged **1.910 seconds**.

The model shown in this UI mode is fixed by the native OpenJev configuration.
Selecting an Ollama model is not applicable to this backend.

## 3. OpenJev native Gemma (experimental)

For a clean one-model installation, backend allow-list, and exact startup
commands, use [README_NATIVE_GEMMA_ONLY.md](README_NATIVE_GEMMA_ONLY.md). Its
`run-gemma-only.ps1` launcher exposes only `openjev_gemma` and does not require
the MiniCPM model or an Ollama model.

Request path:

```text
Streamlit → FastAPI → OpenJev on port 8092 → llama.cpp on port 18081
```

This backend uses the official `ggml-org/gemma-4-E4B-it-GGUF` Q4_0 model with
OpenJev's direct token-logprob method. It is a separate service and does not
replace the fast MiniCPM backend.

The Ollama `gemma4:e4b` blob cannot be reused directly: it is an
Ollama-specific multimodal GGUF that upstream `llama.cpp` rejects with a tensor
count mismatch. The launcher therefore downloads the compatible 4.59 GB
`llama.cpp`-facing GGUF on first use and caches it locally.

This backend returns real option-token distributions, but it is experimental
and slower than MiniCPM because Gemma is much larger. Start it with
`-OpenJevEngine all` or `-OpenJevEngine gemma.cpp`.

On the validated laptop, the first download and load took about seven minutes,
subsequent cached startup took about five seconds, and the latest 412-character
test classification took 1,168.9 ms. The current four-chunk benchmark averaged
**4.914 seconds** while all services were loaded.

## 4. OpenJev with Ollama

For a complete five-backend guide plus a detailed OpenJev-with-Ollama
walkthrough, see
[README_OPENJEV_WITH_OLLAMA.md](README_OPENJEV_WITH_OLLAMA.md).

Request path:

```text
Streamlit → FastAPI → OpenJev on port 8091 → Ollama on port 11434
```

This keeps OpenJev as the decision-service layer but replaces its `llama.cpp`
engine with the configured Ollama model. OpenJev asks the model to generate
non-negative weights for all allowed options, validates the response, and
normalizes the weights to sum to one.

These values are **model-generated weights**, not token logprobs or calibrated
probabilities. They should not be compared directly with native OpenJev's
probabilities.

Use it when:

- testing the OpenJev Ollama adapter itself;
- another client must continue communicating through OpenJev's API; or
- a full distribution-shaped response is required despite the values being
  generated weights.

This is normally the slowest option because the model must generate a verbose
distribution for every configured task. The four-chunk test required 12
generated distributions and took **232.146 seconds** in its successful warm
run. Keep this backend for compatibility and adapter testing, not low-latency
classification.

The model for this OpenJev instance is chosen when the service starts. Changing
the model in the UI does not hot-swap the model inside the running OpenJev
Ollama service. Restart with another `-OllamaModel` value to change it.

## 5. Ollama direct

Request path:

```text
Streamlit → FastAPI → Ollama on port 11434
```

The Python adapter calls Ollama's structured chat API directly. It supplies a
JSON Schema containing the allowed labels, disables thinking, uses temperature
zero, and validates every returned label against `classification.yaml`.

Use it when:

- you want to use any installed completion-capable Ollama model;
- you want lower latency than the OpenJev-with-Ollama generation path; or
- categorical labels are sufficient.

Ollama's normal structured-generation response does not provide the option
token distribution needed by this application. Therefore direct Ollama results
have:

```text
confidence: null
probabilities: null
```

For multi-chunk transcripts, the application combines categorical answers and
returns `vote_counts`. On the four-chunk benchmark, three warm measurements
ranged from **10.706 to 51.885 seconds**, with a **35.491-second mean**, while
all model services were resident. That wide range reflects severe VRAM
contention and cache/offload state; isolated Ollama runs can differ greatly.

## Starting all five choices

Stop the old backend and UI with `Ctrl+C`. Then start the backend from a fresh
PowerShell terminal:

```powershell
# Run from the project directory.
Remove-Item Env:INFERENCE_BACKEND -ErrorAction SilentlyContinue
.\run.ps1 -OpenJevEngine all -OllamaModel gemma4:e4b
```

Start the UI in another terminal:

```powershell
# Run from the project directory.
.\run-ui.ps1
```

Open <http://localhost:8501>. `Smart hybrid (fast)` is selected by default in
`all` mode when `INFERENCE_BACKEND` is not already set.

For an isolated native Gemma run with less GPU competition:

```powershell
Remove-Item Env:INFERENCE_BACKEND -ErrorAction SilentlyContinue
.\run.ps1 -OpenJevEngine gemma.cpp
```

Running `all` keeps MiniCPM, native Gemma, and Ollama-accessible services
available together, but they compete for the laptop's limited GPU memory. Use
the isolated command for meaningful Gemma latency measurements.

## Backend and model selection through the API

The UI sends the selected backend and model to `POST /classify`:

```json
{
  "transcript": "I am calling from a clinic to verify insurance eligibility.",
  "backend": "hybrid",
  "model": "gemma4:e4b"
}
```

Valid backend identifiers are:

| UI name | API value |
|---|---|
| Smart hybrid (fast) | `hybrid` |
| OpenJev native (fast) | `openjev` |
| OpenJev native Gemma (experimental) | `openjev_gemma` |
| OpenJev with Ollama | `openjev_ollama` |
| Ollama direct | `ollama` |

Available choices can be inspected with:

```powershell
Invoke-RestMethod http://localhost:8000/models | ConvertTo-Json -Depth 10
Invoke-RestMethod http://localhost:8000/health | ConvertTo-Json -Depth 10
```

## Which backend should I choose?

- Start with **Smart hybrid (fast)** for normal use.
- Choose **OpenJev native (fast)** when consistent 200–300 ms latency is more
  important than Ollama fallback.
- Choose **OpenJev native Gemma (experimental)** to test Gemma with OpenJev's
  direct token-logprob method; expect more memory use and higher latency.
- Choose **Ollama direct** when you specifically want an installed Ollama model
  and can accept categorical results and higher latency.
- Choose **OpenJev with Ollama** when you specifically need to exercise Ollama
  through OpenJev; it is not the recommended low-latency path.
