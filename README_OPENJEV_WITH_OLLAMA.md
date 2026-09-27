# How All Inference Backends Work

For a cost comparison of every backend, see
[README_INFERENCE_COSTS.md](README_INFERENCE_COSTS.md).
For the exact long test transcript and current five-backend measurements, see
[README_INFERENCE_BENCHMARK.md](README_INFERENCE_BENCHMARK.md).

This document explains every inference backend shown in the OpenJev Call
Intelligence UI:

1. **Smart hybrid (fast)**
2. **OpenJev native (fast)**
3. **OpenJev native Gemma (experimental)**
4. **OpenJev with Ollama**
5. **Ollama direct**

The five choices are alternative routes. Selecting one backend does not run all
five backends.

## All five backends at a glance

| UI choice | Actual execution order | Inference model | Returned evidence |
|---|---|---|---|
| Smart hybrid (fast) | Native OpenJev, then direct Ollama only if uncertain | MiniCPM first; selected Ollama model only on fallback | Native probabilities or categorical Ollama result |
| OpenJev native (fast) | OpenJev → `llama.cpp` | MiniCPM GGUF | Token-logprob probabilities |
| OpenJev native Gemma (experimental) | OpenJev → `llama.cpp` | Gemma 4 E4B Instruct Q4_0 GGUF | Token-logprob probabilities |
| OpenJev with Ollama | OpenJev → Ollama | Configured Ollama model | Model-generated weights normalized by OpenJev |
| Ollama direct | Python application → Ollama | Selected Ollama model | Categorical labels without probabilities |

```text
UI backend selection
    │
    ├── Smart hybrid ─────── Native OpenJev ── uncertain? ── Direct Ollama
    │
    ├── OpenJev native ───── OpenJev :8090 ──────────────── llama.cpp :18080
    │
    ├── Native Gemma ─────── OpenJev :8092 ──────────────── llama.cpp :18081
    │
    ├── OpenJev with Ollama ─ OpenJev :8091 ─────────────── Ollama :11434
    │
    └── Ollama direct ─────── Python adapter ────────────── Ollama :11434
```

Only **Smart hybrid** can execute two inference paths for one classification
request. The other three choices use exactly one path.

## Backend 1: Smart hybrid (fast)

Smart hybrid is an application-level router designed to keep clear calls fast
while still allowing Ollama to handle ambiguous calls.

```text
Transcript
    ↓
Native OpenJev classification
    ↓
Are all labels known and confidence >= 0.65?
    ├── Yes → return native result; do not call Ollama
    └── No  → classify the complete transcript with Ollama direct
              and return the Ollama result
```

Technical behavior:

- FastAPI selects the `hybrid` route in the generic Python classifier.
- Python first calls native OpenJev on port 8090.
- Native OpenJev uses MiniCPM through `llama.cpp` on port 18080.
- Python checks the final result for every configured task.
- Any `unknown` task or confidence below `hybrid.fallback_confidence` activates
  the fallback.
- The fallback calls the direct Python Ollama adapter, not the OpenJev service
  on port 8091.
- The model selected in the UI is the Ollama fallback model.
- The two results are not averaged or merged. A confident native response is
  returned as-is; otherwise the Ollama response replaces it.
- `model.routing` explains which branch was returned and why.
- When a fallback occurs, `model.routing.fallback_tasks` lists each uncertain
  task, its native confidence, raw label, and relevant thresholds. The UI shows
  these task names in the fallback message.

Configuration:

```yaml
hybrid:
  fallback_confidence: 0.65
  fallback_on_unknown: true
```

Measured examples on this laptop:

- Historical confident one-chunk call: about 230 ms because Ollama was skipped.
- Current 14,705-character, four-chunk call: 1.873 seconds mean, also without
  fallback.
- A fallback costs the entire native pass plus an entire direct Ollama pass.

Use this backend for normal operation when you want the best average latency
with an Ollama safety net for uncertain native classifications.

## Backend 2: OpenJev native (fast)

For the isolated MiniCPM setup and launcher, see
[README_NATIVE_OPENJEV_ONLY.md](README_NATIVE_OPENJEV_ONLY.md).

This route uses OpenJev's native direct-decision setup:

```text
Streamlit
    ↓
FastAPI :8000
    ↓
OpenJev :8090
    ↓
llama.cpp :18080
    ↓
MiniCPM GGUF model
```

Technical behavior:

- Python sends all configured classification questions to OpenJev.
- OpenJev creates one work item per question.
- `llama.cpp` reads the probability of each constrained option token.
- OpenJev applies softmax, selects the strongest option, and calculates
  confidence from the resulting distribution.
- Python aggregates distributions if the transcript has multiple chunks.
- Python applies each task's minimum confidence threshold.

The returned distributions are derived from model token logprobs. Native Gemma
provides the same evidence type through a larger model.

Measured short-transcript latency on this laptop is normally around 200–300 ms
after the model is loaded; the current four-chunk benchmark averaged 1.910
seconds. Use it when consistent low latency and probability evidence are more
important than consulting an Ollama model.

The model is fixed by the native OpenJev service configuration. The Ollama model
selector is not used in this route.

## Backend 3: OpenJev native Gemma (experimental)

To install and expose only this backend without downloading MiniCPM or an
Ollama model, follow
[README_NATIVE_GEMMA_ONLY.md](README_NATIVE_GEMMA_ONLY.md).

This backend runs Gemma through the same OpenJev direct-decision mechanism as
the fast native backend:

```text
Streamlit → FastAPI :8000 → OpenJev :8092 → llama.cpp :18081
                                      ↓
                       Gemma 4 E4B Instruct Q4_0
```

It returns option-token logprob distributions. It does not use Ollama at
inference time. The model is the official 4.59 GB `llama.cpp`-facing GGUF from
`ggml-org/gemma-4-E4B-it-GGUF`, cached after its first download.

The installed Ollama `gemma4:e4b` blob is not used because that package is not
a standalone upstream-`llama.cpp` file and fails with a known tensor-count
mismatch. Both packages are Gemma 4 E4B, but they use different runtime-facing
GGUF layouts and quantizations.

Use this backend for model comparison and evaluation, not as the default fast
path. Gemma is larger than MiniCPM, so loading and inference require more RAM,
VRAM, and time.

Validated locally:

- first download plus load: approximately seven minutes;
- subsequent cached model load: approximately five seconds;
- latest 412-character, one-chunk classification: 1,168.9 ms;
- 14,705-character, four-chunk mean: 4,914.0 ms;
- result evidence: native-style `token_logprobs`;
- long-test result: healthcare `yes` 100.0%, caller `doctor_office` 99.5%,
  intent `eligibility_check` 95.6%.

Use `-OpenJevEngine gemma.cpp` for isolated benchmarks. `-OpenJevEngine all`
exposes every backend simultaneously but creates GPU-memory competition.

## Backend 4: OpenJev with Ollama

This route uses OpenJev as the decision framework and Ollama as its only model
engine:

```text
Streamlit
    ↓
FastAPI :8000
    ↓
OpenJev :8091
    ↓
Ollama :11434
    ↓
Configured Ollama model
```

There is no native MiniCPM or `llama.cpp` inference in this path. The complete
step-by-step technical flow appears later in this document.

Technical behavior:

- Python sends all configured tasks to OpenJev on port 8091.
- OpenJev normally creates a separate Ollama request for every task.
- Ollama generates a numeric weight for every allowed answer.
- OpenJev validates and normalizes those weights, chooses the highest option,
  and calculates a generation confidence.
- Python performs multi-chunk aggregation and final thresholding.

The displayed distribution consists of model-generated weights. It is not a
token-logprob distribution and is not a calibrated probability measurement.

This route is useful when another system must use OpenJev's decision API while
the underlying engine must be Ollama. It is not the low-latency Ollama path.
The four-chunk benchmark made 12 generated-distribution calls and took 232.146
seconds in its successful warm run.

## Backend 5: Ollama direct

This route bypasses OpenJev completely:

```text
Streamlit
    ↓
FastAPI :8000
    ↓
Python Ollama adapter
    ↓
Ollama :11434
    ↓
Selected Ollama model
```

Technical behavior:

- Python puts every configured task into one structured chat request per
  transcript chunk.
- A JSON Schema restricts each task to its allowed labels.
- The request uses `temperature: 0`, `think: false`, and `stream: false`.
- Python validates every returned label against `classification.yaml`.
- For multiple chunks, Python combines categorical votes.

This route requests only the winning label for each task. Ollama does not return
a comparable full option-token distribution, so the API intentionally reports:

```text
confidence: null
probabilities: null
```

It returns `vote_counts` when multiple chunks are combined. The four-chunk
benchmark ranged from 10.706 to 51.885 seconds across three warm runs, with a
35.491-second mean while all model services occupied the 8 GB GPU.

Use this backend when you specifically want a selectable Ollama model and do
not need probability charts or native OpenJev's lower latency.

## Recommended backend by requirement

| Requirement | Recommended backend |
|---|---|
| Best average latency with automatic Ollama fallback | Smart hybrid (fast) |
| Most consistent short-request latency | OpenJev native (fast) |
| Native token-logprob distributions | OpenJev native (fast) |
| Compare Gemma with native-style token logprobs | OpenJev native Gemma (experimental) |
| Use any installed Ollama completion model | Ollama direct |
| Use Ollama specifically behind the OpenJev API | OpenJev with Ollama |
| Inspect generated weights for every option | OpenJev with Ollama |

The remainder of this document explains **OpenJev with Ollama** in greater
depth because its name is the easiest to misunderstand.

## The most important answer

**OpenJev with Ollama does not run the native OpenJev model and then run
Ollama.**

For this backend, OpenJev uses Ollama as its **only model engine**:

```text
Transcript
    ↓
Python application
    ↓
OpenJev decision service
    ↓
Ollama model
    ↓
OpenJev validates and packages the model's answer
    ↓
Python application returns it to the UI
```

The native `llama.cpp` model is not called anywhere in that path.

## A simple analogy

Think of OpenJev as a supervisor holding a strict multiple-choice form, and
Ollama as the worker who reads the transcript and fills in that form.

- **OpenJev** defines the question, allowed answers, output format, validation,
  confidence calculation, and final response structure.
- **Ollama** performs the actual language-model inference and supplies the
  option weights.

OpenJev is therefore still being used, but in this mode it is not the thing
that understands the transcript. The Ollama model does that work.

## Why the name can be confusing

There are three separate OpenJev services in the project:

| UI backend | OpenJev port | Model engine used by that service |
|---|---:|---|
| OpenJev native (fast) | 8090 | `llama.cpp` with MiniCPM GGUF |
| OpenJev with Ollama | 8091 | Ollama with `gemma4:e4b` or another configured model |
| OpenJev native Gemma (experimental) | 8092 | `llama.cpp` with official Gemma 4 E4B Instruct GGUF |

They share OpenJev's decision logic, but they do not run one after the other.
They are alternative services. The UI selection determines which one receives
the request.

```text
                         ┌─ OpenJev :8090 ─ llama.cpp :18080
FastAPI backend choice ──┤
                         ├─ OpenJev :8091 ─ Ollama :11434
                         └─ OpenJev :8092 ─ llama.cpp :18081
```

Selecting **OpenJev with Ollama** chooses only the lower branch.

## Exact technical flow

Assume the UI sends this request:

```json
{
  "transcript": "I am calling from a clinic to verify insurance eligibility.",
  "backend": "openjev_ollama",
  "model": "gemma4:e4b"
}
```

### Step 1: The Streamlit UI calls FastAPI

The UI sends the transcript, backend identifier, and model name to:

```text
POST http://127.0.0.1:8000/classify
```

FastAPI sees `backend: openjev_ollama` and selects the OpenJev client configured
for port 8091. It does not call port 8090.

### Step 2: Python prepares classification questions

The generic classifier reads `config/classification.yaml` and creates one
OpenJev choice question for every enabled task. In the current configuration,
these are:

1. `healthcare_related`
2. `caller_type`
3. `intent`

Each question contains:

- the transcript;
- task instructions;
- every allowed label; and
- the description of each label.

For example, the intent question contains allowed labels such as
`appointment`, `billing`, `insurance_verification`, and
`prior_authorization`.

### Step 3: Python sends one OpenJev request

Python sends all three questions to the OpenJev service on port 8091:

```text
POST http://127.0.0.1:8091/v1/systemone
```

The request tells OpenJev to use its `generation` method. Ollama cannot provide
the constrained option-token logprobs required by OpenJev's native `direct`
method, so `generation` is required for this engine.

Conceptually, the request looks like this:

```json
{
  "state": "the transcript",
  "questions": {
    "healthcare_related": { "type": "choice", "criteria": {} },
    "caller_type": { "type": "choice", "criteria": {} },
    "intent": { "type": "choice", "criteria": {} }
  },
  "options": {
    "method": "generation"
  }
}
```

The real `criteria` objects contain every configured label and description.

### Step 4: OpenJev turns each task into a constrained Ollama request

OpenJev creates one internal work item per question. With the current three
tasks, it normally makes three Ollama `/api/chat` calls. Its worker pool can run
those calls concurrently.

For each task, OpenJev:

1. Converts the allowed choices to lettered options such as `A`, `B`, and `C`.
2. Adds the transcript and task instructions to the prompt.
3. Builds a JSON Schema requiring one numeric value from 0 to 1 for every exact
   option.
4. Calls the configured model at `http://127.0.0.1:11434/api/chat`.
5. Uses `temperature: 0`, `think: false`, and `stream: false`.

Conceptually, Ollama is asked to return something like:

```json
{
  "A: appointment": 0.05,
  "B: billing": 0.10,
  "C: insurance_verification": 0.75,
  "D: prior_authorization": 0.10
}
```

This is a simplified example. The actual intent task contains all enabled
labels.

### Step 5: OpenJev validates and normalizes the generated weights

Ollama may generate valid non-negative values that do not add up to exactly
one. The OpenJev Ollama adapter normalizes them:

```text
normalized value = generated value / sum of all generated values
```

OpenJev then checks that:

- every required option is present;
- no unexpected option is present;
- every value is numeric and between 0 and 1; and
- the normalized values add up to one.

It selects the label with the highest normalized value and calculates its
generation confidence from the shape of the distribution.

These numbers are **generated option weights**. They are not the native
model's token logprobs and should not be treated as calibrated probabilities.

### Step 6: OpenJev returns all answers to Python

OpenJev maps the lettered options back to the original label names and returns
a response shaped roughly like this:

```json
{
  "model": "gemma4:e4b",
  "answers": {
    "healthcare_related": {
      "choice": "yes",
      "confidence": 0.93,
      "probabilities": {
        "yes": 0.93,
        "no": 0.07
      },
      "method": "generation"
    },
    "caller_type": {
      "choice": "clinic",
      "confidence": 0.72,
      "probabilities": {}
    },
    "intent": {
      "choice": "insurance_verification",
      "confidence": 0.81,
      "probabilities": {}
    }
  }
}
```

The empty probability objects above are abbreviated for readability; the real
response contains the complete allowed-label distribution for every task.

### Step 7: Python aggregates and applies application thresholds

For a short transcript there is normally one chunk, so the OpenJev answer is
almost the final result. For a long transcript, Python repeats the process for
every chunk and aggregates the returned distributions.

Python then applies the task thresholds from `classification.yaml`. If the
winning aggregate probability is below a task's threshold, the final label can
become `unknown`.

### Step 8: FastAPI returns the result to Streamlit

FastAPI returns the normalized application response. Streamlit displays:

- the selected label;
- the OpenJev generation confidence;
- the normalized generated-weight chart; and
- the total request latency.

The return path is therefore:

```text
Ollama generated weights
    ↓
OpenJev validation, normalization, label selection, and confidence
    ↓
Python chunk aggregation and task thresholding
    ↓
FastAPI JSON response
    ↓
Streamlit display
```

## What is actually running the model?

| Responsibility | Component |
|---|---|
| Understand transcript text | Ollama model |
| Generate option weights | Ollama model |
| Enforce the structured schema | Ollama API and OpenJev adapter |
| Normalize generated weights | OpenJev Ollama adapter |
| Validate all allowed options | OpenJev |
| Choose the highest-weight label | OpenJev |
| Calculate generation confidence | OpenJev |
| Combine results from multiple transcript chunks | Python classifier |
| Apply final task threshold | Python classifier |
| Display result | Streamlit |

## How this differs from Smart hybrid

This distinction is essential:

### OpenJev with Ollama

```text
OpenJev framework → Ollama model → return result
```

- No native OpenJev model is called.
- Ollama is always called.
- OpenJev requests a full generated option-weight object for each task.
- The four-chunk benchmark took 232.146 seconds.

### Smart hybrid

```text
Native OpenJev model
    ↓
confident? ─ yes → return native result
    │
    no
    ↓
Direct Ollama adapter → return Ollama result
```

- The native model is always tried first.
- Direct Ollama is called only when native output is uncertain.
- The Ollama fallback returns categorical labels, not generated distributions.
- A historical clear one-chunk call took about 230 ms because Ollama was skipped.
- The current clear four-chunk call averaged 1.873 seconds without fallback.
- A fallback adds a complete direct Ollama pass to the native pass.

Only **Smart hybrid** has the “native first, Ollama second if needed” behavior.

## How this differs from Ollama direct

Both choices use the Ollama model, but the route and requested output differ:

| Detail | OpenJev with Ollama | Ollama direct |
|---|---|---|
| Route | Python → OpenJev → Ollama | Python → Ollama |
| Ollama calls | Normally one per task | One request containing all tasks per chunk |
| Output requested | Numeric weight for every option | One label per task |
| Returned probability field | Full generated-weight distribution | `null` |
| Confidence field | Calculated from generated weights | `null` |
| Current four-chunk measurement | 232.146 seconds (one successful warm run) | 35.491-second mean; 10.706–51.885-second range |

Direct Ollama is faster because it asks for a small JSON object containing only
the chosen labels. OpenJev with Ollama asks the model to generate every option
and weight for each task separately.

## Starting only OpenJev with Ollama

Backend terminal:

```powershell
# Run from the project directory.
$env:INFERENCE_BACKEND = 'openjev_ollama'
.\run.ps1 -OpenJevEngine ollama -OllamaModel gemma4:e4b
```

UI terminal:

```powershell
# Run from the project directory.
.\run-ui.ps1
```

Then select **OpenJev with Ollama** in the UI.

## Final summary

`OpenJev with Ollama` means:

> Use OpenJev to construct, control, validate, normalize, and package a
> classification decision, while Ollama is the only model that performs the
> language inference.

It does **not** mean:

> Run the native OpenJev model first and then run Ollama.

That second behavior belongs only to **Smart hybrid**.
