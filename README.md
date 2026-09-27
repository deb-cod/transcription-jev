# Local OpenJev Call Classifier

## Documentation map

- [README_NATIVE_GEMMA_ONLY.md](README_NATIVE_GEMMA_ONLY.md) — complete one-model installation and runtime profile that exposes only native Gemma
- [README_BEFORE_OLLAMA.md](README_BEFORE_OLLAMA.md) — original OpenJev-only installation and run procedure
- [README_AFTER_OLLAMA.md](README_AFTER_OLLAMA.md) — current installation guide for OpenJev and any supported local Ollama model
- [README_INFERENCE_BACKENDS.md](README_INFERENCE_BACKENDS.md) — how every UI inference backend works and when to use it
- [README_OPENJEV_WITH_OLLAMA.md](README_OPENJEV_WITH_OLLAMA.md) — layman and technical guide to all five backends, including native Gemma and the complete OpenJev-with-Ollama flow
- [README_INFERENCE_COSTS.md](README_INFERENCE_COSTS.md) — measured latency, storage, memory, electricity formulas, and operating cost for all backends
- [README_INFERENCE_BENCHMARK.md](README_INFERENCE_BENCHMARK.md) — reproducible long-transcript benchmark, exact test transcript, raw samples, results, and limitations
- [architecture.md](architecture.md) — complete project architecture
- [openjev-architecture.md](openjev-architecture.md) — detailed OpenJev runtime architecture

A local proof of concept for classifying long call transcripts into dynamic,
structured decisions using OpenJev or locally installed Ollama models. It is a classifier, not
a chatbot. No cloud LLM API, transcript telemetry, permanent transcript
storage, embeddings, RAG, or vector database is used.

This is a technical POC and is **not a claim of HIPAA compliance**.

## Proven legacy `llama.cpp` stack

Validated on Windows 11 with an RTX 4060 Laptop GPU (8 GB VRAM):

- `axsh/openjev` commit `65ae076b501b464f0180e43f574ab451bc918e20`
- OpenJev's pinned llama.cpp b11056 CUDA 12.4 build
- OpenBMB MiniCPM5-2B Q4_K_M GGUF (repository-pinned revision)
- Go 1.27 and Python 3.12.10
- one llama.cpp slot, 2,048-token context, `-ngl 99`
- observed warm VRAM use: about 1,981 MiB
- observed cold llama.cpp model startup: about 2.5 seconds

OpenJev is a Go HTTP service, not a Python package. The Python application uses
its documented local `POST /v1/systemone` endpoint. All OpenJev request and
response details are isolated in `app/classifier/openjev_client.py`.

## Architecture

```text
classification.yaml -> dynamic tasks, labels, definitions, thresholds
                                  |
long transcript -> safe preprocessing -> speaker-aware chunks
                                  |
             one backend request per chunk (all tasks)
                                  |
               chunk probabilities or categorical labels
                                  |
                task-configurable evidence aggregation
                                  |
             structured labels + available evidence
```

Models remain persistent across requests, chunks, and tasks. OpenJev can use
Ollama's configured `keep_alive` or its legacy long-running `llama.cpp` process.

### Current long-transcript measurement

The reproducible 14,705-character test produced four chunks. Every backend
returned `yes / doctor_office / eligibility_check`.

| Backend | Observed application latency | What the result represents |
|---|---:|---|
| Smart hybrid | 1.873 s mean | MiniCPM accepted; Ollama skipped |
| Native MiniCPM | 1.910 s mean | 12 constrained one-token decisions |
| Native Gemma | 4.914 s mean | 12 constrained one-token decisions |
| Ollama direct | 35.491 s mean | 4 structured generations; high variance under VRAM contention |
| OpenJev with Ollama | 232.146 s | 12 generated option-weight distributions |

See [README_INFERENCE_BENCHMARK.md](README_INFERENCE_BENCHMARK.md) for the exact
transcript, raw samples, confidence values, USD estimates, failure notes, and
reproduction commands. These timings are local observations, not guarantees.

## Windows setup

For a fresh installation that downloads only the native Gemma Q4_0 GGUF and
exposes only that backend in the UI, follow
[README_NATIVE_GEMMA_ONLY.md](README_NATIVE_GEMMA_ONLY.md). The general setup
below supports the multi-backend application and can require additional model
artifacts.

On a new Windows computer, the automated native-Gemma setup is:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup-gemma-only.ps1 -InstallPrerequisites
.\run-gemma-only.ps1
```

Start `.\run-ui.ps1` in a second terminal after the backend is ready.

Validate the machine in PowerShell:

```powershell
nvidia-smi
python --version
pip --version
git --version
go version
```

If Python is installed but not on `PATH`, use its full path. On the validated
machine it was:

```powershell
& 'C:\Users\Debesh Pramanick\AppData\Local\Programs\Python\Python312\python.exe' --version
```

Install missing prerequisites:

```powershell
winget install --id GoLang.Go --exact --source winget --accept-source-agreements --accept-package-agreements --silent
winget install --id GitHub.cli --exact --source winget --accept-source-agreements --accept-package-agreements --silent
```

Clone and set up OpenJev from this repository root:

```powershell
git clone https://github.com/axsh/openjev.git vendor/openjev
git -C vendor/openjev checkout 65ae076b501b464f0180e43f574ab451bc918e20
git -C vendor/openjev apply ..\..\patches\openjev-top-logprobs.patch
git -C vendor/openjev apply ..\..\patches\openjev-ollama-engine.patch

$env:Path = 'C:\Program Files\GitHub CLI;C:\Program Files\Go\bin;' + $env:Path
Push-Location vendor/openjev
& 'C:\Program Files\Git\bin\bash.exe' ./scripts/process/build.sh
Pop-Location
```

The default OpenJev engine now calls Ollama, so installing `llama.cpp` and the
MiniCPM GGUF is optional. Install those legacy components only when you intend
to run `-OpenJevEngine llama.cpp`.

The compatibility patch widens OpenJev's direct-decision logprob read window
from a minimum of 64 to 1,024. A 14,704-character, four-chunk Gemma test
otherwise reproducibly failed with `missing option logit for G`: that valid but
extremely unlikely choice ranked below the first 512 returned tokens. The patch
does not synthesize scores; it allows `llama.cpp` to return every configured
option logit. The patched OpenJev test/build pipeline passes.

When the optional legacy engine is installed, OpenJev downloads prebuilt
`llama.cpp` CUDA binaries, so this pinned setup does not require CMake or Visual
Studio Build Tools.

Create the Python environment:

```powershell
& 'C:\Users\Debesh Pramanick\AppData\Local\Programs\Python\Python312\python.exe' -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## Run

Start Ollama first, then start both OpenJev engines and FastAPI together. The
native engine runs on port `8090`; OpenJev with Ollama runs on port `8091`.

```powershell
ollama list
.\run.ps1 -OpenJevEngine both -OllamaModel gemma4:e4b
```

This `both` mode offers `Smart hybrid (fast)`, `OpenJev native (fast)`,
`OpenJev with Ollama`, and `Ollama direct`. Use `-OpenJevEngine all` to add the
fifth choice, `OpenJev native Gemma (experimental)`. Hybrid is the default in
dual/all-engine mode: it returns confident native results immediately and calls
the selected direct Ollama model only when native confidence is low or a task is
`unknown`.

This routing makes the common confident path comparable to native OpenJev
latency; it does not make an 8B Ollama generation itself run in 300 ms. A
fallback request takes native time plus Ollama generation time. Tune the routing
policy under `hybrid` in `config/classification.yaml`.

Use the legacy `llama.cpp` engine when token-logprob decisions are required:

```powershell
$env:OPENJEV_METHOD = 'direct'
.\run.ps1 -OpenJevEngine llama.cpp
```

Check the API:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/config

$Body = @{
  transcript = "Caller: This is Dr. Patel's office checking a prior authorization."
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/classify `
  -ContentType 'application/json' -Body $Body
```

Interactive docs are at `http://127.0.0.1:8000/docs`.

## Ollama models

Any installed model advertising Ollama's `completion` capability can be selected
from the UI. Embedding-only models are excluded because they cannot classify
text. The adapter uses Ollama's JSON Schema structured output and validates
every returned label against `classification.yaml`.

Run the API using Ollama without starting OpenJev:

```powershell
# Run from the project directory.
ollama list
$env:INFERENCE_BACKEND = 'ollama'
.\run.ps1 -SkipOpenJev
```

In a second terminal:

```powershell
# Run from the project directory.
.\run-ui.ps1
```

The UI discovers installed models from Ollama. Select `ollama` and then a model,
such as `gemma4:e4b`. Newly installed completion models appear automatically
after the UI's ten-second discovery cache expires.

The API also accepts backend and model explicitly:

```powershell
$Body = @{
  transcript = "Caller: This is Dr. Patel's office checking prior authorization."
  backend = 'ollama'
  model = 'gemma4:e4b'
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/classify `
  -ContentType 'application/json' -Body $Body
```

Ollama does not expose the same complete option-token probability distribution
used by OpenJev's `direct` method. The direct Python Ollama backend therefore
returns `confidence: null` and `probabilities: null`. When Ollama is used
*through OpenJev*, OpenJev uses its `generation` method: the model supplies
non-negative option weights, the adapter normalizes them to sum to one, and
OpenJev validates and aggregates that distribution. These generated values are
not token logprobs or calibrated probabilities.

## Dynamic configuration

Edit `config/classification.yaml`. Python source contains no caller-type or
intent taxonomy. Each enabled task is converted to one OpenJev choice question;
each enabled label becomes a criterion. New tasks are discovered automatically.

OpenJev currently accepts 2–20 choice criteria, so configuration validation
rejects tasks outside that range with a clear error. Quote YAML keys such as
`"yes"` and `"no"` because YAML otherwise treats them as booleans.

Reload after a local edit:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/config/reload
```

Reload is enabled for local development by default. Set
`ENABLE_CONFIG_RELOAD=false` when the API is exposed beyond localhost. The API
never edits the configuration file.

## Long transcripts and confidence

- Safe preprocessing normalizes line endings/whitespace and adjacent exact
  duplicates; it does not summarize or remove organizations, medical terms, or
  end-of-call content.
- Speaker-aware chunking is configurable. An individual utterance is split
  only when it exceeds the whole chunk limit.
- Input above `max_total_characters` is rejected and explicitly not truncated.
- The default 4,000-character chunk is conservative for the pinned 2,048-token
  context after dynamic instructions/criteria. OpenJev reports actual prompt
  tokens for every decision. Token-aware chunking through a stable OpenJev
  tokenizer API is a future improvement; the current OpenJev API does not
  expose one.
- OpenJev probabilities are direct first-token probabilities. OpenJev's chunk
  confidence is entropy-based. Final confidence is the highest normalized
  aggregate probability and is compared with the YAML threshold.
- Healthcare uses `positive_evidence_max`, appropriate to an existential
  question. Other tasks use confidence-powered probability averaging. Both
  strategies live in `evidence_aggregator.py` and are YAML-selectable.

Set `RETURN_CHUNK_DETAILS=true` to include chunk predictions. It is false by
default. Raw transcripts are never logged; `LOG_TRANSCRIPTS` remains false by
default and is reserved for an explicit future diagnostic implementation.

## Tests and evaluation

```powershell
.\venv\Scripts\Activate.ps1
python -m pytest -q
python scripts\test_model.py
python scripts\test_model.py --backend ollama --model gemma4:e4b
python scripts\benchmark_requests.py
python scripts\benchmark_all_backends.py --warmups 1 --runs 3
python scripts\evaluate.py
python scripts\evaluate.py --backend ollama --model gemma4:e4b
```

The dataset has 5 short, 10 medium, and 5 long fictional calls. Long cases put
decisive evidence at the end. Latest measured synthetic results on the target
machine:

| Split/task | Healthcare | Caller type | Intent |
|---|---:|---:|---:|
| All 20 | 70% | 80% | 95% |
| Long 5 | 100% | 100% | 80% |

These figures are a pipeline check, not a production quality claim. The
evaluator prints per-label precision/recall/F1, confusion matrices, average
latency, latency per chunk, chunk count, process RAM, and GPU VRAM snapshots.
The five long samples averaged 4.8 chunks and about 2.98 seconds.

## Streamlit

With FastAPI running:

```powershell
.\run-ui.ps1
```

Alternatively, invoke Streamlit through the venv interpreter explicitly. This
avoids accidentally using a global Python/Streamlit installation:

```powershell
.\venv\Scripts\python.exe -m streamlit run ui.py
```

The UI accepts long pasted transcripts, discovers OpenJev/Ollama models,
displays task labels and available evidence, reports chunking/no-truncation,
and shows current labels.

## Optional Docker

Native Windows execution is the validated path. The optional container packages
only the Python API and calls the host's already-running OpenJev or Ollama service:

```powershell
docker compose up --build
```

For a future all-in-Docker GPU deployment, install Docker Desktop with WSL2 and
the NVIDIA Container Toolkit, then containerize the pinned OpenJev/llama.cpp
runtime. That path is not claimed as validated by this POC.
