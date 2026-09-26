# Local OpenJev Call Classifier

A local proof of concept for classifying long call transcripts into dynamic,
structured decisions with real OpenJev probabilities. It is a classifier, not
a chatbot. No cloud LLM API, transcript telemetry, permanent transcript
storage, embeddings, RAG, or vector database is used.

This is a technical POC and is **not a claim of HIPAA compliance**.

## Proven local stack

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
                one OpenJev request per chunk (all tasks)
                                  |
                    chunk probability distributions
                                  |
                task-configurable evidence aggregation
                                  |
                   structured labels + probabilities
```

The model is loaded once by llama.cpp and remains persistent across requests,
chunks, and tasks.

## Windows setup

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

$env:Path = 'C:\Program Files\GitHub CLI;C:\Program Files\Go\bin;' + $env:Path
Push-Location vendor/openjev
& 'C:\Program Files\Git\bin\bash.exe' ./scripts/setup/install_llama_cpp.sh
& 'C:\Program Files\Git\bin\bash.exe' ./scripts/setup/download_model.sh
& 'C:\Program Files\Git\bin\bash.exe' ./scripts/process/build.sh
Pop-Location
```

The compatibility patch widens OpenJev's direct-decision logprob read window
from a minimum of 64 to 256. A real 12-option request otherwise reproducibly
failed with `missing option logit for G`. It does not synthesize scores; it
allows llama.cpp to return every configured option logit. The patched OpenJev
test/build pipeline passes.

OpenJev downloads prebuilt llama.cpp CUDA binaries, so this pinned setup does
not require CMake or Visual Studio Build Tools.

Create the Python environment:

```powershell
& 'C:\Users\Debesh Pramanick\AppData\Local\Programs\Python\Python312\python.exe' -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## Run

Start OpenJev and FastAPI together. The model processes are hidden child
processes; logs go to `tmp/runtime`, and processes started by this script stop
when the foreground API stops.

```powershell
.\run.ps1
```

Or use three terminals for direct visibility:

```powershell
# Terminal 1
Set-Location vendor/openjev
& 'C:\Program Files\Git\bin\bash.exe' ./scripts/setup/run_llama_server.sh --parallel 1

# Terminal 2
Set-Location vendor/openjev
.\bin\decision-test.exe

# Terminal 3, project root
.\venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
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
- Probabilities are OpenJev direct first-token probabilities. OpenJev's chunk
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
python scripts\benchmark_requests.py
python scripts\evaluate.py
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

The UI accepts long pasted transcripts, displays task labels/confidences and
probability charts, reports chunking/no-truncation, and shows current labels.

## Optional Docker

Native Windows execution is the validated path. The optional container packages
only the Python API and calls the host's already-running OpenJev service:

```powershell
docker compose up --build
```

For a future all-in-Docker GPU deployment, install Docker Desktop with WSL2 and
the NVIDIA Container Toolkit, then containerize the pinned OpenJev/llama.cpp
runtime. That path is not claimed as validated by this POC.
