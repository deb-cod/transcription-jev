# Native Gemma-only setup

This guide installs and runs only this inference path:

```text
Browser :8501
  -> Streamlit UI
  -> FastAPI :8000
  -> OpenJev direct decisions :8092
  -> llama.cpp :18081
  -> gemma-4-E4B-it-Q4_0.gguf
```

Use this profile when the only UI choice should be **OpenJev native Gemma
(experimental)**. It does not start Ollama, pull an Ollama model, download
MiniCPM, or start the legacy MiniCPM service.

## Fastest setup on another Windows computer

First copy, extract, or clone this complete project onto the other computer.
Do not copy the old computer's `venv` directory because Python virtual
environments are machine-specific. Then open PowerShell **in the project
directory**. The directory can be on any drive and have any name. Every script
uses `$PSScriptRoot`, so no project path is hardcoded.

If Git, Python 3.12, Go, or GitHub CLI are not installed, run PowerShell as
Administrator and use:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup-gemma-only.ps1 -InstallPrerequisites
```

If those prerequisites are already installed, a normal PowerShell window is
enough:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup-gemma-only.ps1
```

The execution-policy change applies only to that PowerShell process. The setup
script performs the following operations in order:

1. validates the NVIDIA driver, Git, Python, Go, GitHub CLI, and Git Bash;
2. creates `venv` and installs the FastAPI, Streamlit, and HTTP dependencies;
3. clones the pinned OpenJev commit if it is absent;
4. applies only the direct/logprob compatibility patch;
5. installs the pinned CUDA `llama.cpp` runtime;
6. downloads only `gemma-4-E4B-it-Q4_0.gguf` into
   `models\gemma4-e4b`;
7. verifies the model's published SHA-256 checksum.

If GitHub CLI requests authentication while fetching `llama.cpp`, run:

```powershell
gh auth login
.\setup-gemma-only.ps1
```

After setup finishes, start the backend:

```powershell
.\run-gemma-only.ps1
```

Leave that terminal open. In a second PowerShell terminal:

```powershell
.\run-ui.ps1
```

Open <http://localhost:8501>. This starts the complete chain from the browser
UI through FastAPI, OpenJev, `llama.cpp`, and the single Gemma model.

If the 4.59 GB GGUF was copied from the old computer into
`models\gemma4-e4b\gemma-4-E4B-it-Q4_0.gguf` before setup, the installer reuses
it and verifies its checksum instead of downloading it again.

## What is downloaded

| Artifact | Purpose | Approximate size | Required? |
|---|---|---:|---|
| Python packages from `requirements.txt` | API and UI | Environment-dependent | Yes |
| OpenJev source at the pinned commit | Decision service | Small | Yes |
| Pinned `llama.cpp` b11056 Windows CUDA runtime | GGUF inference server | Environment-dependent | Yes |
| `gemma-4-E4B-it-Q4_0.gguf` | The only model weights | 4.59 GB (4.28 GiB) | Yes |
| MiniCPM GGUF | Legacy native backend | Not downloaded | No |
| Ollama and `gemma4:e4b` Ollama blob | Ollama backends | Not downloaded | No |
| Gemma multimodal projector and MTP files | Unused image/speculative features | Not downloaded | No |

The selected model file is the official
[`gemma-4-E4B-it-Q4_0.gguf`](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/blob/main/gemma-4-E4B-it-Q4_0.gguf).
Its published SHA-256 is
`a555b900214b477d8880e7832e0b8925e139b0159640036b09fe472b6f2097f2`.

Allow extra disk space for the Python environment, OpenJev source, runtime
binaries, download metadata, and temporary download data. The 4.59 GB model
size is not the total installation size.

## 1. Hardware and software prerequisites

The validated development machine used Windows 11 and an NVIDIA RTX 4060
Laptop GPU with 8 GB VRAM. Native Gemma can run differently on another GPU;
4.59 GB on disk does not mean that exactly 4.59 GB of VRAM is sufficient after
the model context, compute buffers, and application overhead are included.

Install these tools:

- Windows 10 or 11;
- a current NVIDIA driver with CUDA support visible to `nvidia-smi`;
- Git for Windows;
- Python 3.12 (Python 3.10 or newer is supported by this application);
- Go 1.24 or newer, as required by the pinned OpenJev module;
- GitHub CLI, used by OpenJev's installer to fetch the pinned `llama.cpp`
  release;
- Git Bash, installed with Git for Windows.

Example package installation from an administrator PowerShell terminal:

```powershell
winget install --id Git.Git --exact
winget install --id Python.Python.3.12 --exact
winget install --id GoLang.Go --exact
winget install --id GitHub.cli --exact
```

Restart PowerShell and verify the tools:

```powershell
nvidia-smi
git --version
py -3.12 --version
go version
gh --version
& 'C:\Program Files\Git\bin\bash.exe' --version
```

If `gh release download` later asks for authentication, run `gh auth login`
once and select GitHub.com.

## 2. Open the project and create its Python environment

Download or clone this project first, open PowerShell in its root directory,
and run:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use the development requirements only if tests will be run:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Verify that installation and execution use the same interpreter:

```powershell
.\venv\Scripts\python.exe -c "import fastapi, httpx, streamlit, uvicorn; print('Python environment ready')"
```

Always use `.\venv\Scripts\python.exe` or the provided launchers. This avoids
installing `httpx` into one Python environment and launching Streamlit from a
different one.

## 3. Install only the required OpenJev source

Skip the clone command if `vendor\openjev` already exists. For a fresh project:

```powershell
git clone https://github.com/axsh/openjev.git vendor/openjev
git -C vendor/openjev checkout 65ae076b501b464f0180e43f574ab451bc918e20
git -C vendor/openjev apply ..\..\patches\openjev-top-logprobs.patch
```

Only `openjev-top-logprobs.patch` is required for this direct/logprob backend.
The Ollama engine patch is not required for a Gemma-only deployment.

Do **not** run this legacy command:

```text
vendor/openjev/scripts/setup/download_model.sh
```

That script downloads the MiniCPM model used by the older native backend. It is
not part of this setup.

## 4. Install `llama.cpp` without downloading MiniCPM

The following script downloads the pinned Windows CUDA `llama.cpp` runtime. It
does not download a model:

```powershell
Push-Location vendor/openjev
& 'C:\Program Files\Git\bin\bash.exe' ./scripts/setup/install_llama_cpp.sh
Pop-Location
```

Confirm the server binary exists:

```powershell
Test-Path .\vendor\openjev\third_party\llama.cpp\b11056\llama-server.exe
```

The result must be `True`. There is no need to run OpenJev's complete legacy
`build.sh`: `run.ps1` builds the dedicated
`vendor\openjev\bin\decision-test-gemma.exe` from the pinned source when this
backend starts.

## 5. Download exactly one model file

Choose one of the following methods. Do not use both.

### Method A: explicit one-file download (recommended)

Install the official Hugging Face CLI into the project virtual environment:

```powershell
.\venv\Scripts\python.exe -m pip install --upgrade huggingface_hub
```

Download only the Q4_0 GGUF, not the complete model repository:

```powershell
$ModelDirectory = Join-Path (Get-Location) 'models\gemma4-e4b'
New-Item -ItemType Directory -Force $ModelDirectory | Out-Null
.\venv\Scripts\hf.exe download `
  ggml-org/gemma-4-E4B-it-GGUF `
  gemma-4-E4B-it-Q4_0.gguf `
  --local-dir $ModelDirectory
```

The positional filename is important. Omitting it would download the entire
repository, including quantizations and auxiliary files that this application
does not need. Hugging Face documents this single-file syntax and the
`--local-dir` behavior in its
[`hf download` guide](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-download).

Verify the file:

```powershell
$GemmaGguf = Join-Path (Get-Location) 'models\gemma4-e4b\gemma-4-E4B-it-Q4_0.gguf'
Get-Item -LiteralPath $GemmaGguf | Select-Object FullName, Length
(Get-FileHash -LiteralPath $GemmaGguf -Algorithm SHA256).Hash.ToLowerInvariant()
```

The hash should equal the value shown in the **What is downloaded** section.

### Method B: let `llama.cpp` manage its cache

No separate model command is required. The first run uses:

```text
-hf ggml-org/gemma-4-E4B-it-GGUF:Q4_0 --no-mmproj
```

It downloads the Q4_0 model into the Hugging Face cache. Later launches reuse
the cached file. `--no-mmproj` prevents the unused multimodal projector from
being loaded or fetched. This is convenient, but the cache path is less
obvious than Method A.

## 6. Backend interface configuration

The Gemma-only profile sets two different controls:

| Variable | Value | Meaning |
|---|---|---|
| `INFERENCE_BACKEND` | `openjev_gemma` | Default when a request does not name one |
| `INFERENCE_BACKENDS` | `openjev_gemma` | Allow-list of clients created and published by the API |
| `OPENJEV_GEMMA_URL` | `http://127.0.0.1:8092` | OpenJev Gemma service URL |

The first variable selects a default. The second prevents the API and UI from
advertising MiniCPM, hybrid, OpenJev-with-Ollama, or Ollama-direct interfaces.
The reference values are in `.env.gemma-only.example`; PowerShell does not load
that file automatically. `run-gemma-only.ps1` sets the two selection variables
for you.

API requests may omit `backend`, or specify the only accepted backend:

```json
{
  "transcript": "Caller: This is a doctor's office checking eligibility.",
  "backend": "openjev_gemma"
}
```

## 7. Run only native Gemma

Stop old backend and UI terminals with `Ctrl+C` first so their ports and
environment do not affect this run.

With the explicit model from Method A, start the backend in terminal 1:

```powershell
$GemmaGguf = Join-Path (Get-Location) 'models\gemma4-e4b\gemma-4-E4B-it-Q4_0.gguf'
.\run-gemma-only.ps1 -NativeGemmaGguf $GemmaGguf
```

With the automatic cache from Method B, use:

```powershell
.\run-gemma-only.ps1
```

Method B's first startup downloads and loads the model; Method A only loads the
already downloaded file. Either can take longer than a later warm startup.
Leave terminal 1 running. The launcher owns these processes:

| Process | Address | Role |
|---|---|---|
| `llama-server.exe` | `127.0.0.1:18081` | Runs the single Gemma GGUF |
| `decision-test-gemma.exe` | `127.0.0.1:8092` | OpenJev direct/logprob decisions |
| FastAPI/Uvicorn | `127.0.0.1:8000` | Application API and backend catalog |

Start the UI in terminal 2:

```powershell
.\run-ui.ps1
```

Open <http://localhost:8501>. The inference selector should contain only
**OpenJev native Gemma (experimental)**.

## 8. Verify that only one backend is exposed

In a third PowerShell terminal:

```powershell
$Models = Invoke-RestMethod http://127.0.0.1:8000/models
$Models.default_backend
$Models.backends.Keys

Invoke-RestMethod http://127.0.0.1:8092/health | ConvertTo-Json -Depth 10
Invoke-RestMethod http://127.0.0.1:18081/v1/models | ConvertTo-Json -Depth 10
```

The API default and the only key under `backends` should be `openjev_gemma`.
The model ID reported by the runtime should be `gemma4-e4b-native`.

Test classification without the UI:

```powershell
$Body = @{
  transcript = "Caller: I am calling from Dr. Patel's office to verify insurance eligibility."
  backend = 'openjev_gemma'
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/classify `
  -ContentType 'application/json' -Body $Body | ConvertTo-Json -Depth 10
```

## What this mode prevents—and what it does not

- It prevents this application from starting or presenting the other inference
  backends.
- It never runs `ollama pull` and never runs OpenJev's MiniCPM download script.
- It downloads one Gemma quantization when the one-file commands above are
  followed.
- It does not delete models that were already installed by Ollama or cached by
  other applications.
- It does not prevent an independently started Ollama desktop process from
  running, but this application will not create an Ollama client in this mode.

To return to the full multi-backend application later, stop this launcher,
open a fresh PowerShell terminal, and use `run.ps1` with the desired engine.
