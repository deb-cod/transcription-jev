# Native Gemma 4 E2B-only setup

This profile runs the smaller Gemma 4 E2B instruction model through the same
OpenJev direct/logprob path as the existing E4B profile:

```text
Browser :8501
  -> Streamlit UI
  -> FastAPI :8000
  -> OpenJev direct decisions :8092
  -> llama.cpp :18081
  -> gemma-4-E2B-it-Q4_0.gguf
```

It exposes only `openjev_gemma`. It does not start Ollama, download MiniCPM,
or load E4B. E2B and E4B deliberately share ports and the API backend name, so
run only one native Gemma profile at a time.

## Quick setup on Windows

Open PowerShell in the project root. If Git, Python 3.12, Go, or GitHub CLI
still need to be installed, use an Administrator terminal:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup-gemma-e2b-only.ps1 -InstallPrerequisites
```

If the prerequisites are already installed, a normal PowerShell terminal is
enough:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup-gemma-e2b-only.ps1
```

The setup script:

1. checks Git, Python, Go 1.24+, GitHub CLI, Git Bash, and the NVIDIA driver;
2. creates or reuses `venv` and installs the application dependencies;
3. clones or validates the pinned OpenJev source;
4. applies the direct/logprob compatibility patch;
5. installs OpenJev's pinned `llama.cpp` b11056 CUDA 12.4 runtime;
6. downloads only `gemma-4-E2B-it-Q4_0.gguf`;
7. verifies the GGUF against its published SHA-256 checksum.

If the GitHub CLI requests authentication while obtaining `llama.cpp`, run
`gh auth login` and then repeat the setup command.

## Model artifact

The profile downloads this one model file:

| Property | Value |
|---|---|
| Repository | `ggml-org/gemma-4-E2B-it-GGUF` |
| File | `gemma-4-E2B-it-Q4_0.gguf` |
| Local directory | `models\gemma4-e2b` |
| Published size | 2.84 GB decimal |
| SHA-256 | `8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52` |
| Runtime model ID | `gemma4-e2b-native` |

The source is the official
[`ggml-org/gemma-4-E2B-it-GGUF` Q4_0 file](https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/blob/main/gemma-4-E2B-it-Q4_0.gguf).
The setup does not fetch the multimodal projector, MTP file, other
quantizations, or the E4B model.

To reuse a model copied from another computer, place it at
`models\gemma4-e2b\gemma-4-E2B-it-Q4_0.gguf` before running setup. The script
skips the download but still checks the full hash.

## Run

Start the backend in terminal 1:

```powershell
.\run-gemma-e2b-only.ps1
```

The launcher prefers the explicitly downloaded file at the default path. If it
is absent, `llama.cpp` uses
`-hf ggml-org/gemma-4-E2B-it-GGUF:Q4_0 --no-mmproj` and manages the download in
its Hugging Face cache.

To use an E2B GGUF stored elsewhere:

```powershell
.\run-gemma-e2b-only.ps1 -NativeGemmaGguf 'D:\models\gemma-4-E2B-it-Q4_0.gguf'
```

Leave terminal 1 open. Start the UI in terminal 2:

```powershell
.\run-ui.ps1
```

Open <http://localhost:8501>. The selector should contain only **OpenJev native
Gemma (experimental)**. The label stays model-size-neutral because both E2B and
E4B use the same `openjev_gemma` API contract.

## Verify

In a third PowerShell terminal:

```powershell
$Models = Invoke-RestMethod http://127.0.0.1:8000/models
$Models.default_backend
$Models.backends.Keys

Invoke-RestMethod http://127.0.0.1:8092/health | ConvertTo-Json -Depth 10
Invoke-RestMethod http://127.0.0.1:18081/v1/models | ConvertTo-Json -Depth 10
```

The API default and only backend key should be `openjev_gemma`. The service and
`llama.cpp` model ID should be `gemma4-e2b-native`.

Test a classification without the UI:

```powershell
$Body = @{
  transcript = "Caller: I am calling from Dr. Patel's office to verify insurance eligibility."
  backend = 'openjev_gemma'
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/classify `
  -ContentType 'application/json' -Body $Body | ConvertTo-Json -Depth 10
```

Runtime logs are written under `tmp\runtime`. If startup fails, inspect
`llama-gemma.stderr.log` first.

## Manual one-file download

The automated setup is recommended. To download only the model after the
virtual environment already exists:

```powershell
$ModelDirectory = Join-Path (Get-Location) 'models\gemma4-e2b'
New-Item -ItemType Directory -Force $ModelDirectory | Out-Null
.\venv\Scripts\hf.exe download `
  ggml-org/gemma-4-E2B-it-GGUF `
  gemma-4-E2B-it-Q4_0.gguf `
  --local-dir $ModelDirectory

$Model = Join-Path $ModelDirectory 'gemma-4-E2B-it-Q4_0.gguf'
(Get-FileHash -LiteralPath $Model -Algorithm SHA256).Hash.ToLowerInvariant()
```

The returned hash must match the value in the model table above.

## Switch between E2B and E4B

Stop the current backend with `Ctrl+C` before switching; both profiles use
ports 18081, 8092, and 8000.

```powershell
# Smaller E2B profile
.\run-gemma-e2b-only.ps1

# Existing E4B profile
.\run-gemma-only.ps1
```

The launchers pass distinct model IDs and OpenJev configuration files into the
shared `run.ps1`. Existing E4B commands and defaults are unchanged.
