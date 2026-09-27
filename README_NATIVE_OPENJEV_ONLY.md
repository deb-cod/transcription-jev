# Native OpenJev-only setup

This profile runs the original fast native OpenJev backend with MiniCPM:

```text
Browser :8501
  -> Streamlit UI
  -> FastAPI :8000
  -> OpenJev direct decisions :8090
  -> llama.cpp :18080
  -> MiniCPM5-2B-Q4_K_M.gguf
```

It exposes only **OpenJev native (fast)** in the UI. It does not start Ollama,
native Gemma, OpenJev-with-Ollama, or the hybrid route.

## Automated setup on Windows

Copy, extract, or clone the complete project onto the computer. Open PowerShell
in the project directory. The folder can be on any drive; the scripts resolve
their paths through `$PSScriptRoot`.

If Git, Python 3.12, Go, or GitHub CLI are not installed, open PowerShell as
Administrator and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup-openjev-native.ps1 -InstallPrerequisites
```

If those programs are already installed:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup-openjev-native.ps1
```

The setup performs these operations:

1. checks the NVIDIA driver, Git, Git Bash, Python, Go, and GitHub CLI;
2. creates `venv` and installs the API and Streamlit UI packages;
3. clones the pinned OpenJev commit if it is absent;
4. applies the direct/logprob compatibility patch;
5. installs the pinned `llama.cpp` CUDA runtime;
6. downloads only `MiniCPM5-2B-Q4_K_M.gguf` from the revision-pinned URL;
7. runs the OpenJev Go tests and builds `decision-test.exe`;
8. verifies the model SHA-256 checksum.

If GitHub CLI asks for authentication while fetching `llama.cpp`:

```powershell
gh auth login
.\setup-openjev-native.ps1
```

The model is stored at:

```text
vendor/openjev/models/MiniCPM5-2B-Q4_K_M.gguf
```

Its size is 1,561,318,368 bytes, approximately 1.45 GiB. The setup does not
download the 4.59 GB Gemma GGUF and does not execute `ollama pull`.

## Run only native OpenJev

Terminal 1, opened in the project directory:

```powershell
.\run-openjev-native-only.ps1
```

Leave it running. It starts:

| Service | Address | Purpose |
|---|---|---|
| MiniCPM `llama.cpp` | `127.0.0.1:18080` | Persistent model runtime |
| OpenJev | `127.0.0.1:8090` | Direct constrained-label decisions |
| FastAPI | `127.0.0.1:8000` | Classification and backend API |

Terminal 2, also opened in the project directory:

```powershell
.\run-ui.ps1
```

Open <http://localhost:8501>. The UI should show only **OpenJev native
(fast)**.

## Backend interface isolation

The launcher sets:

```text
INFERENCE_BACKEND=openjev
INFERENCE_BACKENDS=openjev
```

`INFERENCE_BACKEND` selects the request default. `INFERENCE_BACKENDS` restricts
the clients created and advertised by `/models` and `/health`. Therefore an
installed Gemma or Ollama model will not appear in this profile.
Equivalent deployment values are recorded in
`.env.openjev-native-only.example`; PowerShell does not load that file
automatically because the dedicated launcher sets them directly.

Verify the interface:

```powershell
$Models = Invoke-RestMethod http://127.0.0.1:8000/models
$Models.default_backend
$Models.backends.Keys

Invoke-RestMethod http://127.0.0.1:8090/health | ConvertTo-Json -Depth 10
Invoke-RestMethod http://127.0.0.1:18080/v1/models | ConvertTo-Json -Depth 10
```

The default and only backend key should be `openjev`. The runtime model should
be `minicpm5-2b-q4_k_m`.

Test a classification without Streamlit:

```powershell
$Body = @{
  transcript = "Caller: This is Dr. Patel's office checking member eligibility."
  backend = 'openjev'
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/classify `
  -ContentType 'application/json' -Body $Body | ConvertTo-Json -Depth 10
```

## Switching between native backends

Stop the current backend and UI with `Ctrl+C` before switching.

Use MiniCPM native OpenJev:

```powershell
.\run-openjev-native-only.ps1
```

Use native Gemma:

```powershell
.\run-gemma-only.ps1
```

Each launcher exposes one backend and starts only its own model process. Having
both GGUF files on disk does not mean both are loaded into RAM or VRAM.

## Native OpenJev versus native Gemma

| Characteristic | Native OpenJev | Native Gemma |
|---|---|---|
| API backend | `openjev` | `openjev_gemma` |
| Model | MiniCPM5-2B Q4_K_M | Gemma 4 E4B Q4_0 |
| Model file | About 1.45 GiB | About 4.28 GiB |
| OpenJev method | Direct/logprob | Direct/logprob |
| Typical goal | Lowest latency and memory use | Better classification accuracy |
| Model port | 18080 | 18081 |
| OpenJev port | 8090 | 8092 |

Both return label-token probability distributions. Neither path uses Ollama.
Observed accuracy and latency depend on the transcript, labels, thresholds,
GPU, driver, and model residency.
