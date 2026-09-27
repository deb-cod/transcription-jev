# Local Inference Cost Guide

This document compares the practical cost of every inference backend in the
OpenJev Call Intelligence application. Because all inference runs locally,
there is no per-token cloud API bill. The relevant costs are:

- response time;
- GPU, CPU, and memory usage;
- electricity;
- model storage and download bandwidth;
- startup time; and
- the accuracy or confidence tradeoff for the work performed.

The reproducible long-transcript measurements used for the current operational
comparison are documented in
[README_INFERENCE_BENCHMARK.md](README_INFERENCE_BENCHMARK.md).

The measurements in this document were collected on the project laptop on
2026-09-27. They are local observations, not performance guarantees.

## Backends covered

| UI backend | API value | Main model path |
|---|---|---|
| Smart hybrid (fast) | `hybrid` | MiniCPM first; direct Ollama only on fallback |
| OpenJev native (fast) | `openjev` | MiniCPM through `llama.cpp` |
| OpenJev native Gemma (experimental) | `openjev_gemma` | Gemma 4 E4B through `llama.cpp` |
| OpenJev with Ollama | `openjev_ollama` | Gemma through OpenJev generation |
| Ollama direct | `ollama` | Gemma through the direct Python adapter |

## Direct monetary cost

The application does not call a paid inference API. After the laptop and models
are available:

```text
Cloud API charge per request = $0 USD
Cloud token charge           = $0 USD
Internet required per request = No
```

This does not mean inference is free. Electricity, disk space, hardware time,
and operator time still have costs. Model downloads require internet bandwidth,
but classification against localhost does not.

All monetary examples in this document are in US dollars (USD). Hardware
purchase cost, maintenance, and staff time are not included.

## Controlled long-transcript comparison

Every backend classified the same 14,705-character synthetic transcript. Safe
preprocessing produced 14,704 characters and four speaker-aware chunks. The
first four backends received one warm-up followed by three measured sequential
runs. OpenJev with Ollama has one successful warm measurement because an
earlier sustained pass lost the FastAPI connection after its warm-up.

| Backend | Mean application latency | Range | Relative to MiniCPM | Returned intent | Evidence type |
|---|---:|---:|---:|---|---|
| Smart hybrid, native accepted | 1.873 s | 1.862–1.891 s | 0.98× | `eligibility_check` | Native token logprobs |
| OpenJev native MiniCPM | 1.910 s | 1.880–1.928 s | 1.00× | `eligibility_check` | Native token logprobs |
| OpenJev native Gemma | 4.914 s | 4.840–4.954 s | 2.57× | `eligibility_check` | Native token logprobs |
| Ollama direct | 35.491 s | 10.706–51.885 s | 18.58× | `eligibility_check` | Categorical label only |
| OpenJev with Ollama | 232.146 s | one successful run | 121.53× | `eligibility_check` | Generated normalized weights |

Smart hybrid accepted MiniCPM because all final native confidences exceeded the
`0.65` fallback threshold. Its small apparent advantage over the standalone
MiniCPM run is cache and run-order noise, not a faster algorithm.

The Ollama-direct spread is operationally important. All model services were
loaded on an 8 GB GPU with little free VRAM, so model/context reuse, offloading,
and contention caused much greater variance than on the native paths. These
figures are observations of this laptop in `all` mode, not universal model
benchmarks.

All five backends returned the expected labels for this deliberately clear
transcript. That is one correctness case, not an accuracy percentage. Overall
accuracy requires a labeled multi-call evaluation set.

## Why each backend costs what it does

### Smart hybrid (fast)

```text
Always pay MiniCPM cost
        +
Pay direct Ollama cost only when native output is uncertain
```

Approximate expected latency can be estimated as:

```text
expected latency = native latency + fallback rate × Ollama fallback latency
```

Using the long-test means of 1.91 seconds for native MiniCPM and 35.49 seconds
for direct Ollama under the measured `all`-mode load:

| Fallback rate | Approximate average latency |
|---:|---:|
| 0% | 1.91 s |
| 10% | 5.46 s |
| 25% | 10.78 s |
| 50% | 19.66 s |
| 100% | 37.40 s |

The actual fallback request can differ because prompt caching, transcript size,
and model load state change both measurements. Hybrid is generally the best
average-cost choice when most transcripts are clear and only a minority need
Gemma.

### OpenJev native MiniCPM

This is the lowest predictable compute-cost backend:

- one small 2B-class Q4 model;
- one-token direct decisions for each task;
- full token-logprob distributions;
- approximately 200–400 ms for tested one-chunk transcripts.

Its tradeoff is model capacity. It can safely return `unknown` when the winning
label does not meet the configured threshold, as happened with the earlier
52.8% `doctor_office` caller-type result.

### OpenJev native Gemma

This uses the larger Gemma 4 E4B Instruct Q4_0 model but retains OpenJev's cheap
direct-token decision method. On the four-chunk benchmark it averaged 4.914
seconds, about 2.57 times the MiniCPM mean, and returned complete token-logprob
distributions.

In the tested examples, caller-type accuracy improved, but that observation is
not enough to establish overall accuracy. Run the complete evaluation set
before selecting it as the production default.

### Ollama direct

The direct adapter makes one structured Ollama request containing every task
for each transcript chunk. It asks for only the selected labels, so it is much
cheaper than asking Gemma to generate every option weight.

Its returned labels are categorical:

```text
confidence: null
probabilities: null
```

This reduces generation work, but it also removes confidence-based abstention.

### OpenJev with Ollama

This is the most expensive path in the current implementation. OpenJev normally
creates one Ollama generation request per task and asks Gemma to produce a
numeric weight for every allowed option.

For the current three tasks, that means for every transcript chunk:

- the transcript is processed repeatedly;
- three model generations are requested;
- approximately 35 option weights are generated;
- OpenJev validates and normalizes the generated objects.

It took 232.146 seconds for the four-chunk benchmark—approximately 121.5 times
the MiniCPM mean. It returned the expected labels, but the extra generated
distributions made it far more expensive than either native direct path.

## Model storage cost

| Model package | Local size | Purpose |
|---|---:|---|
| MiniCPM5-2B Q4_K_M GGUF | 1.45 GiB | Fast native OpenJev |
| Gemma 4 E4B Instruct Q4_0 GGUF | 4.28 GiB | Native Gemma OpenJev |
| Ollama `gemma4:e4b` blob Q4_K_M | 8.95 GiB | Ollama backends |
| **Combined model storage** | **14.68 GiB** | All five UI choices |

The native Gemma and Ollama Gemma files cannot currently be deduplicated. The
Ollama multimodal blob is not a standalone upstream-`llama.cpp` model, so the
native backend needs the separate compatible GGUF.

Additional storage is used by Ollama metadata, `llama.cpp` binaries, application
dependencies, logs, and caches. The model files dominate the total.

## Startup and download cost

| Component | First use | Later startup |
|---|---|---|
| MiniCPM native | Existing project model | A few seconds |
| Native Gemma | Downloads 4.28 GiB; about 7 minutes on the measured connection | About 5 seconds from cache |
| Ollama Gemma | Requires `ollama pull` once | Loaded on demand; retained according to `keep_alive` |

First-run download time is primarily a bandwidth cost and varies with network
speed. It should not be included in normal per-request latency.

## Memory cost when every backend is running

After running every backend and leaving the Ollama model warm, the measured
snapshot was:

```text
NVIDIA GPU memory: 7,844 MiB used / 8,188 MiB total
GPU memory free:      113 MiB
GPU power draw:       23.52 W at the sampled instant
```

Approximate Windows process working sets at that moment:

| Process | Port | Working set |
|---|---:|---:|
| MiniCPM `llama-server` | 18080 | 2.76 GiB |
| Gemma `llama-server` | 18081 | 3.65 GiB |
| Each OpenJev service | 8090–8092 | 0.01–0.02 GiB |
| FastAPI | 8000 | 0.05 GiB |

Ollama reported its active Gemma allocation as 3.2 GB and 100% GPU at that
instant. Process working-set figures include mapped files and should not be
added together as an exact physical-RAM total.

The important operational result is that `all` mode can consume approximately
96% of GPU memory after Ollama becomes warm. This leaves little headroom and can
cause variable latency, model offloading, or allocation failures under other GPU
workloads.

## Electricity-cost calculation

Use a wall-power meter for an accurate total-laptop measurement. `nvidia-smi`
reports GPU power only and omits CPU, memory, display, fans, and conversion
losses.

For a measured average system power value:

```text
energy per request (kWh) = watts × latency_seconds / 3,600,000
cost per request         = energy_kWh × electricity_price_per_kWh
```

Example only, assuming 100 W total system power while processing and an
electricity price of **$0.15 USD/kWh**:

| Backend | Measured latency | Cost per 10,000 requests | Cost per 1,000,000 requests |
|---|---:|---:|---:|
| Smart hybrid, native accepted | 1.873 s | $0.078 | $7.81 |
| Native MiniCPM | 1.910 s | $0.080 | $7.96 |
| Native Gemma | 4.914 s | $0.205 | $20.48 |
| Ollama direct | 35.491 s | $1.479 | $147.88 |
| OpenJev with Ollama | 232.146 s | $9.673 | $967.27 |

These USD values are estimates, not measured billing figures. They count only
the assumed energy consumed during inference. They exclude idle time, startup,
model downloads, hardware purchase, and maintenance. Replace 100 W and
$0.15/kWh with wall-meter data and the actual local electricity tariff.

For idle operation, a device drawing `W` watts continuously costs:

```text
monthly kWh = W / 1000 × 24 × 30
monthly cost = monthly kWh × electricity tariff
```

At the same illustrative **$0.15 USD/kWh** rate:

| Operating pattern | Energy per 30-day month | Electricity cost per month |
|---|---:|---:|
| 100 W for 8 hours/day | 24.00 kWh | $3.60 |
| 100 W continuously (24/7) | 72.00 kWh | $10.80 |
| 23.52 W continuously (sampled GPU power only) | 16.93 kWh | $2.54 |

The 23.52 W row is a GPU-only lower-bound illustration, not the laptop's total
operating cost. A wall-power meter is needed to capture the CPU, memory,
display, fans, power-conversion losses, idle periods, and changing workload.

## Running every backend

Start every inference backend with:

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

This mode is useful for side-by-side testing. It is not the lowest-cost way to
operate continuously because multiple model runtimes remain resident.

## Lower-cost operating modes

### Lowest latency and resource use

```powershell
.\run.ps1 -OpenJevEngine llama.cpp
```

Use **OpenJev native (fast)**.

### Higher-capacity direct decisions

```powershell
.\run.ps1 -OpenJevEngine gemma.cpp
```

Use **OpenJev native Gemma (experimental)**. This avoids simultaneous model
competition and gives a more meaningful Gemma benchmark.

### Ollama only

```powershell
$env:INFERENCE_BACKEND = 'ollama'
.\run.ps1 -SkipOpenJev
```

Use **Ollama direct**. This avoids both `llama.cpp` model processes.

### Balanced normal operation

```powershell
Remove-Item Env:INFERENCE_BACKEND -ErrorAction SilentlyContinue
.\run.ps1 -OpenJevEngine both -OllamaModel gemma4:e4b
```

Use **Smart hybrid (fast)**. This keeps the existing fast native path and uses
Ollama only when the confidence policy requires it.

## Cost-based recommendation

| Priority | Recommended backend |
|---|---|
| Lowest consistent latency | OpenJev native MiniCPM |
| Better observed accuracy with probabilities | OpenJev native Gemma |
| Lowest average cost with uncertainty fallback | Smart hybrid |
| Explicit use of a selectable Ollama model | Ollama direct |
| OpenJev Ollama-adapter testing | OpenJev with Ollama |

For this laptop, **Smart hybrid** remains the practical default. Use **native
Gemma** when its improved evaluation accuracy justifies roughly 2–3 times the
MiniCPM latency. Keep **OpenJev with Ollama** for adapter testing rather than
routine classification because its generation cost is much higher.

## Reproducing the comparison

Latency should be measured with:

- the same transcript;
- the same chunk count;
- models already loaded for warm measurements;
- no unrelated GPU workload;
- several repetitions rather than one request; and
- labels checked against known expected answers.

Run the automated evaluation for accuracy and the benchmark separately for
latency. A cheaper incorrect answer is not a useful optimization.
