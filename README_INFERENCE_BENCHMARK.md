# Long-Transcript Inference Benchmark

This document records one reproducible local benchmark across every inference
backend in the OpenJev Call Intelligence UI. It includes the exact synthetic
transcript, raw timing samples, returned labels, confidence semantics,
architecture, illustrative USD energy cost, and limitations.

This is a latency and pipeline-correctness test. It is **not** an accuracy study
and it is not a production capacity guarantee.

## Result in one table

The source transcript has 14,705 characters. Preprocessing removes the final
line ending, producing 14,704 processed characters and four speaker-aware
chunks of 3,916, 3,994, 3,875, and 3,934 characters.

| UI backend | API backend | Model | Warm application latency | Returned labels | Evidence |
|---|---|---|---:|---|---|
| Smart hybrid (fast) | `hybrid` | MiniCPM accepted; Ollama skipped | 1.873 s mean (3 runs) | `yes / doctor_office / eligibility_check` | Native token logprobs |
| OpenJev native (fast) | `openjev` | `minicpm5-2b-q4_k_m` | 1.910 s mean (3 runs) | `yes / doctor_office / eligibility_check` | Native token logprobs |
| OpenJev native Gemma (experimental) | `openjev_gemma` | `gemma4-e4b-native` | 4.914 s mean (3 runs) | `yes / doctor_office / eligibility_check` | Native token logprobs |
| Ollama direct | `ollama` | `gemma4:e4b` | 35.491 s mean (3 runs) | `yes / doctor_office / eligibility_check` | Categorical votes |
| OpenJev with Ollama | `openjev_ollama` | `gemma4:e4b` | 232.146 s (1 successful run) | `yes / doctor_office / eligibility_check` | Generated normalized weights |

The expected answer was healthcare `yes`, caller type `doctor_office`, and
intent `eligibility_check`. Every successful backend returned that answer.

## Plain-language interpretation

- **Native MiniCPM** read the long call in four pieces and answered quickly.
- **Smart hybrid** first used the same MiniCPM path. It decided the native
  answers were confident enough, so it never called Ollama.
- **Native Gemma** did the same one-letter decision work using a larger model.
  It took about 2.57 times as long as MiniCPM and returned stronger confidence
  on this one deliberately clear call.
- **Ollama direct** asked Gemma for one small JSON object per chunk. It got the
  labels right, but its runtime varied widely while all models competed for the
  laptop's 8 GB GPU.
- **OpenJev with Ollama** asked Gemma to generate a complete option-weight table
  separately for every task and every chunk. Four chunks times three tasks
  produced 12 generations, making it by far the slowest route.

“Stronger confidence” is not the same as “higher accuracy.” Accuracy measures
how often predictions match ground truth over many labeled calls. Confidence
describes how concentrated one model response was. This single call proves
that the pipeline can produce the expected result; it cannot establish an
accuracy percentage.

## Test environment and method

- Date: 2026-09-27.
- Host: Windows 11 laptop.
- GPU: NVIDIA RTX 4060 Laptop GPU with 8 GB VRAM.
- Startup mode: `-OpenJevEngine all` with all model services resident.
- Ollama model: `gemma4:e4b`, 8B Q4_K_M.
- Native MiniCPM: MiniCPM5-2B Q4_K_M through `llama.cpp`.
- Native Gemma: Gemma 4 E4B Instruct Q4_0 through `llama.cpp`.
- Chunk configuration: 4,000 characters, 400-character speaker-aware overlap.
- Request order: hybrid, native MiniCPM, native Gemma, Ollama direct, then
  OpenJev with Ollama.
- First four backends: one warm-up, then three sequential measured requests.
- OpenJev with Ollama: one successful warm measurement after the first
  sustained pass lost its FastAPI connection.
- Latency source: `model.latency_ms`, measured inside the Python classifier.
- Cost assumption: 100 W average whole-system power during inference and
  `$0.15 USD/kWh`.

Running every model at once is useful for UI comparison, but it is not an
isolated model benchmark. Earlier resource inspection showed about 96% of VRAM
in use after Ollama became warm. This explains why Ollama results can change
with cache state, offloading, model residency, and run order.

## Raw latency samples

| Backend | Run 1 | Run 2 | Run 3 | Mean | Median | Range |
|---|---:|---:|---:|---:|---:|---:|
| Smart hybrid, native accepted | 1,862.053 ms | 1,891.458 ms | 1,866.175 ms | 1,873.229 ms | 1,866.175 ms | 29.405 ms |
| Native MiniCPM | 1,922.237 ms | 1,880.142 ms | 1,928.129 ms | 1,910.169 ms | 1,922.237 ms | 47.987 ms |
| Native Gemma | 4,839.810 ms | 4,953.992 ms | 4,948.168 ms | 4,913.990 ms | 4,948.168 ms | 114.182 ms |
| Ollama direct | 51,885.402 ms | 43,881.747 ms | 10,705.754 ms | 35,490.968 ms | 43,881.747 ms | 41,179.648 ms |
| OpenJev with Ollama | 232,145.623 ms | — | — | 232,145.623 ms | 232,145.623 ms | One run |

Three samples are enough to expose large differences, but not enough to claim
a stable p95 or throughput service-level objective. The script prints an
interpolated p95 for convenience; do not treat a three-sample p95 as a
production percentile.

## Returned evidence

| Backend | Healthcare | Caller type | Intent | Probability source |
|---|---:|---:|---:|---|
| Smart hybrid | `yes` 99.96% | `doctor_office` 95.89% | `eligibility_check` 84.53% | MiniCPM token logprobs |
| Native MiniCPM | `yes` 99.96% | `doctor_office` 95.89% | `eligibility_check` 84.53% | MiniCPM token logprobs |
| Native Gemma | `yes` 100.00% | `doctor_office` 99.49% | `eligibility_check` 95.57% | Gemma token logprobs |
| Ollama direct | `yes` | `doctor_office` | `eligibility_check` | No probabilities; categorical chunk votes |
| OpenJev with Ollama | `yes` 100.00% | `doctor_office` 95.86% | `eligibility_check` 65.47% | Model-generated weights normalized by OpenJev |

Native confidence and OpenJev-with-Ollama confidence do not have the same
meaning. Native values come from constrained first-token logits. The
OpenJev-with-Ollama values are based on numbers the model generated as text.
Ollama direct intentionally returns `null` confidence rather than pretending a
categorical choice is a measured probability.

## Why the work differs by backend

The current configuration has four chunks (`C = 4`) and three tasks (`T = 3`).

### Smart hybrid

```text
4 chunks -> MiniCPM native pass -> aggregate -> inspect all task confidences
                                      |
                    all >= 0.65 and none unknown
                                      |
                                  return native
```

This test used 4 OpenJev requests and 12 native one-token decisions. Ollama was
not called. If even one final task were unknown or below 0.65, hybrid would run
another complete four-request Ollama-direct pass and return the Ollama result.
It does not average MiniCPM and Ollama together.

### OpenJev native MiniCPM

```text
4 chunks -> OpenJev :8090 -> 3 queued tasks per chunk
                              -> llama.cpp :18080
                              -> one constrained answer token per task
                              -> full allowed-label logprob vectors
```

OpenJev receives one HTTP request per chunk containing all tasks. Internally it
schedules each task separately. The model reads the entire prompt even though
the output is one letter. Python combines four probability vectors per task and
applies each task's `0.60` threshold.

### OpenJev native Gemma

The service flow is identical to Native MiniCPM, but ports `8092` and `18081`
lead to the larger Gemma GGUF. The four chunks produced 12 direct decisions.
Gemma reported 12,214 prompt tokens across those decisions versus 11,796 from
MiniCPM; tokenizer counts are model-specific and are not directly comparable.

The original long test exposed `missing option logit for G`. Gemma assigned the
valid but irrelevant caller-type option `G` a logprob near `-20.7`, below the
first 512 returned tokens in chunks 2–4. OpenJev needs every option logit before
softmax, so it correctly failed instead of inventing a value. The compatibility
patch now requests at least 1,024 top logprobs; the fixed request returns all
labels and retains explicit failure if any label is still absent.

### Ollama direct

```text
4 chunks -> Python Ollama adapter -> 4 /api/chat calls
                                      each returns one JSON object:
                                      healthcare + caller type + intent
```

This route makes one generation per chunk, not one per task. JSON Schema limits
each field to configured labels, `temperature` is zero, thinking is disabled,
and Python validates the output. The four calls reported 5,626 prompt tokens.
Because the response has labels rather than distributions, Python aggregates
categorical votes.

### OpenJev with Ollama

```text
4 chunks -> OpenJev :8091 -> 3 task generations per chunk -> Ollama :11434
                                              |
                       healthcare: generate all 2 weights
                       caller type: generate all 12 weights
                       intent: generate all 20 weights
```

This path made 12 Ollama generations and reported 13,574 prompt tokens. It also
generated approximately 34 option weights per chunk, or about 136 values for
the complete transcript. OpenJev validates keys and values, normalizes complete
non-negative weights, calculates entropy confidence, and returns the result to
Python for cross-chunk aggregation.

The first sustained benchmark completed its warm-up but later lost the FastAPI
connection with Windows error `10054`. OpenJev and Ollama remained healthy.
After FastAPI was restarted, one warm measured request succeeded in 232.146
seconds. This failure is part of the operational result: prolonged requests
need process supervision and timeouts appropriate to the selected path.

## Illustrative USD electricity cost

Local inference has no cloud token or API fee. The table below estimates only
active-request electricity:

```text
energy per request (kWh) = watts × latency_seconds / 3,600,000
cost per request = energy per request × USD per kWh
```

At an assumed 100 W whole-system draw and `$0.15 USD/kWh`:

| Backend | Latency used | Per 10,000 requests | Per 1,000,000 requests |
|---|---:|---:|---:|
| Smart hybrid, native accepted | 1.873 s | $0.078 | $7.81 |
| Native MiniCPM | 1.910 s | $0.080 | $7.96 |
| Native Gemma | 4.914 s | $0.205 | $20.48 |
| Ollama direct | 35.491 s | $1.479 | $147.88 |
| OpenJev with Ollama | 232.146 s | $9.673 | $967.27 |

These are formula estimates, not electricity-meter measurements or cloud
prices. They exclude idle power, startup, downloads, hardware purchase,
maintenance, and concurrency. Replace both assumptions with a wall meter and
the actual local electricity tariff before budgeting.

## Reproduce the benchmark

Start all services:

```powershell
# Run from the project directory.
Remove-Item Env:INFERENCE_BACKEND -ErrorAction SilentlyContinue
.\run.ps1 -OpenJevEngine all -OllamaModel gemma4:e4b
```

In another terminal, run the UI if desired:

```powershell
# Run from the project directory.
.\run-ui.ps1
```

In a third terminal, benchmark all backends:

```powershell
# Run from the project directory.
.\venv\Scripts\python.exe scripts\benchmark_all_backends.py `
  --transcript samples\very_long_benchmark_transcript.txt `
  --warmups 1 --runs 3 `
  --watts 100 --electricity-usd-per-kwh 0.15
```

OpenJev with Ollama can take several minutes per long request. To measure only
one backend or shorten the run:

```powershell
.\venv\Scripts\python.exe scripts\benchmark_all_backends.py `
  --backend openjev_ollama --warmups 0 --runs 1
```

The script prints machine-readable JSON, individual samples, routing details,
returned labels, token counts, and illustrative costs. Run isolated backends
separately when comparing model speed without GPU contention.

## Exact synthetic test transcript

The same text is stored in
[`samples/very_long_benchmark_transcript.txt`](samples/very_long_benchmark_transcript.txt).
It contains no real patient information.

```text
Virtual Agent: Thank you for calling Northstar Health Plan provider services. This call is a synthetic benchmark and contains no real patient information. Please state your organization and the reason for your call.
Caller: Hello, this is Sarah Mitchell from the billing and insurance verification team at Dr. Michael Patel's cardiology office. I am calling to verify active medical insurance eligibility and benefit coverage for a fictional patient before a scheduled outpatient cardiology consultation.
Virtual Agent: Thank you. Are you calling as the patient, a family member, a physician office, a hospital, a pharmacy, or another organization?
Caller: I am calling on behalf of the physician's office. Dr. Patel is the treating cardiologist, and I am an authorized staff member in his office. I am not the patient and I am not calling from an insurance company.
Virtual Agent: Which topic best describes the primary purpose of the call: member eligibility, claim status, prior authorization, payment, referral, or general information?
Caller: The primary purpose is member eligibility. We need to confirm that coverage is active on the planned date of service and understand the applicable specialist benefits. We are not checking a submitted claim and are not requesting payment.
Virtual Agent: Please provide the fictional member identifier used for this benchmark.
Caller: The synthetic member identifier is NSH-482916. It was created only for local testing and does not identify a real person.
Virtual Agent: Please provide the fictional patient's name and date of birth.
Caller: The benchmark patient is Jordan Example, date of birth January 15, 1980. Both the name and date are fabricated test data.
Virtual Agent: What is the planned date and place of service?
Caller: The planned date is October 14, 2026, at Dr. Patel's outpatient cardiology office. The visit is an initial specialist consultation, not an inpatient admission or emergency service.
Virtual Agent: I will review the test eligibility record. While that runs, please confirm the provider name, specialty, and billing identifier.
Caller: The provider is Dr. Michael Patel, cardiology. The fictional billing identifier is 1234567890, and the practice is Northstar Cardiology Associates. These values are synthetic and must not be used for an actual transaction.
Virtual Agent: The benchmark record shows the member as active. Would you like the effective date and plan type?
Caller: Yes. Please confirm the effective date, termination status, plan type, and whether the cardiology office is treated as participating for this test scenario.
Virtual Agent: The synthetic plan became effective January 1, 2026, has no termination date on file, and is represented as a preferred-provider plan. The test directory marks the physician as participating.
Caller: Thank you. Please also confirm whether the eligibility response applies on October 14, 2026, because that is the specific service date our office is checking.
Virtual Agent: For this fictional scenario, the plan is active on October 14, 2026. Eligibility is a snapshot and does not guarantee payment of a future claim.
Caller: Understood. Our office knows that eligibility information is not a guarantee of payment. We still need the specialist office-visit benefit so that we can give the fictional patient an estimate.
Virtual Agent: The synthetic in-network specialist benefit has a forty-dollar copayment after the normal plan rules. Deductible and coinsurance may apply to diagnostic services performed separately.
Caller: Does the forty-dollar amount apply to the consultation itself, and are electrocardiogram or echocardiogram services handled under separate diagnostic benefits?
Virtual Agent: Correct. In the benchmark record, the office consultation uses the specialist copayment. Diagnostic tests are evaluated under their own benefit categories and medical-necessity rules.
Caller: Please confirm the annual deductible and how much of it the fictional member has met. Again, this is only to exercise a long local classification request.
Virtual Agent: The synthetic individual deductible is one thousand dollars, with six hundred dollars shown as met. The synthetic family deductible is two thousand dollars, with nine hundred dollars shown as met.
Caller: What are the in-network coinsurance and out-of-pocket maximum in this fictional plan?
Virtual Agent: After applicable deductible requirements, the benchmark plan uses twenty percent member coinsurance for covered diagnostic services. The individual out-of-pocket maximum is five thousand dollars, of which one thousand two hundred dollars is shown as accumulated.
Caller: I would like to make sure we are still discussing current eligibility and benefit verification, rather than a claim decision. No service has occurred and there is no claim number.
Virtual Agent: That is correct. This interaction is an eligibility and benefits inquiry from a doctor's office. There is no claim being reviewed.
Caller: Does the plan require a referral from a primary care physician for the initial cardiology consultation?
Virtual Agent: The fictional preferred-provider plan does not require a primary-care referral for an in-network specialist consultation. This statement does not replace any prior-authorization requirement for specific procedures.
Caller: We are not requesting an authorization during this call, but please tell me whether the consultation itself is listed as requiring prior authorization.
Virtual Agent: The benchmark rules show no prior authorization for the consultation. Advanced imaging and selected cardiac procedures may require authorization, which would be a separate workflow.
Caller: That distinction is helpful. Our primary request remains verification that the member is eligible and that specialist benefits exist on the scheduled date.
Virtual Agent: Confirmed. The synthetic member is eligible on the requested date, and the plan includes an in-network specialist office-visit benefit.
Caller: Could you identify whether telehealth and an in-person cardiology visit use the same specialist copayment in this test record?
Virtual Agent: In this benchmark scenario, an in-network synchronous video consultation and an in-person specialist office visit both show a forty-dollar copayment, subject to coding and plan rules.
Caller: We currently expect an in-person visit. If the provider orders an echocardiogram after the consultation, would that change today's eligibility answer?
Virtual Agent: It would not change the member's eligibility status. It could change the benefit category, estimated member responsibility, or authorization requirements for the later service.
Caller: Please confirm whether the provider directory record is active at the practice address used by this fictional case.
Virtual Agent: The synthetic directory lists Dr. Patel as active at 100 Example Avenue, Suite 200, Testville. The address is fictitious and included only for benchmark completeness.
Caller: Thank you. Is there a separate facility fee expected for the scheduled office consultation?
Virtual Agent: The benchmark record represents a freestanding physician office, so no separate hospital outpatient facility fee is indicated for the consultation itself.
Caller: Please review whether the coverage is primary in this synthetic record or whether another health plan is listed first.
Virtual Agent: The test coordination-of-benefits record lists Northstar Health Plan as primary and contains no active secondary medical carrier.
Caller: If the patient later provides different coordination-of-benefits information, our office would update the account, but today we only need the current eligibility response.
Virtual Agent: Correct. Today’s response reflects the fictional information currently present in the benchmark record.
Caller: Can you confirm whether routine laboratory work ordered at the visit is covered when sent to an in-network laboratory?
Virtual Agent: The synthetic plan includes outpatient laboratory benefits when medically necessary and performed by an eligible participating laboratory. Exact member cost depends on the billed service.
Caller: We are not calling from a laboratory. I am still Sarah from Dr. Patel's office, asking on behalf of the physician practice before the appointment.
Virtual Agent: Understood. The caller organization remains classified as the doctor's office, and the primary transaction remains an eligibility check.
Caller: Please provide the fictional payer identifier that the practice would place on an electronic eligibility inquiry.
Virtual Agent: The benchmark payer identifier is TEST123. It is intentionally invalid outside this local demonstration.
Caller: Does this synthetic plan show coverage for preventive cardiovascular counseling, or would that require diagnosis and procedure codes before a definitive benefit response?
Virtual Agent: A general preventive benefit appears in the test record, but a definitive response would require the exact service code, diagnosis context, provider status, and plan rules.
Caller: That makes sense. I do not want to convert this into a coding review. The main operational question is whether the patient can be treated as currently insured for the upcoming specialist visit.
Virtual Agent: Yes. Based on the fictional record, the member can be treated as currently eligible for the scheduled date, subject to the standard disclaimer that benefits are not a guarantee of claim payment.
Caller: Are there any waiting periods or pre-existing-condition exclusions shown for the consultation in this benchmark plan?
Virtual Agent: No waiting period or pre-existing-condition exclusion is shown in the synthetic medical eligibility record for the consultation.
Caller: Is the specialist copayment collected once per date of service, or could the benchmark plan apply it more than once when multiple professionals participate?
Virtual Agent: The example benefit is generally assessed per specialist office visit. Separate professional or diagnostic services can produce additional member responsibility according to coding and adjudication.
Caller: Please confirm that the member's fictional policy is an individual policy rather than a workers' compensation or automobile medical claim.
Virtual Agent: Confirmed. The benchmark record is ordinary employer-sponsored medical coverage and is not workers' compensation, automobile coverage, or another liability claim.
Caller: Does the eligibility record include prescription coverage, and if so, is it relevant to this request?
Virtual Agent: A synthetic pharmacy benefit is present, but it is not the primary subject of this call. The requested transaction concerns medical eligibility and specialist benefits.
Caller: Agreed. We do not need a prescription refill, medication question, payment arrangement, or claim status today.
Virtual Agent: Noted. I will keep the interaction categorized as an eligibility inquiry from a physician office.
Caller: Before we finish, could you repeat the active coverage dates and the expected in-network specialist amount for the fictional consultation?
Virtual Agent: The synthetic effective date is January 1, 2026, there is no termination date, coverage is active on October 14, 2026, and the listed in-network specialist copayment is forty dollars.
Caller: Please also repeat the disclaimer so our test transcript captures it clearly.
Virtual Agent: Eligibility and benefit information reflects the fictional record at the time of inquiry. It is not a guarantee of payment. Final responsibility depends on actual services, coding, medical necessity, authorization, coordination of benefits, and claim adjudication.
Caller: I acknowledge the disclaimer. No claim has been filed, and our office is not asking you to guarantee reimbursement.
Virtual Agent: Is there another eligibility detail needed for the upcoming visit?
Caller: Yes. Does the plan show an age or visit-frequency restriction for an initial outpatient cardiology evaluation?
Virtual Agent: The benchmark benefit does not show an age restriction for the consultation. Frequency limitations can depend on coding and clinical circumstances, so the response remains subject to plan terms.
Caller: Does the synthetic member need to choose a designated primary care physician before seeing this in-network cardiologist?
Virtual Agent: No designated primary care physician is required under the fictional preferred-provider plan, although primary care coordination may still be encouraged.
Caller: Are out-of-network cardiology services covered if the appointment location changes?
Virtual Agent: The test plan includes an out-of-network benefit with a higher deductible and coinsurance, but balance billing may apply. The currently scheduled physician and location are represented as in network.
Caller: We intend to keep the appointment at the participating location. I am confirming this only because provider network status is part of a complete eligibility check.
Virtual Agent: Understood. For the scheduled participating location, use the in-network specialist information already provided.
Caller: If an authorization becomes necessary for a later procedure, should our office open a new request instead of treating this eligibility call as authorization approval?
Virtual Agent: Yes. Eligibility verification does not create an authorization. A separate prior-authorization submission would be needed for any service that requires it.
Caller: Excellent. Please state the primary call category one final time for quality review.
Virtual Agent: The primary category is member eligibility and benefit verification. The caller is staff from a doctor's office, and the context is healthcare-related.
Caller: That matches our purpose. I am Sarah Mitchell from Dr. Michael Patel's cardiology office, and I called to verify the fictional patient's active coverage and specialist benefits before the scheduled appointment.
Virtual Agent: Your synthetic reference number is ELIG-TEST-2026-1042. This number is not valid in any production system.
Caller: I have recorded the fictional reference number. Thank you for reviewing the eligibility, effective date, network status, specialist copayment, deductible, and benefit limitations.
Virtual Agent: You are welcome. Is there anything else related to this benchmark eligibility inquiry?
Caller: No. The physician office has the test information it needs. Please close the call without opening a claim, collecting a payment, scheduling an appointment, or submitting an authorization.
Virtual Agent: The benchmark interaction is complete. Thank you for calling Northstar Health Plan provider services.
```

## What to test next

For model-quality decisions, run the labeled evaluation dataset per backend and
report accuracy, per-label precision/recall/F1, confusion matrices, abstention
rate, and latency distributions. For production-capacity decisions, run at
least dozens of isolated warm requests at the intended concurrency, measure
wall power, and record p50/p95/p99 latency rather than extrapolating from this
small engineering benchmark.
