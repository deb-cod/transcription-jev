param(
    [int]$ApiPort = 8000,
    [string]$NativeGemmaRepo = 'ggml-org/gemma-4-E2B-it-GGUF:Q4_0',
    [string]$NativeGemmaGguf = ''
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$DefaultGemmaGguf = Join-Path $ProjectRoot 'models\gemma4-e2b\gemma-4-E2B-it-Q4_0.gguf'

if (-not $NativeGemmaGguf -and (Test-Path -LiteralPath $DefaultGemmaGguf -PathType Leaf)) {
    $NativeGemmaGguf = $DefaultGemmaGguf
}

# Publish only the native Gemma client. E2B uses the same API backend name as
# E4B because this profile runs one native Gemma model at a time.
$env:INFERENCE_BACKEND = 'openjev_gemma'
$env:INFERENCE_BACKENDS = 'openjev_gemma'

& (Join-Path $ProjectRoot 'run.ps1') `
    -OpenJevEngine gemma.cpp `
    -ApiPort $ApiPort `
    -NativeGemmaRepo $NativeGemmaRepo `
    -NativeGemmaGguf $NativeGemmaGguf `
    -NativeGemmaModelID 'gemma4-e2b-native' `
    -NativeGemmaConfig 'config\openjev-gemma-e2b.yaml' `
    -NativeGemmaDownloadSize '2.84 GB'

exit $LASTEXITCODE
