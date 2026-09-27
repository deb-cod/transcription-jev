param(
    [int]$ApiPort = 8000,
    [string]$NativeGemmaRepo = 'ggml-org/gemma-4-E4B-it-GGUF:Q4_0',
    [string]$NativeGemmaGguf = ''
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$DefaultGemmaGguf = Join-Path $ProjectRoot 'models\gemma4-e4b\gemma-4-E4B-it-Q4_0.gguf'

if (-not $NativeGemmaGguf -and (Test-Path -LiteralPath $DefaultGemmaGguf -PathType Leaf)) {
    $NativeGemmaGguf = $DefaultGemmaGguf
}

# Keep both the API default and its published backend catalog limited to the
# native Gemma path. This does not start, query, or download an Ollama model or
# the legacy MiniCPM model.
$env:INFERENCE_BACKEND = 'openjev_gemma'
$env:INFERENCE_BACKENDS = 'openjev_gemma'

& (Join-Path $ProjectRoot 'run.ps1') `
    -OpenJevEngine gemma.cpp `
    -ApiPort $ApiPort `
    -NativeGemmaRepo $NativeGemmaRepo `
    -NativeGemmaGguf $NativeGemmaGguf

exit $LASTEXITCODE
