param(
    [int]$ApiPort = 8000
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot

# Publish only the MiniCPM-backed native OpenJev interface. Gemma and Ollama
# may be installed on the computer, but this launcher does not start or expose
# either of them.
$env:INFERENCE_BACKEND = 'openjev'
$env:INFERENCE_BACKENDS = 'openjev'

& (Join-Path $ProjectRoot 'run.ps1') `
    -OpenJevEngine llama.cpp `
    -ApiPort $ApiPort

exit $LASTEXITCODE
