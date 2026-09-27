param(
    [switch]$SkipOpenJev,
    [int]$ApiPort = 8000,
    [ValidateSet('all', 'both', 'ollama', 'llama.cpp', 'gemma.cpp')]
    [string]$OpenJevEngine = 'both',
    [string]$OllamaModel = 'gemma4:e4b',
    [string]$NativeGemmaRepo = 'ggml-org/gemma-4-E4B-it-GGUF:Q4_0',
    [string]$NativeGemmaGguf = ''
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$OpenJevRoot = Join-Path $ProjectRoot 'vendor\openjev'
$Python = Join-Path $ProjectRoot 'venv\Scripts\python.exe'
$Llama = Join-Path $OpenJevRoot 'third_party\llama.cpp\b11056\llama-server.exe'
$Model = Join-Path $OpenJevRoot 'models\MiniCPM5-2B-Q4_K_M.gguf'
$GemmaModelID = 'gemma4-e4b-native'
$Decision = Join-Path $OpenJevRoot 'bin\decision-test.exe'
$GemmaDecision = Join-Path $OpenJevRoot 'bin\decision-test-gemma.exe'
$LlamaConfig = Join-Path $OpenJevRoot 'settings\decision-test.yaml'
$OllamaConfig = Join-Path $OpenJevRoot 'settings\decision-ollama.yaml'
$GemmaConfig = Join-Path $ProjectRoot 'config\openjev-gemma.yaml'
$GemmaModule = Join-Path $OpenJevRoot 'features\decision-test\go.mod'
$RuntimeLogs = Join-Path $ProjectRoot 'tmp\runtime'
$UsesLlama = -not $SkipOpenJev -and $OpenJevEngine -in @('all', 'both', 'llama.cpp')
$UsesGemma = -not $SkipOpenJev -and $OpenJevEngine -in @('all', 'gemma.cpp')
$UsesOllama = -not $SkipOpenJev -and $OpenJevEngine -in @('all', 'both', 'ollama')

function Assert-Gguf([string]$Path) {
    $Stream = [System.IO.File]::OpenRead($Path)
    try {
        $Header = New-Object byte[] 4
        if ($Stream.Read($Header, 0, 4) -ne 4 -or [Text.Encoding]::ASCII.GetString($Header) -ne 'GGUF') {
            throw "Native Gemma model is not a GGUF file: $Path"
        }
    } finally {
        $Stream.Dispose()
    }
}

$GemmaGgufPath = $null
if ($UsesGemma -and $NativeGemmaGguf) {
    if (-not (Test-Path -LiteralPath $NativeGemmaGguf -PathType Leaf)) {
        throw "Native Gemma GGUF not found: $NativeGemmaGguf"
    }
    $GemmaGgufPath = (Resolve-Path -LiteralPath $NativeGemmaGguf).Path
    Assert-Gguf $GemmaGgufPath
}

$RequiredFiles = @($Python)
if ($UsesLlama -or $UsesOllama) { $RequiredFiles += @($Decision) }
if ($UsesLlama) { $RequiredFiles += @($Llama, $Model, $LlamaConfig) }
if ($UsesOllama) { $RequiredFiles += @($OllamaConfig) }
if ($UsesGemma) { $RequiredFiles += @($Llama, $GemmaConfig, $GemmaModule) }
if ($GemmaGgufPath) { $RequiredFiles += @($GemmaGgufPath) }
foreach ($Required in $RequiredFiles) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Required file not found: $Required. Follow README setup first."
    }
}
if ($UsesGemma -and -not (Get-Command go -ErrorAction SilentlyContinue)) {
    throw "Go is required to build the native Gemma OpenJev service. Install Go 1.24 or newer and reopen PowerShell."
}

New-Item -ItemType Directory -Force -Path $RuntimeLogs | Out-Null
$OwnedProcesses = @()

function Wait-ForHealth(
    [string]$Url,
    [int]$Seconds = 120,
    [System.Diagnostics.Process]$Process = $null
) {
    $Deadline = (Get-Date).AddSeconds($Seconds)
    do {
        if ($Process -and $Process.HasExited) {
            throw "Process $($Process.Id) exited before $Url became ready (exit code $($Process.ExitCode)). Check tmp/runtime logs."
        }
        try {
            $Response = Invoke-RestMethod -Uri $Url -TimeoutSec 3
            if ($Response.status -eq 'ok' -or $Response.ready -eq $true) { return }
        } catch { }
        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $Deadline)
    throw "Timed out waiting for $Url"
}

function Start-OpenJevService(
    [string]$Config,
    [int]$Port,
    [string]$ExpectedEngine,
    [string]$ExpectedModel,
    [string]$LogPrefix,
    [string]$Executable = $Decision,
    [switch]$BuildFromSource
) {
    $HealthUrl = "http://127.0.0.1:$Port/health"
    try { $Ready = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 2 }
    catch { $Ready = $null }

    if ($Ready -and ($Ready.engine -ne $ExpectedEngine -or $Ready.model -ne $ExpectedModel)) {
        throw "Port $Port already has OpenJev engine '$($Ready.engine)' model '$($Ready.model)'. Stop it before starting $ExpectedEngine / $ExpectedModel."
    }
    if ($Ready.ready) { return $null }

    if ($BuildFromSource) {
        $ModulePath = Join-Path $OpenJevRoot 'features\decision-test'
        Push-Location $ModulePath
        try {
            & go build -o $Executable ./cmd/decision-test
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to build the OpenJEV service executable at $Executable."
            }
        } finally {
            Pop-Location
        }
    }

    $Process = Start-Process -FilePath $Executable `
      -ArgumentList @('--config', $Config, '--port', $Port) `
      -WorkingDirectory $OpenJevRoot -WindowStyle Hidden `
      -RedirectStandardOutput (Join-Path $RuntimeLogs "$LogPrefix.stdout.log") `
      -RedirectStandardError (Join-Path $RuntimeLogs "$LogPrefix.stderr.log") -PassThru
    Wait-ForHealth $HealthUrl
    return $Process
}

try {
    if ($UsesLlama) {
        try { Invoke-RestMethod -Uri 'http://127.0.0.1:18080/health' -TimeoutSec 2 | Out-Null }
        catch {
            $ModelLoadStarted = Get-Date
            $LlamaProcess = Start-Process -FilePath $Llama -ArgumentList @(
                '-m', $Model, '-c', '2048', '-b', '512', '-ngl', '99', '-np', '1',
                '--jinja', '--host', '127.0.0.1', '--port', '18080'
            ) -WorkingDirectory $OpenJevRoot -WindowStyle Hidden `
              -RedirectStandardOutput (Join-Path $RuntimeLogs 'llama.stdout.log') `
              -RedirectStandardError (Join-Path $RuntimeLogs 'llama.stderr.log') -PassThru
            $OwnedProcesses += $LlamaProcess
            Wait-ForHealth 'http://127.0.0.1:18080/health'
            $ModelLoadMilliseconds = ((Get-Date) - $ModelLoadStarted).TotalMilliseconds
            Write-Host ("llama.cpp model load: {0:N0} ms" -f $ModelLoadMilliseconds)
        }
        $NativeProcess = Start-OpenJevService $LlamaConfig 8090 'llama.cpp' 'minicpm5-2b-q4_k_m' 'openjev-native'
        if ($NativeProcess) { $OwnedProcesses += $NativeProcess }
    }

    if ($UsesGemma) {
        $GemmaLlamaReady = $false
        try {
            $GemmaHealth = Invoke-RestMethod -Uri 'http://127.0.0.1:18081/health' -TimeoutSec 2
            if ($GemmaHealth.status -eq 'ok') {
                $GemmaModels = Invoke-RestMethod -Uri 'http://127.0.0.1:18081/v1/models' -TimeoutSec 3
                $RunningModel = $GemmaModels.data | Select-Object -First 1 -ExpandProperty id
                if ($RunningModel -ne $GemmaModelID) {
                    throw "Port 18081 has llama.cpp model '$RunningModel', expected '$GemmaModelID'. Stop it before starting native Gemma."
                }
                $GemmaLlamaReady = $true
            }
        } catch {
            if ($_.Exception.Message -like 'Port 18081 has*') { throw }
        }
        if (-not $GemmaLlamaReady) {
            $GemmaLoadStarted = Get-Date
            if ($GemmaGgufPath) {
                # Start-Process joins ArgumentList values into one Windows command
                # line, so an explicit quote is required for paths containing spaces.
                $GemmaSourceArguments = @('-m', ('"' + $GemmaGgufPath + '"'))
            } else {
                Write-Host "Starting native Gemma from $NativeGemmaRepo. The first run downloads about 4.6 GB; later runs use the local cache."
                $GemmaSourceArguments = @('-hf', $NativeGemmaRepo, '--no-mmproj')
            }
            $GemmaArguments = $GemmaSourceArguments + @(
                '-c', '2048', '-b', '256', '-ub', '128',
                '-ngl', 'auto', '--fit', 'on', '--fit-target', '1536', '-np', '1',
                '--jinja', '--alias', $GemmaModelID,
                '--host', '127.0.0.1', '--port', '18081'
            )
            $GemmaLlamaProcess = Start-Process -FilePath $Llama -ArgumentList $GemmaArguments `
              -WorkingDirectory $OpenJevRoot -WindowStyle Hidden `
              -RedirectStandardOutput (Join-Path $RuntimeLogs 'llama-gemma.stdout.log') `
              -RedirectStandardError (Join-Path $RuntimeLogs 'llama-gemma.stderr.log') -PassThru
            $OwnedProcesses += $GemmaLlamaProcess
            Wait-ForHealth 'http://127.0.0.1:18081/health' 1200 $GemmaLlamaProcess
            $GemmaLoadMilliseconds = ((Get-Date) - $GemmaLoadStarted).TotalMilliseconds
            Write-Host ("llama.cpp Gemma model load: {0:N0} ms" -f $GemmaLoadMilliseconds)
        }
        $GemmaProcess = Start-OpenJevService $GemmaConfig 8092 'llama.cpp' $GemmaModelID 'openjev-gemma' $GemmaDecision -BuildFromSource
        if ($GemmaProcess) { $OwnedProcesses += $GemmaProcess }
    }

    if ($UsesOllama) {
        try { $Tags = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 5 }
        catch { throw "Ollama is not reachable at http://127.0.0.1:11434. Start Ollama first." }
        $SelectedModel = $Tags.models | Where-Object { $_.name -eq $OllamaModel -or $_.model -eq $OllamaModel } | Select-Object -First 1
        if (-not $SelectedModel) {
            throw "Ollama model '$OllamaModel' is not installed. Run: ollama pull $OllamaModel"
        }
        if ($SelectedModel.capabilities -and 'completion' -notin @($SelectedModel.capabilities)) {
            throw "Ollama model '$OllamaModel' does not support text completion. Choose a completion-capable model."
        }
        $env:OPENJEV_OLLAMA_MODEL = $OllamaModel
        $OllamaProcess = Start-OpenJevService $OllamaConfig 8091 'ollama' $OllamaModel 'openjev-ollama'
        if ($OllamaProcess) { $OwnedProcesses += $OllamaProcess }
    }

    $env:OPENJEV_URL = 'http://127.0.0.1:8090'
    $env:OPENJEV_OLLAMA_SERVICE_URL = 'http://127.0.0.1:8091'
    $env:OPENJEV_GEMMA_URL = 'http://127.0.0.1:8092'
    if (-not $env:INFERENCE_BACKEND) {
        $env:INFERENCE_BACKEND = switch ($OpenJevEngine) {
            'all' { 'hybrid' }
            'both' { 'hybrid' }
            'ollama' { 'openjev_ollama' }
            'gemma.cpp' { 'openjev_gemma' }
            default { 'openjev' }
        }
    }
    & $Python -m uvicorn app.main:app --host 127.0.0.1 --port $ApiPort
} finally {
    foreach ($Process in $OwnedProcesses) {
        if ($Process -and -not $Process.HasExited) { Stop-Process -Id $Process.Id }
    }
}
