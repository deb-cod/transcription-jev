param(
    [switch]$SkipOpenJev,
    [int]$ApiPort = 8000
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$OpenJevRoot = Join-Path $ProjectRoot 'vendor\openjev'
$Python = Join-Path $ProjectRoot 'venv\Scripts\python.exe'
$Llama = Join-Path $OpenJevRoot 'third_party\llama.cpp\b11056\llama-server.exe'
$Model = Join-Path $OpenJevRoot 'models\MiniCPM5-2B-Q4_K_M.gguf'
$Decision = Join-Path $OpenJevRoot 'bin\decision-test.exe'
$RuntimeLogs = Join-Path $ProjectRoot 'tmp\runtime'

$RequiredFiles = @($Python)
if (-not $SkipOpenJev) { $RequiredFiles += @($Llama, $Model, $Decision) }
foreach ($Required in $RequiredFiles) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Required file not found: $Required. Follow README setup first."
    }
}
New-Item -ItemType Directory -Force -Path $RuntimeLogs | Out-Null
$OwnedProcesses = @()

function Wait-ForHealth([string]$Url, [int]$Seconds = 120) {
    $Deadline = (Get-Date).AddSeconds($Seconds)
    do {
        try {
            $Response = Invoke-RestMethod -Uri $Url -TimeoutSec 3
            if ($Response.status -eq 'ok' -or $Response.ready -eq $true) { return }
        } catch { }
        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $Deadline)
    throw "Timed out waiting for $Url"
}

try {
    if (-not $SkipOpenJev) {
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
        try { $Ready = Invoke-RestMethod -Uri 'http://127.0.0.1:8090/health' -TimeoutSec 2 }
        catch { $Ready = $null }
        if (-not $Ready.ready) {
            $DecisionProcess = Start-Process -FilePath $Decision -WorkingDirectory $OpenJevRoot `
              -WindowStyle Hidden -RedirectStandardOutput (Join-Path $RuntimeLogs 'openjev.stdout.log') `
              -RedirectStandardError (Join-Path $RuntimeLogs 'openjev.stderr.log') -PassThru
            $OwnedProcesses += $DecisionProcess
            Wait-ForHealth 'http://127.0.0.1:8090/health'
        }
    }
    & $Python -m uvicorn app.main:app --host 127.0.0.1 --port $ApiPort
} finally {
    foreach ($Process in $OwnedProcesses) {
        if ($Process -and -not $Process.HasExited) { Stop-Process -Id $Process.Id }
    }
}
