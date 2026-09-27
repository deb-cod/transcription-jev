[CmdletBinding()]
param(
    [switch]$InstallPrerequisites,
    [string]$ModelDirectory = ''
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$OpenJevRoot = Join-Path $ProjectRoot 'vendor\openjev'
$ExpectedOpenJevCommit = '65ae076b501b464f0180e43f574ab451bc918e20'
$ModelFileName = 'gemma-4-E4B-it-Q4_0.gguf'
$ExpectedModelSha256 = 'a555b900214b477d8880e7832e0b8925e139b0159640036b09fe472b6f2097f2'

if (-not $ModelDirectory) {
    $ModelDirectory = Join-Path $ProjectRoot 'models\gemma4-e4b'
}
$ModelDirectory = [System.IO.Path]::GetFullPath($ModelDirectory)
$ModelPath = Join-Path $ModelDirectory $ModelFileName

function Refresh-ProcessPath {
    $MachinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $UserPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$MachinePath;$UserPath"
}

function Assert-Command([string]$Name, [string]$InstallHint) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found. $InstallHint"
    }
}

Write-Host 'Native Gemma-only setup'
Write-Host "Project: $ProjectRoot"
Write-Host "Model:   $ModelPath"

if ($InstallPrerequisites) {
    Assert-Command 'winget' 'Install App Installer from the Microsoft Store.'
    $Packages = @(
        'Git.Git',
        'Python.Python.3.12',
        'GoLang.Go',
        'GitHub.cli'
    )
    foreach ($Package in $Packages) {
        Write-Host "Installing/verifying $Package..."
        & winget install --id $Package --exact --source winget `
            --accept-source-agreements --accept-package-agreements --silent
        if ($LASTEXITCODE -ne 0) {
            throw "winget could not install $Package (exit code $LASTEXITCODE)."
        }
    }
    Refresh-ProcessPath
}

Assert-Command 'git' 'Install Git for Windows or rerun with -InstallPrerequisites.'
Assert-Command 'go' 'Install Go 1.24 or newer or rerun with -InstallPrerequisites.'
Assert-Command 'gh' 'Install GitHub CLI or rerun with -InstallPrerequisites.'
Assert-Command 'nvidia-smi' 'Install a compatible NVIDIA driver, then restart Windows.'

$GoVersionOutput = (& go version).Trim()
if ($GoVersionOutput -notmatch 'go(?<major>\d+)\.(?<minor>\d+)') {
    throw "Could not parse the installed Go version: $GoVersionOutput"
}
if ([int]$Matches.major -lt 1 -or ([int]$Matches.major -eq 1 -and [int]$Matches.minor -lt 24)) {
    throw "Go 1.24 or newer is required; found $GoVersionOutput."
}

$GitExecutable = (Get-Command git).Source
$GitRoot = Split-Path (Split-Path $GitExecutable -Parent) -Parent
$GitBash = Join-Path $GitRoot 'bin\bash.exe'
if (-not (Test-Path -LiteralPath $GitBash -PathType Leaf)) {
    throw "Git Bash was not found at $GitBash. Install Git for Windows with Git Bash."
}

$BasePython = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonOutput = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $PythonOutput) {
        $BasePython = ($PythonOutput | Select-Object -Last 1).Trim()
    }
}
if (-not $BasePython -and (Get-Command python -ErrorAction SilentlyContinue)) {
    $PythonOutput = & python -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $PythonOutput) {
        $BasePython = ($PythonOutput | Select-Object -Last 1).Trim()
    }
}
if (-not $BasePython -or -not (Test-Path -LiteralPath $BasePython -PathType Leaf)) {
    throw 'Python 3.10 or newer was not found. Install Python 3.12 or rerun with -InstallPrerequisites.'
}

$PythonVersion = & $BasePython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$PythonParts = $PythonVersion.Trim().Split('.')
if ([int]$PythonParts[0] -lt 3 -or ([int]$PythonParts[0] -eq 3 -and [int]$PythonParts[1] -lt 10)) {
    throw "Python 3.10 or newer is required; found $PythonVersion."
}

$VenvPython = Join-Path $ProjectRoot 'venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    Write-Host "Creating Python virtual environment with $BasePython..."
    & $BasePython -m venv (Join-Path $ProjectRoot 'venv')
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create the Python virtual environment.' }
}

Write-Host 'Installing API, UI, and one-file download dependencies...'
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Failed to upgrade pip.' }
& $VenvPython -m pip install -r (Join-Path $ProjectRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Failed to install application requirements.' }
& $VenvPython -m pip install --upgrade huggingface_hub
if ($LASTEXITCODE -ne 0) { throw 'Failed to install the Hugging Face CLI.' }

if (-not (Test-Path -LiteralPath (Join-Path $OpenJevRoot '.git') -PathType Container)) {
    if (Test-Path -LiteralPath $OpenJevRoot) {
        throw "$OpenJevRoot exists but is not a Git checkout. Move it aside and rerun setup."
    }
    Write-Host 'Cloning the pinned OpenJev source...'
    & git clone https://github.com/axsh/openjev.git $OpenJevRoot
    if ($LASTEXITCODE -ne 0) { throw 'Failed to clone OpenJev.' }
    & git -C $OpenJevRoot checkout $ExpectedOpenJevCommit
    if ($LASTEXITCODE -ne 0) { throw 'Failed to check out the pinned OpenJev commit.' }
} else {
    $CurrentCommit = (& git -C $OpenJevRoot rev-parse HEAD).Trim()
    if ($CurrentCommit -ne $ExpectedOpenJevCommit) {
        throw "OpenJev is at $CurrentCommit; expected $ExpectedOpenJevCommit. Use a clean pinned checkout."
    }
}

$TopLogprobsPatch = Join-Path $ProjectRoot 'patches\openjev-top-logprobs.patch'
& git -C $OpenJevRoot apply --reverse --check $TopLogprobsPatch 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host 'OpenJev top-logprobs patch is already applied.'
} else {
    & git -C $OpenJevRoot apply --check $TopLogprobsPatch
    if ($LASTEXITCODE -ne 0) {
        throw 'The OpenJev patch cannot be applied cleanly. Restore a clean pinned checkout and rerun setup.'
    }
    & git -C $OpenJevRoot apply $TopLogprobsPatch
    if ($LASTEXITCODE -ne 0) { throw 'Failed to apply the OpenJev top-logprobs patch.' }
}

Write-Host 'Installing the pinned llama.cpp CUDA runtime (no model is downloaded here)...'
Push-Location $OpenJevRoot
try {
    & $GitBash './scripts/setup/install_llama_cpp.sh'
    if ($LASTEXITCODE -ne 0) {
        throw 'llama.cpp installation failed. Run gh auth login if GitHub CLI requested authentication.'
    }
} finally {
    Pop-Location
}

$HfExecutable = Join-Path $ProjectRoot 'venv\Scripts\hf.exe'
if (-not (Test-Path -LiteralPath $HfExecutable -PathType Leaf)) {
    throw "Hugging Face CLI was not found at $HfExecutable."
}
New-Item -ItemType Directory -Force -Path $ModelDirectory | Out-Null
if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) {
    Write-Host 'Downloading exactly one model file: Gemma 4 E4B Instruct Q4_0...'
    & $HfExecutable download `
        'ggml-org/gemma-4-E4B-it-GGUF' `
        $ModelFileName `
        --local-dir $ModelDirectory
    if ($LASTEXITCODE -ne 0) { throw 'The Gemma GGUF download failed.' }
} else {
    Write-Host 'The Gemma GGUF already exists; the download is being reused.'
}

Write-Host 'Verifying the 4.59 GB model checksum (this can take a moment)...'
$ActualHash = (Get-FileHash -LiteralPath $ModelPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ActualHash -ne $ExpectedModelSha256) {
    throw "Gemma checksum mismatch. Expected $ExpectedModelSha256 but found $ActualHash."
}

Write-Host ''
Write-Host 'Setup complete. Only the native Gemma model was downloaded.' -ForegroundColor Green
Write-Host 'Start the backend:'
Write-Host '  .\run-gemma-only.ps1'
Write-Host 'Then, in a second PowerShell terminal, start the UI:'
Write-Host '  .\run-ui.ps1'
