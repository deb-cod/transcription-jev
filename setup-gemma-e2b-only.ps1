[CmdletBinding()]
param(
    [switch]$InstallPrerequisites,
    [string]$ModelDirectory = ''
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot

& (Join-Path $ProjectRoot 'setup-gemma-only.ps1') `
    -InstallPrerequisites:$InstallPrerequisites `
    -ModelDirectory $ModelDirectory `
    -ModelRepository 'ggml-org/gemma-4-E2B-it-GGUF' `
    -ModelFileName 'gemma-4-E2B-it-Q4_0.gguf' `
    -ExpectedModelSha256 '8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52' `
    -ModelDisplayName 'Gemma 4 E2B Instruct Q4_0' `
    -DefaultModelDirectoryName 'gemma4-e2b' `
    -RunScriptName 'run-gemma-e2b-only.ps1'

exit $LASTEXITCODE
