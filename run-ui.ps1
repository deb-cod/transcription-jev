param(
    [int]$Port = 8501
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$Python = Join-Path $ProjectRoot 'venv\Scripts\python.exe'
$Ui = Join-Path $ProjectRoot 'ui.py'

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project virtual environment not found at $Python. Follow the README setup first."
}

& $Python -c "import httpx, streamlit; print('Python:', __import__('sys').executable); print('httpx:', httpx.__version__); print('streamlit:', streamlit.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "The project venv is missing UI dependencies. Run: .\venv\Scripts\python.exe -m pip install -r requirements.txt"
}

& $Python -m streamlit run $Ui --server.port $Port
