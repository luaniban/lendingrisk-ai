# Executar na raiz: .\iniciar.ps1
$ErrorActionPreference = 'Stop'
$pythonLocal = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonLocal)) {
    throw 'Ambiente ausente. Crie a .venv seguindo o README.'
}
$previousReports = $env:LENDINGRISK_REPORT_DIR
Push-Location $PSScriptRoot
try {
    $localReports = Join-Path $PSScriptRoot '.venv\relatorios-windows'
    if (Test-Path -LiteralPath (Join-Path $localReports 'metricas.json')) {
        $env:LENDINGRISK_REPORT_DIR = $localReports
    }
    & $pythonLocal -m streamlit run app.py --server.address 127.0.0.1 --server.runOnSave true
} finally {
    $env:LENDINGRISK_REPORT_DIR = $previousReports
    Pop-Location
}
