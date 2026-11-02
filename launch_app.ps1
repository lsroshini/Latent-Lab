$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$mriPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$mriCpuPython = Join-Path $PSScriptRoot '.venv-cpu/Scripts/python.exe'
if ((Test-Path -LiteralPath $mriCpuPython) -and (Test-Path -LiteralPath (Join-Path $PSScriptRoot '.venv-cpu/Lib/site-packages/torch/__init__.py'))) {
    $mriPython = $mriCpuPython
}
if (-not (Test-Path -LiteralPath $mriPython)) {
    throw 'Project environment missing. Follow README.md to create .venv and install dependencies.'
}
& $mriPython -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
