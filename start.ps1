param([ValidateRange(1, 65535)][int]$Port = 7860)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv\Scripts\pythonw.exe'
$app = Join-Path $PSScriptRoot 'app.py'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Missing .venv. Follow the installation steps in README.md.'
}
$process = Start-Process -FilePath $python -ArgumentList @(('"' + $app + '"'), '--open', '--log', '--port', $Port) -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru
Write-Output "Handdraw PID $($process.Id): http://127.0.0.1:$Port"
