param(
    [switch]$List,
    [switch]$NoStream,
    [ValidateRange(0, 6)][int]$ApiRetries = 3,
    [ValidateRange(1, 20)][int]$Repeat = 1,
    [string[]]$Category,
    [string[]]$Case,
    [string]$Provider,
    [string]$Model,
    [string]$Out
)

$ErrorActionPreference = 'Stop'
$backendPath = Join-Path $PSScriptRoot 'backend'
$pythonPath = Join-Path $backendPath '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw '找不到 backend\.venv\Scripts\python.exe，请先按项目步骤准备 Python 环境。'
}
$evalArgs = @('-m', 'tests.eval', 'v1', '--repeat', $Repeat.ToString())
$evalArgs += @('--api-retries', $ApiRetries.ToString())
if ($List) { $evalArgs += '--list' }
if ($NoStream) { $evalArgs += '--no-stream' }
foreach ($item in $Category) { $evalArgs += @('--category', $item) }
foreach ($item in $Case) { $evalArgs += @('--case', $item) }
if ($Provider) { $evalArgs += @('--provider', $Provider) }
if ($Model) { $evalArgs += @('--model', $Model) }
if ($Out) { $evalArgs += @('--out', [System.IO.Path]::GetFullPath($Out)) }
$previousEncoding = $env:PYTHONIOENCODING
try {
    $env:PYTHONIOENCODING = 'utf-8'
    Push-Location -LiteralPath $backendPath
    try {
        & $pythonPath @evalArgs
        $resultCode = $LASTEXITCODE
    }
    finally { Pop-Location }
}
finally { $env:PYTHONIOENCODING = $previousEncoding }
exit $resultCode
