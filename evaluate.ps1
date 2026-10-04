param(
    [ValidateSet('V1', 'V2', 'V3')][string]$Stage = 'V1',
    [switch]$List,
    [switch]$NoStream,
    [ValidateRange(0, 6)][int]$ApiRetries = 3,
    [ValidateRange(1, 20)][int]$Repeat = 1,
    [string[]]$Category,
    [string[]]$Case,
    [string]$Provider,
    [string]$Model,
    [string]$Out,
    [ValidateSet('all', 'dev', 'holdout')][string]$Split = 'all',
    [string]$Baseline,
    [string]$Variant = 'current',
    [switch]$SkipJudge,
    [string]$JudgeProvider,
    [string]$JudgeModel
)

$ErrorActionPreference = 'Stop'
$backendPath = Join-Path $PSScriptRoot 'backend'
$pythonPath = Join-Path $backendPath '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw '找不到 backend\.venv\Scripts\python.exe，请先按项目步骤准备 Python 环境。'
}
if (-not $PSBoundParameters.ContainsKey('Repeat') -and $Stage -in @('V2', 'V3')) { $Repeat = 3 }
if ($Stage -ne 'V3' -and ($SkipJudge -or $JudgeProvider -or $JudgeModel)) {
    throw 'SkipJudge、JudgeProvider、JudgeModel 仅适用于 -Stage V3。'
}
if ($Stage -eq 'V1' -and ($Baseline -or $Split -ne 'all' -or $Variant -ne 'current')) {
    throw 'Baseline、Split、Variant 仅适用于 -Stage V2 或 V3。'
}
$evalArgs = @('-m', 'tests.eval', $Stage.ToLowerInvariant(), '--repeat', $Repeat.ToString())
if ($Stage -in @('V2', 'V3')) {
    $evalArgs += @('--split', $Split, '--variant', $Variant)
    if ($Baseline) { $evalArgs += @('--baseline', [System.IO.Path]::GetFullPath($Baseline)) }
}
if ($Stage -eq 'V3') {
    if ($SkipJudge) { $evalArgs += '--skip-judge' }
    if ($JudgeProvider) { $evalArgs += @('--judge-provider', $JudgeProvider) }
    if ($JudgeModel) { $evalArgs += @('--judge-model', $JudgeModel) }
}
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
