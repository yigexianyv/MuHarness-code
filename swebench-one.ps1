[CmdletBinding()]
param(
    [string]$Instance,
    [ValidateSet('dev', 'test')][string]$Split = 'dev',
    [switch]$Lite30,
    [string]$Experiment,
    [string]$Config,
    [string]$Model,
    [ValidateSet('normal', 'mea')][string]$Mode = 'normal',
    [string]$Python,
    [string]$JudgePython,
    [switch]$SkipGold,
    [switch]$Help
)

$ErrorActionPreference = 'Stop'
if ($Help) {
    Write-Host @'
Run ONLY MuHarness on SWE-bench Lite (one instance, or the fixed Lite30 set),
then grade it with the official harness.

Usage (from the repository root):
  .\swebench-one.ps1                                 first dev instance
  .\swebench-one.ps1 -Instance <instance_id>
  .\swebench-one.ps1 -Split test -Instance django__django-13447
  .\swebench-one.ps1 -Split test -Instance <id> -Mode mea   long-task mode
  .\swebench-one.ps1 -Model gpt-5.6-sol               other model, same endpoint/key
  .\swebench-one.ps1 -Config my-model.json           instead of backend\.env
  .\swebench-one.ps1 -Lite30                         30 test instances; run again
                                                     to resume after an interruption
  .\swebench-one.ps1 -Lite30 -Experiment output\x    separate Lite30 folder

The model, endpoint and key come from backend\.env (MODEL_DEFAULT_PROVIDER and
its *_MODEL / *_BASE_URL / *_API_KEY). The endpoint must support Chat
Completions with tools; this is checked with one small request first.
Each run creates a new folder under output\swe-one\.
'@
    exit 0
}

function Resolve-RepoPath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $PSScriptRoot $Value))
}

# Windows PowerShell 5.1 turns ANY native stderr output (uv progress, Python
# logging, an expected import failure) into a terminating error while
# $ErrorActionPreference is 'Stop'. Run native commands under 'Continue' and
# judge them only by their exit code.
function Invoke-Native([scriptblock]$Block) {
    $saved = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $Block; return $LASTEXITCODE }
    finally { $ErrorActionPreference = $saved }
}

function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    $code = Invoke-Native { & $Executable @Arguments 2>&1 | ForEach-Object { "$_" } | Out-Host }
    if ($code -ne 0) { throw "Command failed (exit $code): $Executable" }
}

function Ensure-EvalPython([string]$Requested, [string]$Directory,
                           [string[]]$Candidates, [string[]]$Packages, [string]$Imports) {
    if ($Requested) {
        $resolved = Resolve-RepoPath $Requested
        if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
            throw "Python not found: $resolved"
        }
        return $resolved
    }
    foreach ($candidate in $Candidates) {
        $resolved = Resolve-RepoPath $candidate
        if (Test-Path -LiteralPath $resolved -PathType Leaf) {
            $code = Invoke-Native { & $resolved -c $Imports *> $null }
            if ($code -eq 0) { return $resolved }
        }
    }
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw 'uv is required to create missing evaluation environments. Install uv first.'
    }
    $environment = Resolve-RepoPath $Directory
    $executable = Join-Path $environment 'Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        Invoke-Checked 'uv' @('venv', '--python', '3.13', $environment)
    }
    Invoke-Checked 'uv' (@('pip', 'install', '--python', $executable) + $Packages)
    return $executable
}

$backend = Join-Path $PSScriptRoot 'backend'
$previousEncoding = $env:PYTHONIOENCODING
# Python writes UTF-8; make the console decode it as UTF-8 (Chinese progress lines).
$previousConsoleEncoding = [Console]::OutputEncoding
try {
    $env:PYTHONIOENCODING = 'utf-8'
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    Push-Location -LiteralPath $backend
    try {
        $controller = Ensure-EvalPython $Python 'backend\.venv-swebench' @(
            'backend\.venv-swebench\Scripts\python.exe',
            'backend\.venv-eval\Scripts\python.exe',
            'backend\.venv\Scripts\python.exe'
        ) @('-r', (Join-Path $backend 'requirements.txt'), '-r',
            (Join-Path $backend 'tests\eval\swebench\requirements-data.txt')) `
            'import app.application, datasets, huggingface_hub'
        $judge = Ensure-EvalPython $JudgePython 'output\venv-judge' @(
            'output\venv-judge\Scripts\python.exe',
            'output\swebench-judge-venv\Scripts\python.exe'
        ) @('swebench==5.0.2') 'import swebench.harness.run_evaluation'
        $arguments = @('-m', 'tests.eval.swebench.one', '--judge-python', $judge,
            '--split', $Split)
        if ($Instance) { $arguments += @('--instance', $Instance) }
        if ($Lite30) { $arguments += '--lite30' }
        if ($Experiment) { $arguments += @('--experiment', (Resolve-RepoPath $Experiment)) }
        if ($Config) { $arguments += @('--config', (Resolve-RepoPath $Config)) }
        if ($Model) { $arguments += @('--model', $Model) }
        $arguments += @('--mode', $Mode)
        if ($SkipGold) { $arguments += '--skip-gold' }
        Invoke-Checked $controller $arguments
    }
    finally { Pop-Location }
}
finally {
    $env:PYTHONIOENCODING = $previousEncoding
    [Console]::OutputEncoding = $previousConsoleEncoding
}
