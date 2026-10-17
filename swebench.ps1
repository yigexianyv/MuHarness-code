param(
    [string]$Experiment = 'output\swe-lite30',
    [string]$Config = 'backend\tests\eval\swebench\config.example.json',
    [string]$Python,
    [string]$MiniPython,
    [string]$JudgePython,
    [switch]$CheckOnly,
    [switch]$Help
)

$ErrorActionPreference = 'Stop'
if ($Help) {
    Write-Host @'
Usage (from the repository root):
  .\swebench.ps1                  Prepare, dev gold, Lite30 both groups, grade, report
  .\swebench.ps1 -CheckOnly       Dependencies and Docker only; no model calls
  .\swebench.ps1 -Config path.json -Experiment output\my-experiment

Missing evaluation environments are installed with uv. Existing environments are reused.
The model endpoint/name comes from Config; the key comes from environment or backend/.env.
Running again resumes recorded attempts. Formal comparisons should use Linux/WSL.
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
try {
    $env:PYTHONIOENCODING = 'utf-8'
    Push-Location -LiteralPath $backend
    try {
        $controller = Ensure-EvalPython $Python 'backend\.venv-swebench' @(
            'backend\.venv-swebench\Scripts\python.exe',
            'backend\.venv-eval\Scripts\python.exe',
            'backend\.venv\Scripts\python.exe'
        ) @('-r', (Join-Path $backend 'requirements.txt'), '-r',
            (Join-Path $backend 'tests\eval\swebench\requirements-data.txt')) `
            'import app.application, datasets, huggingface_hub'
        $mini = Ensure-EvalPython $MiniPython 'output\venv-mini' @(
            'output\venv-mini\Scripts\python.exe',
            'output\swebench-dev-venv\Scripts\python.exe'
        ) @('mini-swe-agent==2.4.6', 'litellm==1.105.1') `
            'from minisweagent.agents.default import DefaultAgent'
        $judge = Ensure-EvalPython $JudgePython 'output\venv-judge' @(
            'output\venv-judge\Scripts\python.exe',
            'output\swebench-judge-venv\Scripts\python.exe'
        ) @('swebench==5.0.2') 'import swebench.harness.run_evaluation'
        $arguments = @('-m', 'tests.eval.swebench', 'all',
            '--experiment', (Resolve-RepoPath $Experiment),
            '--config', (Resolve-RepoPath $Config),
            '--mini-python', $mini, '--judge-python', $judge)
        if ($CheckOnly) { $arguments += '--check-only' }
        Invoke-Checked $controller $arguments
    }
    finally { Pop-Location }
}
finally { $env:PYTHONIOENCODING = $previousEncoding }
