# Execution workflow for MLflow Champion-Challenger Mini Project
# Student: M K Mohammed Kashif | USN: 1MS25CI072 | Course: Skill Lab - MLOps with MLflow
# Usage: powershell -ExecutionPolicy Bypass -File run_all.ps1

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$RepoName = "1MS25CI072-mlflow-champion-challenger"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Run($exe, [string[]]$argv) {
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { throw "Execution failed: $exe $($argv -join ' ')" }
}
function Commit($msg) {
    git add -A
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) {
        Run git @("commit", "-q", "-m", "$msg")
        Write-Host "   Committed: $msg"
    }
}

# 1. Git initialization and pre-run setup commit
Step "1. Git Repository Setup"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is required. Please install git and ensure it is in PATH."
}
if (-not (Test-Path .git)) { Run git @("init", "-q", "-b", "main") }
if (-not (git config user.name))  { git config user.name "M K Mohammed Kashif" }
if (-not (git config user.email)) { git config user.email "kashif.mohammed@msrit.edu" }

if (-not (git rev-parse --verify HEAD 2>$null)) {
    git add README.md champion_rule.json requirements.txt .gitignore data/new_data.csv
    Run git @("commit", "-q", "-m", "Project setup and predefined Champion rule, established before evaluation")
    Write-Host "   Committed: Initial setup and Champion rule"
    Commit "Add evaluation pipeline, MLflow pyfunc wrapper, and registry scripts"
}

# 2. Python environment
Step "2. Python Environment Configuration"
$py = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { "python" }
if (-not (Test-Path .venv\Scripts\python.exe)) {
    if ($py -eq "py") { Run py @("-3", "-m", "venv", "--system-site-packages", ".venv") } else { Run python @("-m", "venv", "--system-site-packages", ".venv") }
}
$PY = (Resolve-Path .venv\Scripts\python.exe).Path
Run $PY @("-m", "pip", "install", "-q", "-r", "requirements.txt")

# 3. Tracking Server Verification
Step "3. MLflow Tracking Server Check (http://127.0.0.1:5000)"
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
function ServerUp {
    try {
        (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5000 -TimeoutSec 2).StatusCode -eq 200
    } catch {
        $false
    }
}
if (-not (ServerUp)) {
    Start-Process -FilePath $PY -ArgumentList "-m", "mlflow", "server", "--host", "127.0.0.1", "--port", "5000" `
                  -WindowStyle Minimized
    $waited = 0
    while (-not (ServerUp) -and $waited -lt 30) {
        Start-Sleep -Seconds 1
        $waited++
    }
    if (-not (ServerUp)) { throw "Failed to start MLflow server on port 5000 within 30 seconds." }
}
Write-Host "   MLflow server is active on http://127.0.0.1:5000"

# 4. Pipeline Execution
Step "4. Pipeline Step 1 - Sampling SST-2 evaluation set"
Run $PY @("prepare_data.py")
Commit "Add shared evaluation set data/eval_inputs.csv (seed=42)"

Step "5. Pipeline Step 2 - Evaluating Model A vs Model B"
Run $PY @("evaluate.py")

Step "6. Pipeline Step 3 - Registering Champion and Challenger"
Run $PY @("register_champion.py")

Step "7. Pipeline Step 4 - Running inference on novel unseen data"
Run $PY @("predict_from_mlflow.py")
Run $PY @("predict_from_mlflow.py", "--alias", "challenger")

Step "8. Pipeline Step 5 - Capturing submission screenshots"
Run $PY @("capture_screenshots.py")

Step "9. Pipeline Step 6 - Updating README report with evaluation results"
Run $PY @("write_report.py")

Commit "Add model evaluation results, registry aliases, plots, and updated report"

Step "Done! All pipeline artifacts and MLflow runs are generated."
Write-Host "Inspect results in README.md or visit http://127.0.0.1:5000" -ForegroundColor Green
