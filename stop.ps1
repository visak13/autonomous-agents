<#
.SYNOPSIS
  Stop ONLY the processes launch.ps1 started (tracked in var\launcher.pids.json).

.DESCRIPTION
  Self-scoped teardown (d8). Reads the PID store written by launch.ps1 and stops
  exactly those PIDs - the native Ollama on :11434 and/or the app on :8000 -
  ONLY if this launcher started them. It NEVER does an image-wide kill
  (no `taskkill /IM ollama.exe|python.exe`), so:
    - a native Ollama / app that was ALREADY running when you launched (and thus
      NOT tracked) is left untouched;
    - the et-tu-brute Docker Ollama on :11435 and every foreign PID are untouched.

  For the app, this is a normal terminate of the single uvicorn python PID. If
  you prefer the FastAPI lifespan teardown to run, Ctrl-C the app in its own
  window instead - but launch.ps1 starts it windowless, so this scoped stop is
  the intended path.

  Each PID is identity-checked (still alive AND name matches what we started)
  before stopping, so a recycled PID belonging to some other process is skipped.
#>
[CmdletBinding()]
param(
    [string]$AppUrl = "http://127.0.0.1:8000"
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
$PidStore = Join-Path $RepoRoot "var\launcher.pids.json"

function Say  ([string]$m) { Write-Host "[stop] $m" -ForegroundColor Cyan }
function Ok   ([string]$m) { Write-Host "[stop] $m" -ForegroundColor Green }
function Warn ([string]$m) { Write-Host "[stop] $m" -ForegroundColor Yellow }

if (-not (Test-Path $PidStore)) {
    Warn "no PID store at $PidStore - this launcher has not started anything (or it was already stopped). Nothing to do."
    return
}

$store = Get-Content -Raw -LiteralPath $PidStore | ConvertFrom-Json

# key -> the process name we expect for that slot (identity guard against PID reuse)
$expect = @{ ollama = "ollama"; app = "python" }

$stoppedAny = $false
foreach ($key in @("app", "ollama")) {   # app first, then its model server
    $procId = $store.$key
    if (-not $procId) { Say "${key}: nothing tracked"; continue }

    $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
    if (-not $proc) {
        Say "${key}: pid $procId already gone"
    }
    elseif ($proc.ProcessName -notlike "$($expect[$key])*") {
        # PID was recycled by an unrelated process - DO NOT touch it.
        Warn "${key}: pid $procId is now '$($proc.ProcessName)', not '$($expect[$key])' - skipping (PID reused, not ours)"
    }
    else {
        Say "${key}: stopping pid $procId ($($proc.ProcessName)) ..."
        try {
            Stop-Process -Id $procId -Force -ErrorAction Stop
            Ok "${key}: stopped"
            $stoppedAny = $true
        } catch {
            Warn "${key}: could not stop pid $procId ($($_.Exception.Message))"
        }
    }
    # clear the slot regardless so a re-run does not chase a dead/reused PID
    $store.$key = $null
}

# persist the cleared store (so stop is idempotent)
$store | ConvertTo-Json | Set-Content -LiteralPath $PidStore -Encoding utf8

if ($stoppedAny) {
    Ok "done - stopped only what this launcher started."
} else {
    Say "done - nothing of ours was running."
}
