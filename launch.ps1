<#
.SYNOPSIS
  One-click launcher for the ReactiveAgents stack (native Gemma-4 on Ollama
  :11434 + the FastAPI/SSE app on :8000). Idempotent and re-runnable.

.DESCRIPTION
  Brings up, in ONE action, the whole finished s10-themed stack:

    1. Native sidecar Ollama on :11434, started WINDOWLESS with the s8-optimal
       server env-vars, ONLY if :11434 is not already serving.
    2. The runtime model: ensures the base `gemma4:e2b-it-qat` is pulled, builds
       the custom `gemma4-e2b-agent` tag (the tag the app actually drives) from
       the committed Modelfile if missing, then warms it so the first real
       request is fast.
    3. `uv sync`, then the app: uvicorn chat_app.app:app on :8000 with
       REACTIVE_AGENTS_LIVE=1, started ONLY if :8000 is not already serving.
    4. Waits for GET /health == 200, then opens the default browser.

  HOST SCOPE (d8): manages ONLY this recipe's OWN native :11434 Ollama and the
  app on :8000. It NEVER touches the et-tu-brute Docker Ollama on :11435 or any
  foreign PID. Only PIDs THIS launcher starts are tracked (var\launcher.pids.json)
  so stop.ps1 can stop exactly those and nothing else.

  Prints clear per-step status and an ACTIONABLE error if a prerequisite is
  missing (uv / native ollama / the model). Safe to double-click repeatedly.
#>
[CmdletBinding()]
param(
    [string]$OllamaExe = "$env:USERPROFILE\ollama-native\ollama.exe",
    [string]$OllamaUrl = "http://127.0.0.1:11434",
    [string]$AppUrl    = "http://127.0.0.1:8000",
    [string]$BaseModel = "gemma4:e2b-it-qat",
    [string]$ModelTag  = "gemma4-e2b-agent",
    [string]$Modelfile = "",
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
# Resolve the script dir in the BODY (not a param default): under Windows
# PowerShell 5.1 invoked via `-File`, $PSScriptRoot is empty during param-default
# binding, which made the (Join-Path $PSScriptRoot ...) default throw. Body scope
# populates it reliably; fall back to the invocation path if ever empty.
$RepoRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $Modelfile) { $Modelfile = Join-Path $RepoRoot "Modelfile.gemma4-e2b-agent" }
$PidStore = Join-Path $RepoRoot "var\launcher.pids.json"
$AppLog   = Join-Path $RepoRoot "var\app-8000.log"

# ---------------------------------------------------------------------------
# console helpers
# ---------------------------------------------------------------------------
function Say  ([string]$m) { Write-Host "[launch] $m" -ForegroundColor Cyan }
function Ok   ([string]$m) { Write-Host "[launch] $m" -ForegroundColor Green }
function Warn ([string]$m) { Write-Host "[launch] $m" -ForegroundColor Yellow }
function Die  ([string]$m) {
    Write-Host ""
    Write-Host "[launch] ERROR: $m" -ForegroundColor Red
    Write-Host ""
    exit 1
}

# ---------------------------------------------------------------------------
# PID tracking - record ONLY processes we start, so stop.ps1 is self-scoped.
# Shape: { "ollama": <pid|null>, "app": <pid|null> }
# ---------------------------------------------------------------------------
function Read-Pids {
    if (Test-Path $PidStore) {
        try { return (Get-Content -Raw -LiteralPath $PidStore | ConvertFrom-Json) }
        catch { return $null }
    }
    return $null
}
function Save-Pid ([string]$key, $procId) {
    $store = Read-Pids
    if ($null -eq $store) { $store = [pscustomobject]@{ ollama = $null; app = $null } }
    $store | Add-Member -NotePropertyName $key -NotePropertyValue $procId -Force
    $dir = Split-Path -Parent $PidStore
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    $store | ConvertTo-Json | Set-Content -LiteralPath $PidStore -Encoding utf8
}

# ---------------------------------------------------------------------------
# liveness probes
# ---------------------------------------------------------------------------
function Test-Url ([string]$url) {
    try { $null = Invoke-RestMethod -Uri $url -TimeoutSec 3; return $true }
    catch { return $false }
}

Write-Host ""
Say "ReactiveAgents one-click launcher"
Say "repo: $RepoRoot"
Write-Host ""

# ===========================================================================
# STEP 1 - native Ollama on :11434 (windowless, s8 env), skip if already up
# ===========================================================================
Say "step 1/4: native Ollama on :11434"
if (Test-Url "$OllamaUrl/api/version") {
    Ok "  already serving at $OllamaUrl (left as-is, not restarted)"
} else {
    if (-not (Test-Path $OllamaExe)) {
        Die ("native ollama.exe not found at: $OllamaExe`n" +
             "  ReactiveAgents uses a standalone (no-installer) native Ollama for its OWN serve on :11434 (separate from any Docker Ollama on :11435).`n" +
             "  FIX: install the native build there, or pass -OllamaExe <path\to\ollama.exe>.")
    }
    # s8-optimal server env for THIS child process only (never the global config):
    # single-model VRAM discipline + flash attention + q8_0 KV cache for the 6 GB GPU.
    $env:OLLAMA_HOST              = "127.0.0.1:11434"
    $env:OLLAMA_FLASH_ATTENTION   = "1"
    $env:OLLAMA_KV_CACHE_TYPE     = "q8_0"
    $env:OLLAMA_MAX_LOADED_MODELS = "1"
    $env:OLLAMA_NUM_PARALLEL      = "1"

    Say "  starting windowless serve (FLASH_ATTENTION=1, KV_CACHE_TYPE=q8_0, MAX_LOADED=1, PARALLEL=1)"
    $p = Start-Process -FilePath $OllamaExe -ArgumentList "serve" `
                       -WindowStyle Hidden -PassThru
    Save-Pid "ollama" $p.Id
    for ($i = 0; $i -lt 40 -and -not (Test-Url "$OllamaUrl/api/version"); $i++) {
        Start-Sleep -Milliseconds 500
    }
    if (-not (Test-Url "$OllamaUrl/api/version")) {
        Die "native ollama serve did not come up on $OllamaUrl within ~20s (pid $($p.Id))."
    }
    Ok "  serve up (pid $($p.Id), windowless)"
}

# ===========================================================================
# STEP 2 - runtime model: base present -> custom tag built -> warmed
# ===========================================================================
Say "step 2/4: runtime model '$ModelTag' (base '$BaseModel')"
try {
    $tags = (Invoke-RestMethod -Uri "$OllamaUrl/api/tags" -TimeoutSec 10).models.name
} catch {
    Die "could not query $OllamaUrl/api/tags - is the native serve healthy? ($($_.Exception.Message))"
}

# 2a) base model present (needed to build the custom tag) - pull if missing
if (($tags -contains $BaseModel) -or ($tags -contains "${BaseModel}:latest")) {
    Ok "  base '$BaseModel' present"
} else {
    Say "  pulling base '$BaseModel' (first run only, several GB) ..."
    & $OllamaExe pull $BaseModel
    if ($LASTEXITCODE -ne 0) {
        Die ("failed to pull base model '$BaseModel' (exit $LASTEXITCODE).`n" +
             "  FIX: check the native Ollama is online and the tag name is correct, then re-run. Manual: ollama pull $BaseModel")
    }
    Ok "  base '$BaseModel' pulled"
}

# 2b) custom agent tag built from the committed Modelfile - idempotent
if (($tags -contains $ModelTag) -or ($tags -contains "${ModelTag}:latest")) {
    Ok "  custom tag '$ModelTag' already built"
} else {
    if (-not (Test-Path $Modelfile)) { Die "Modelfile not found at $Modelfile" }
    Say "  building '$ModelTag' from $(Split-Path -Leaf $Modelfile) ..."
    & $OllamaExe create $ModelTag -f $Modelfile
    if ($LASTEXITCODE -ne 0) { Die "ollama create '$ModelTag' failed (exit $LASTEXITCODE)." }
    Ok "  '$ModelTag' created"
}

# 2c) warm the SHIP-PATH tag (gemma4-e2b-agent) so the first real turn is fast.
Say "  warming '$ModelTag' (tiny generate, keep_alive 30m) ..."
try {
    $body = @{
        model      = $ModelTag
        prompt     = "ok"
        stream     = $false
        keep_alive = "30m"
        think      = $false
        options    = @{ temperature = 0; num_predict = 1 }
    } | ConvertTo-Json -Depth 5
    $null = Invoke-RestMethod -Uri "$OllamaUrl/api/generate" -Method Post `
                              -Body $body -ContentType "application/json" -TimeoutSec 180
    Ok "  '$ModelTag' warm and resident"
} catch {
    # Non-fatal: the app warms it again at startup. Surface but continue.
    Warn "  warm-up call did not complete ($($_.Exception.Message)); the app will warm it on boot"
}

# ===========================================================================
# STEP 3 - uv sync + start the app on :8000 (live), skip if already serving
# ===========================================================================
Say "step 3/4: app on :8000 (uvicorn chat_app.app:app, REACTIVE_AGENTS_LIVE=1)"
if (Test-Url "$AppUrl/health") {
    Ok "  already serving at $AppUrl (left as-is, not restarted)"
} else {
    # uv present?
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uv) {
        Die ("'uv' is not installed / not on PATH. ReactiveAgents uses uv for its Python env.`n" +
             "  FIX: install uv (https://docs.astral.sh/uv/) then re-run this launcher.")
    }

    Say "  uv sync (ensuring the workspace .venv is current) ..."
    Push-Location $RepoRoot
    try {
        & uv sync
        if ($LASTEXITCODE -ne 0) { Die "uv sync failed (exit $LASTEXITCODE). See output above." }
    } finally { Pop-Location }
    Ok "  uv sync complete"

    $venvPy = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $venvPy)) {
        Die "workspace venv python not found at $venvPy after uv sync."
    }

    # Bridge the repo-root .env into the app process environment (F1, s9/b4).
    # The SMTP send_mail adapter resolves creds from the .env FILE directly
    # (reactive_tools.config.load_smtp_config reads <repo>/.env), so a populated
    # .env already lets send_mail send out-of-box; bridging the keys here ALSO
    # backstops that loader's os.environ fallback and any other os.environ reader
    # (e.g. tracing), so the unattended-email scenario is robust regardless of the
    # child process working directory. Secret VALUES are never printed.
    $envFile = Join-Path $RepoRoot ".env"
    if (Test-Path $envFile) {
        $bridged = 0
        foreach ($line in (Get-Content -LiteralPath $envFile -Encoding UTF8)) {
            # utf-8-sig tolerance: drop a leading BOM, then trim whitespace.
            $t = $line.TrimStart([char]0xFEFF).Trim()
            if (-not $t -or $t.StartsWith("#") -or ($t -notmatch "=")) { continue }
            if ($t.StartsWith("export ")) { $t = $t.Substring(7).TrimStart() }
            $idx = $t.IndexOf("=")
            $k = $t.Substring(0, $idx).Trim()
            $v = $t.Substring($idx + 1).Trim()
            if ($v.Length -ge 2 -and `
                (($v[0] -eq '"' -and $v[-1] -eq '"') -or ($v[0] -eq "'" -and $v[-1] -eq "'"))) {
                $v = $v.Substring(1, $v.Length - 2)
            }
            if ($k) { Set-Item -Path "Env:$k" -Value $v; $bridged++ }
        }
        Ok "  bridged $bridged .env key(s) into the app environment (values not shown)"
    } else {
        Warn "  no .env at $envFile — send_mail will error at call time until SMTP_* creds are set"
    }

    # Start uvicorn via the venv python DIRECTLY (single tracked PID, clean stop -
    # never `uv run` whose child python would orphan past the uv parent).
    # REACTIVE_AGENTS_LIVE=1 selects the live OllamaTransport (defaults already
    # point at native :11434 + gemma4-e2b-agent). Windowless; logs to var\app-8000.log.
    $env:REACTIVE_AGENTS_LIVE = "1"
    Say "  starting uvicorn (windowless; logs -> $(Split-Path -Leaf $AppLog)) ..."
    $appArgs = @("-m", "uvicorn", "chat_app.app:app", "--host", "127.0.0.1", "--port", "8000")
    $p = Start-Process -FilePath $venvPy -ArgumentList $appArgs `
                       -WorkingDirectory $RepoRoot -WindowStyle Hidden -PassThru `
                       -RedirectStandardOutput $AppLog -RedirectStandardError "$AppLog.err"
    Save-Pid "app" $p.Id
    Ok "  app process started (pid $($p.Id))"
}

# ===========================================================================
# STEP 4 - wait for /health, then open the browser
# ===========================================================================
Say "step 4/4: waiting for $AppUrl/health ..."
$healthy = $false
for ($i = 0; $i -lt 120; $i++) {   # up to ~60s (app does a uv-synced cold import)
    if (Test-Url "$AppUrl/health") { $healthy = $true; break }
    Start-Sleep -Milliseconds 500
}
if (-not $healthy) {
    Warn "health did not return 200 within ~60s."
    Warn "check the app log: $AppLog (and $AppLog.err)"
    Die "app failed to become healthy on $AppUrl."
}
Ok "  /health is 200 - stack is UP"

if (-not $NoBrowser) {
    Say "opening browser at $AppUrl"
    Start-Process $AppUrl | Out-Null
}

Write-Host ""
Ok "ReactiveAgents is ready:  $AppUrl"
$store = Read-Pids
$started = @()
if ($store -and $store.ollama) { $started += "ollama=$($store.ollama)" }
if ($store -and $store.app)    { $started += "app=$($store.app)" }
if ($started.Count -gt 0) {
    Say "this launcher started: $($started -join ', ')  ->  run stop.bat to stop ONLY these"
} else {
    Say "nothing new started (everything was already running); stop.bat will leave them alone"
}
Write-Host ""
