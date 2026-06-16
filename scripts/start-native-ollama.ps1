<#
.SYNOPSIS
  Start the NATIVE Ollama serve that backs ReactiveAgents' runtime model
  (Gemma-4 E2B) and ensure the optimized custom model tag exists.

.DESCRIPTION
  s8/b1 model swap (d17). The app's OllamaTransport now defaults to the native
  Ollama at http://127.0.0.1:11434 running `gemma4-e2b-agent` — NOT the foreign
  Docker Ollama on :11435 (project et-tu-brute), which this script NEVER touches.

  The native Ollama is a standalone (no-installer) build under
  C:\Users\aksou\ollama-native\ (a1). This script:
    1. Sets the server env the model needs (flash-attention, single-model VRAM
       discipline) so the 6 GB RTX 4050 stays within budget.
    2. Starts `ollama serve` DETACHED+HIDDEN on :11434 if it is not already up.
    3. Builds the custom `gemma4-e2b-agent` tag from the committed Modelfile if
       it is not already present (idempotent — `ollama create` is a no-op when the
       layers already exist).

  Idempotent and safe to re-run. Leaves the Docker :11435 stack untouched.

.NOTES
  The model's runtime knobs num_ctx/temperature/top_p/top_k/num_predict are BAKED
  into the `gemma4-e2b-agent` tag (see ../Modelfile.gemma4-e2b-agent). The decisive
  `think=false` control and `keep_alive` are passed by the transport per call, not
  here. To run the app against this serve: set REACTIVE_AGENTS_LIVE=1.
#>
[CmdletBinding()]
param(
    [string]$OllamaExe = "$env:USERPROFILE\ollama-native\ollama.exe",
    [string]$BaseUrl   = "http://127.0.0.1:11434",
    [string]$ModelTag  = "gemma4-e2b-agent",
    [string]$Modelfile = (Join-Path $PSScriptRoot "..\Modelfile.gemma4-e2b-agent")
)

$ErrorActionPreference = "Stop"

# --- server env (read by `ollama serve`) -----------------------------------
# Single-model VRAM discipline on the shared 6 GB GPU + flash attention (a2/a3).
$env:OLLAMA_HOST              = "127.0.0.1:11434"
$env:OLLAMA_FLASH_ATTENTION   = "1"
$env:OLLAMA_NUM_PARALLEL      = "1"
$env:OLLAMA_MAX_LOADED_MODELS = "1"

function Test-Serve {
    try { $null = Invoke-RestMethod -Uri "$BaseUrl/api/version" -TimeoutSec 3; return $true }
    catch { return $false }
}

# --- 1) ensure the native serve is up ---------------------------------------
if (Test-Serve) {
    Write-Host "[native-ollama] already serving at $BaseUrl"
} else {
    if (-not (Test-Path $OllamaExe)) {
        throw "native ollama.exe not found at $OllamaExe (see s8/a1 install note)"
    }
    Write-Host "[native-ollama] starting detached serve at $BaseUrl ..."
    Start-Process -FilePath $OllamaExe -ArgumentList "serve" -WindowStyle Hidden
    for ($i = 0; $i -lt 30 -and -not (Test-Serve); $i++) { Start-Sleep -Milliseconds 500 }
    if (-not (Test-Serve)) { throw "native ollama serve did not come up on $BaseUrl" }
    Write-Host "[native-ollama] serve is up"
}

# --- 2) ensure the optimized custom tag exists ------------------------------
$tags = (Invoke-RestMethod -Uri "$BaseUrl/api/tags").models.name
if ($tags -contains "${ModelTag}:latest" -or $tags -contains $ModelTag) {
    Write-Host "[native-ollama] model '$ModelTag' already built"
} else {
    if (-not (Test-Path $Modelfile)) { throw "Modelfile not found at $Modelfile" }
    Write-Host "[native-ollama] building '$ModelTag' from $Modelfile ..."
    & $OllamaExe create $ModelTag -f $Modelfile
    Write-Host "[native-ollama] model '$ModelTag' created"
}

Write-Host "[native-ollama] ready. Run the app with: `$env:REACTIVE_AGENTS_LIVE='1'"
