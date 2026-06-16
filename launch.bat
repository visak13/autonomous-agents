@echo off
REM ===========================================================================
REM  ReactiveAgents one-click launcher (double-clickable).
REM  Brings up the native Gemma-4 Ollama (:11434) + the FastAPI/SSE app (:8000)
REM  idempotently, waits for /health, and opens the browser.
REM  Scope: manages ONLY this recipe's native :11434 + the app on :8000.
REM         Never touches the Docker Ollama on :11435 or any foreign PID.
REM  Re-runnable: detects already-running services and never double-starts.
REM ===========================================================================
setlocal
set "HERE=%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%HERE%launch.ps1" %*
set "RC=%ERRORLEVEL%"

REM Keep the window open when double-clicked so the user can read status/errors.
echo.
if not "%RC%"=="0" (
  echo [launch] exited with error code %RC%.
) else (
  echo [launch] done. The app keeps running in the background.
  echo [launch] Use stop.bat to stop only what this launcher started.
)
echo.
pause
endlocal & exit /b %RC%
