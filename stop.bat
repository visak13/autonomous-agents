@echo off
REM ===========================================================================
REM  Stop ONLY the processes launch.bat started (native :11434 Ollama and/or the
REM  app on :8000), tracked in var\launcher.pids.json. Self-scoped: never an
REM  image-wide kill, never the Docker Ollama on :11435, never a foreign PID.
REM  Idempotent: safe to run when nothing is running.
REM ===========================================================================
setlocal
set "HERE=%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%HERE%stop.ps1" %*
set "RC=%ERRORLEVEL%"

echo.
pause
endlocal & exit /b %RC%
