@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run START_LUMISIGN.bat once before verification.
  pause
  exit /b 1
)
.venv\Scripts\python.exe verify_evidence.py
.venv\Scripts\python.exe -m pytest -q tests\test_core.py
pause
