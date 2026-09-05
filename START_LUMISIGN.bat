@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating LumiSign environment...
  py -3.12 -m venv .venv
  .venv\Scripts\python.exe -m pip install -r requirements.txt
)
echo Starting LumiSign LASEE...
.venv\Scripts\python.exe -m streamlit run app.py
pause
