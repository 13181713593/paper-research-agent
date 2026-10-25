@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Project Python environment is missing. Please install dependencies first.
  pause
  exit /b 1
)
echo Starting Paper Research Agent at http://localhost:8501
".venv\Scripts\python.exe" -m streamlit run "src\paper_agent\ui\app.py" --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
if errorlevel 1 pause
