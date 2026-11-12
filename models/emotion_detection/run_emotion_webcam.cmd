@echo off
"%~dp0.venv\Scripts\python.exe" "%~dp0webcam_emotion.py" %*
if errorlevel 1 pause
