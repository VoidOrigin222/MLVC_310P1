@echo off
setlocal
cd /d "%~dp0\..\.."
python tools\ui\web_ui.py
endlocal
