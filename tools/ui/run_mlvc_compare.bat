@echo off
setlocal
cd /d "%~dp0\..\.."
python tools\ui\mlvc_compare.py
endlocal
