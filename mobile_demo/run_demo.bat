@echo off
echo.
echo  VVDP - Vietnamese Vishing Detection - Mobile Demo
echo  --------------------------------------------------
echo  Opening: http://localhost:5000
echo.
cd /d "%~dp0"
start "" "http://localhost:5000"
python app.py
pause
