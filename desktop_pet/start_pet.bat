@echo off
cd /d "%~dp0"
python pet.py
if errorlevel 1 pause
