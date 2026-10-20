@echo off
setlocal
python -c "import sys" >nul 2>nul
if not errorlevel 1 goto run_python
py -3 -c "import sys" >nul 2>nul
if not errorlevel 1 goto run_py
echo Scissors requires Python 3.12 or newer on PATH. 1>&2
exit /b 1

:run_python
python "%~dp0scripts\launch.py" %*
exit /b %errorlevel%

:run_py
py -3 "%~dp0scripts\launch.py" %*
exit /b %errorlevel%
