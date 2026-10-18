@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch.ps1" %*
set "semgrainsExitCode=%ERRORLEVEL%"
if not "%semgrainsExitCode%"=="0" (
    echo.
    echo Semgrains could not start. See the error above and README.md.
    if /I not "%CI%"=="true" pause
)
exit /b %semgrainsExitCode%
