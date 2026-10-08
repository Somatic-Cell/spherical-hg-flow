@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem Uses the same interpreter as execute.bat; no CUDA initialization is required.
rem Usage: monitor.bat [log_directory]
set "LOGDIR=runs"
if not "%~1"=="" set "LOGDIR=%~1"

pushd "%~dp0"
if not "%ERRORLEVEL%"=="0" exit /b 1
if not exist "environment.bat" goto :missing_launcher
if not exist "scripts\check_environment.py" goto :missing_launcher
call environment.bat
if not "%ERRORLEVEL%"=="0" goto :failed
if not "%~2"=="" goto :usage
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% "scripts\check_environment.py" monitor
if not "%ERRORLEVEL%"=="0" goto :failed

echo Open http://127.0.0.1:6006 in your browser. Press Ctrl+C to stop TensorBoard.
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m tensorboard.main ^
    --logdir "%LOGDIR%" ^
    --host 127.0.0.1 ^
    --port 6006
if not "%ERRORLEVEL%"=="0" goto :failed
set "EXIT_CODE=0"
goto :finish

:missing_launcher
echo [ERROR] Keep monitor.bat, environment.bat, and scripts\check_environment.py in this checkout.
set "EXIT_CODE=1"
goto :finish

:usage
echo Usage: monitor.bat [log_directory]
set "EXIT_CODE=1"
goto :finish

:failed
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo [ERROR] Monitor failed with exit code %EXIT_CODE%. See the output above.

:finish
popd
echo.
if not "%PHASEFLOW_PAUSE%"=="0" pause
exit /b %EXIT_CODE%
