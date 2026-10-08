@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem Use the same RECORD as your working execute.bat.
rem No argument: run A, select the validation winner, then run B and summarize.
rem Repeating the same command continues only an exactly matching sweep.
rem Usage: sweep.bat [diagnose]
rem diagnose evaluates EXISTING_RUN\best.pt without training.
set "RECORD=..\rainbow\datasets\drop_a1_i20_700nm_q1800x3600_c90x1800"
set "CONFIG=configs\rainbow_single.json"
set "SWEEP_CONFIG=configs\rainbow_sweep.json"
set "OUTPUT=runs\rainbow_sweep_gpu"
set "EXISTING_RUN=runs\rainbow_single_gpu"
set "DEVICE=cuda:0"

pushd "%~dp0"
if not "%ERRORLEVEL%"=="0" exit /b 1
if not exist "environment.bat" goto :missing_launcher
if not exist "scripts\check_environment.py" goto :missing_launcher
call environment.bat
if not "%ERRORLEVEL%"=="0" goto :failed
if not "%~2"=="" goto :usage
set "MODE=%~1"
if "%MODE%"=="" set "MODE=run"
if /i "%MODE%"=="run" goto :check
if /i "%MODE%"=="diagnose" goto :check
goto :usage

:check
echo Checking the shared Python and CUDA environment...
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% "scripts\check_environment.py" check --device "%DEVICE%"
if not "%ERRORLEVEL%"=="0" goto :failed
if not exist "%RECORD%\metadata.json" goto :missing_record
if not exist "%SWEEP_CONFIG%" goto :missing_sweep_config
if /i "%MODE%"=="diagnose" goto :diagnose
if not exist "%CONFIG%" goto :missing_config

echo A: compare learning rates. B: compare point counts using the validation-selected rate.
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow sweep-rainbow ^
    --record "%RECORD%" ^
    --config "%CONFIG%" ^
    --sweep-config "%SWEEP_CONFIG%" ^
    --output "%OUTPUT%" ^
    --device "%DEVICE%"
if not "%ERRORLEVEL%"=="0" goto :failed
echo.
echo [DONE] Sweep summary: "%OUTPUT%\summary.md"
echo Table: "%OUTPUT%\summary.csv"
echo Plot: "%OUTPUT%\summary.png"
echo Monitor: call monitor.bat "%OUTPUT%"
set "EXIT_CODE=0"
goto :finish

:diagnose
if not exist "%EXISTING_RUN%\best.pt" goto :missing_existing_run
if not exist "%EXISTING_RUN%\config.json" goto :missing_existing_run
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow diagnose-rainbow ^
    --record "%RECORD%" ^
    --run "%EXISTING_RUN%" ^
    --sweep-config "%SWEEP_CONFIG%" ^
    --device "%DEVICE%"
if not "%ERRORLEVEL%"=="0" goto :failed
echo.
echo [DONE] Angular diagnostics: "%EXISTING_RUN%\diagnostics\angular_diagnostics.json"
echo Profiles: "%EXISTING_RUN%\diagnostics\angular_profiles.png"
set "EXIT_CODE=0"
goto :finish

:missing_launcher
echo [ERROR] Keep sweep.bat, environment.bat, and scripts\check_environment.py in this checkout.
set "EXIT_CODE=1"
goto :finish

:missing_record
echo [ERROR] Missing "%RECORD%\metadata.json". Copy the working RECORD value from execute.bat.
set "EXIT_CODE=1"
goto :finish

:missing_sweep_config
echo [ERROR] Missing sweep configuration: "%SWEEP_CONFIG%"
set "EXIT_CODE=1"
goto :finish

:missing_config
echo [ERROR] Missing base training configuration: "%CONFIG%"
set "EXIT_CODE=1"
goto :finish

:missing_existing_run
echo [ERROR] EXISTING_RUN must contain config.json and best.pt from a completed run.
set "EXIT_CODE=1"
goto :finish

:usage
echo Usage: sweep.bat [diagnose]
echo No argument runs or continues the A-to-B sweep. diagnose adds profiles to EXISTING_RUN.
set "EXIT_CODE=1"
goto :finish

:failed
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo [ERROR] Command failed with exit code %EXIT_CODE%. See the failing stage above.

:finish
popd
echo.
if not "%PHASEFLOW_PAUSE%"=="0" pause
exit /b %EXIT_CODE%
