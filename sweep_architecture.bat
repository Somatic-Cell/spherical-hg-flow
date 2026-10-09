@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem Use the same RECORD as your working execute.bat.
rem No argument runs or resumes the same planned architecture sweep.
rem check validates the complete plan without starting a training trial.
rem report rebuilds summaries from existing trials without optimizer updates.
rem This launcher does not install packages or replace the selected PyTorch.
set "RECORD=..\rainbow\datasets\drop_a1_i20_700nm_q1800x3600_c90x1800"
set "CONFIG=configs\rainbow_architecture.json"
set "OUTPUT=runs\rainbow_architecture_gpu"
set "DEVICE=cuda:0"

rem Default to unattended execution, including after a failure.
rem Set PHASEFLOW_PAUSE=1 before calling this launcher to opt into pause.
if not defined PHASEFLOW_PAUSE set "PHASEFLOW_PAUSE=0"
if not defined PYTHONUNBUFFERED set "PYTHONUNBUFFERED=1"

pushd "%~dp0"
if not "%ERRORLEVEL%"=="0" exit /b 1
if not exist "environment.bat" goto :missing_launcher
if not exist "scripts\check_environment.py" goto :missing_launcher
call environment.bat
if not "%ERRORLEVEL%"=="0" goto :failed
if not "%~2"=="" goto :usage
set "MODE=%~1"
if "%MODE%"=="" set "MODE=run"
if /i "%MODE%"=="run" goto :preflight
if /i "%MODE%"=="check" goto :preflight
if /i "%MODE%"=="report" goto :preflight
goto :usage

:preflight
echo Checking the shared Python and CUDA environment...
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% "scripts\check_environment.py" check --device "%DEVICE%"
if not "%ERRORLEVEL%"=="0" goto :failed
if not exist "%CONFIG%" goto :missing_config
if not exist "%RECORD%\metadata.json" goto :missing_record
if /i "%MODE%"=="check" goto :check
if /i "%MODE%"=="report" goto :report

echo Running the complete architecture plan. Repeating this command resumes matching work.
echo Only one process may use this OUTPUT at a time: "%OUTPUT%"
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow.architecture_sweep ^
    --record "%RECORD%" ^
    --config "%CONFIG%" ^
    --output "%OUTPUT%" ^
    --device "%DEVICE%"
if not "%ERRORLEVEL%"=="0" goto :failed
echo.
echo [DONE] Architecture sweep completed.
goto :summary

:check
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow.architecture_sweep ^
    --record "%RECORD%" ^
    --config "%CONFIG%" ^
    --output "%OUTPUT%" ^
    --device "%DEVICE%" ^
    --dry-run
if not "%ERRORLEVEL%"=="0" goto :failed
echo.
echo [DONE] Plan and input checks passed. No training trial was started.
set "EXIT_CODE=0"
goto :finish

:report
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow.architecture_sweep ^
    --record "%RECORD%" ^
    --config "%CONFIG%" ^
    --output "%OUTPUT%" ^
    --device "%DEVICE%" ^
    --report-only
if not "%ERRORLEVEL%"=="0" goto :failed
echo.
echo [DONE] Existing architecture results were summarized. No optimizer updates were run.

:summary
echo Summary: "%OUTPUT%\summary.md"
echo Table: "%OUTPUT%\summary.csv"
echo Plot: "%OUTPUT%\summary.png"
echo Monitor in another CMD window: call monitor.bat "%OUTPUT%"
set "EXIT_CODE=0"
goto :finish

:missing_launcher
echo [ERROR] Keep sweep_architecture.bat, environment.bat, and scripts\check_environment.py in this checkout.
set "EXIT_CODE=1"
goto :finish

:missing_config
echo [ERROR] Missing architecture configuration: "%CONFIG%"
set "EXIT_CODE=1"
goto :finish

:missing_record
echo [ERROR] Missing "%RECORD%\metadata.json". Copy the working RECORD from execute.bat.
set "EXIT_CODE=1"
goto :finish

:usage
echo Usage: sweep_architecture.bat [run^|check^|report]
echo No argument or run executes or resumes the same planned sweep.
echo check validates the plan without training. report summarizes existing trials only.
echo Edit RECORD, CONFIG, OUTPUT, and DEVICE at the top of this file.
set "EXIT_CODE=2"
goto :finish

:failed
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo [ERROR] Command failed with exit code %EXIT_CODE%. See the output above.
echo Review any existing "%OUTPUT%\summary.md" and the trial logs.
echo Successful trial results are retained. Resolve the reported issue before retrying.
echo Repeating the same command resumes only matching saved work.

:finish
popd
echo.
if "%PHASEFLOW_PAUSE%"=="1" pause
exit /b %EXIT_CODE%
