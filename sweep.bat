@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem Use the same RECORD as your working execute.bat.
rem No argument: run A, select the validation winner, then run B and summarize.
rem Repeating the same command continues only an exactly matching sweep.
rem Usage: sweep.bat [capacity^|samples^|audit^|diagnose^|replot]
rem diagnose evaluates EXISTING_RUN\best.pt without training.
rem replot redraws all completed OUTPUT trials with every fixed training point.
rem capacity compares spline bins at saved optimization milestones.
rem samples rechecks pool size for the completed capacity winner.
rem audit checks the actual saved training pool without optimization.
set "RECORD=..\rainbow\datasets\drop_a1_i20_700nm_q1800x3600_c90x1800"
set "CONFIG=configs\rainbow_single.json"
set "SWEEP_CONFIG=configs\rainbow_sweep.json"
set "OUTPUT=runs\rainbow_sweep_gpu"
set "EXISTING_RUN=runs\rainbow_single_gpu"
set "DEVICE=cuda:0"
set "CAPACITY_CONFIG=configs\rainbow_capacity.json"
set "CAPACITY_SWEEP_CONFIG=configs\rainbow_capacity_sweep.json"
set "CAPACITY_OUTPUT=runs\rainbow_capacity_gpu"
set "SAMPLES_OUTPUT=runs\rainbow_capacity_samples_gpu"
set "AUDIT_RUN=%OUTPUT%\stage_b\n_262144"
set "AUDIT_OUTPUT=runs\rainbow_sampling_audit_n262144"

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
if /i "%MODE%"=="replot" goto :check
if /i "%MODE%"=="capacity" goto :check
if /i "%MODE%"=="samples" goto :check
if /i "%MODE%"=="audit" goto :check
goto :usage

:check
echo Checking the shared Python and CUDA environment...
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% "scripts\check_environment.py" check --device "%DEVICE%"
if not "%ERRORLEVEL%"=="0" goto :failed
if not exist "%RECORD%\metadata.json" goto :missing_record
if /i "%MODE%"=="replot" goto :replot
if /i "%MODE%"=="capacity" goto :capacity
if /i "%MODE%"=="samples" goto :samples
if /i "%MODE%"=="audit" goto :audit
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

:capacity
if not exist "%CAPACITY_CONFIG%" goto :missing_capacity_config
if not exist "%CAPACITY_SWEEP_CONFIG%" goto :missing_capacity_config
echo Comparing spline bins at fixed optimization milestones.
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow sweep-capacity-rainbow ^
    --record "%RECORD%" ^
    --config "%CAPACITY_CONFIG%" ^
    --sweep-config "%CAPACITY_SWEEP_CONFIG%" ^
    --output "%CAPACITY_OUTPUT%" ^
    --device "%DEVICE%"
if not "%ERRORLEVEL%"=="0" goto :failed
echo.
echo [DONE] Capacity summary: "%CAPACITY_OUTPUT%\summary.md"
echo Table: "%CAPACITY_OUTPUT%\summary.csv"
echo Plot: "%CAPACITY_OUTPUT%\summary.png"
echo Monitor: call monitor.bat "%CAPACITY_OUTPUT%"
set "EXIT_CODE=0"
goto :finish

:samples
if not exist "%CAPACITY_OUTPUT%\selection.json" goto :missing_capacity_run
echo Comparing pool sizes using the validation-selected spline bin count.
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow sweep-samples-rainbow ^
    --record "%RECORD%" ^
    --capacity "%CAPACITY_OUTPUT%" ^
    --output "%SAMPLES_OUTPUT%" ^
    --device "%DEVICE%"
if not "%ERRORLEVEL%"=="0" goto :failed
echo.
echo [DONE] Point-count summary: "%SAMPLES_OUTPUT%\summary.md"
echo Table: "%SAMPLES_OUTPUT%\summary.csv"
echo Monitor: call monitor.bat "%SAMPLES_OUTPUT%"
set "EXIT_CODE=0"
goto :finish

:audit
if not exist "%AUDIT_RUN%\best.pt" goto :missing_audit_run
if not exist "%AUDIT_RUN%\checkpoint.pt" goto :missing_audit_run
if not exist "%AUDIT_RUN%\config.json" goto :missing_audit_run
if not exist "%AUDIT_RUN%\sample_split.json" goto :missing_audit_run
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow audit-sampling-rainbow ^
    --record "%RECORD%" ^
    --run "%AUDIT_RUN%" ^
    --output "%AUDIT_OUTPUT%" ^
    --device "%DEVICE%"
if not "%ERRORLEVEL%"=="0" goto :failed
echo.
echo [DONE] Sampling audit: "%AUDIT_OUTPUT%\sampling_audit.json"
echo Histogram: "%AUDIT_OUTPUT%\sampling_histogram.png"
set "EXIT_CODE=0"
goto :finish

:replot
if not exist "%OUTPUT%\manifest.json" goto :missing_sweep_run
if not exist "%OUTPUT%\summary.json" goto :missing_sweep_run
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow replot-sweep-rainbow ^
    --record "%RECORD%" ^
    --sweep "%OUTPUT%" ^
    --device "%DEVICE%"
if not "%ERRORLEVEL%"=="0" goto :failed
echo.
echo [DONE] Training-point comparisons: "%OUTPUT%\training_point_plots"
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

:missing_sweep_run
echo [ERROR] OUTPUT must contain manifest.json and summary.json from a completed sweep.
set "EXIT_CODE=1"
goto :finish

:missing_existing_run
echo [ERROR] EXISTING_RUN must contain config.json and best.pt from a completed run.
set "EXIT_CODE=1"
goto :finish

:missing_capacity_config
echo [ERROR] Missing "%CAPACITY_CONFIG%" or "%CAPACITY_SWEEP_CONFIG%".
set "EXIT_CODE=1"
goto :finish

:missing_capacity_run
echo [ERROR] CAPACITY_OUTPUT must be a completed bin/step sweep with selection.json.
set "EXIT_CODE=1"
goto :finish

:missing_audit_run
echo [ERROR] AUDIT_RUN must be the original training run, not its replot directory.
echo [ERROR] Required: best.pt, checkpoint.pt, config.json, sample_split.json.
set "EXIT_CODE=1"
goto :finish

:usage
echo Usage: sweep.bat [capacity^|samples^|audit^|diagnose^|replot]
echo No argument runs or continues the original A-to-B sweep.
echo capacity compares bins/steps. samples rechecks N after capacity.
echo audit compares the saved training pool to exact CDF probabilities.
echo diagnose adds profiles to EXISTING_RUN. replot redraws completed OUTPUT trials.
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
