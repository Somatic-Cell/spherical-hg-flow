@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem One pure-log fit: every training point is sampled uniformly in solid angle.
rem Copy RECORD from your working execute.bat. Use a new OUTPUT for this experiment.
rem The JSON config controls the model, point count, updates, and automatic full maps.
set "RECORD=..\rainbow\datasets\drop_a1_i20_700nm_q1800x3600_c90x1800"
set "CONFIG=configs\rainbow_log_uniform.json"
set "DIAGNOSTIC_CONFIG=configs\rainbow_sweep.json"
set "OUTPUT=runs\rainbow_log_uniform_gpu"
set "DEVICE=cuda:0"
set "PROBE_STEPS=250"

pushd "%~dp0"
if not "%ERRORLEVEL%"=="0" exit /b 1
if not "%~2"=="" goto :usage
set "MODE=%~1"
if "%MODE%"=="" set "MODE=run"
if /i "%MODE%"=="run" goto :environment
if /i "%MODE%"=="probe" goto :environment
if /i "%MODE%"=="plot" goto :environment
if /i "%MODE%"=="check" goto :environment
goto :usage

:environment
if not exist "environment.bat" goto :missing_environment
if not exist "scripts\check_environment.py" goto :missing_environment
if not exist "%CONFIG%" goto :missing_config
if not exist "%DIAGNOSTIC_CONFIG%" goto :missing_config
call environment.bat
if not "%ERRORLEVEL%"=="0" goto :failed
echo [1/3] Checking the shared Python and CUDA environment...
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% "scripts\check_environment.py" check --device "%DEVICE%"
if not "%ERRORLEVEL%"=="0" goto :failed
if /i "%MODE%"=="check" goto :checked
if not exist "%RECORD%\metadata.json" goto :missing_record
if /i "%MODE%"=="plot" goto :plot
set "RUN_LIMITS="
if /i "%MODE%"=="probe" set "RUN_LIMITS=--max-steps-this-run %PROBE_STEPS%"

echo [2/3] Training with spherical-uniform query points and the configured objective...
if /i "%MODE%"=="probe" echo Probe: at most %PROBE_STEPS% additional updates; the final plan stays fixed.
if exist "%OUTPUT%\checkpoint.pt" goto :resume
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow train-rainbow ^
    --record "%RECORD%" ^
    --config "%CONFIG%" ^
    --output "%OUTPUT%" ^
    --device "%DEVICE%" %RUN_LIMITS%
if not "%ERRORLEVEL%"=="0" goto :failed
goto :after_training

:resume
echo Resuming the saved optimizer and random state from "%OUTPUT%\checkpoint.pt".
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow train-rainbow ^
    --record "%RECORD%" ^
    --config "%CONFIG%" ^
    --output "%OUTPUT%" ^
    --resume "%OUTPUT%\checkpoint.pt" ^
    --device "%DEVICE%" %RUN_LIMITS%
if not "%ERRORLEVEL%"=="0" goto :failed

:after_training
if /i "%MODE%"=="probe" goto :probe_done
echo [3/3] Saving angular PDF profiles for the selected checkpoint...
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow diagnose-rainbow ^
    --record "%RECORD%" ^
    --run "%OUTPUT%" ^
    --sweep-config "%DIAGNOSTIC_CONFIG%" ^
    --output "%OUTPUT%\diagnostics" ^
    --device "%DEVICE%"
if not "%ERRORLEVEL%"=="0" goto :failed
echo.
echo [DONE] Training, independent evaluation, full PDF maps, and angular profiles finished.
echo Comparison: "%OUTPUT%\plots\comparison.png"
echo Every actual training point: "%OUTPUT%\plots\training_samples.png"
echo Metrics: "%OUTPUT%\metrics.json"
echo Angular profiles: "%OUTPUT%\diagnostics"
echo Monitor: call monitor.bat "%OUTPUT%"
set "EXIT_CODE=0"
goto :finish

:probe_done
echo.
echo [DONE] Probe checkpoint saved. Run this batch without an argument to finish the plan and plots.
echo Continue: call train_log_uniform.bat
echo Preview selected weights now: call train_log_uniform.bat plot
echo Monitor: call monitor.bat "%OUTPUT%"
set "EXIT_CODE=0"
goto :finish

:plot
if not exist "%OUTPUT%\best.pt" goto :missing_checkpoint
echo Rendering the saved validation-selected checkpoint without optimizer updates...
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow plot-rainbow ^
    --record "%RECORD%" ^
    --checkpoint "%OUTPUT%\best.pt" ^
    --output "%OUTPUT%\plots_manual" ^
    --scatter training ^
    --batch-size 4096 ^
    --device "%DEVICE%"
if not "%ERRORLEVEL%"=="0" goto :failed
echo.
echo [DONE] Saved "%OUTPUT%\plots_manual\comparison.png" and all actual training points.
set "EXIT_CODE=0"
goto :finish

:checked
echo [DONE] Shared Python and CUDA check passed.
set "EXIT_CODE=0"
goto :finish

:missing_environment
echo [ERROR] Keep this batch, environment.bat, and scripts\check_environment.py in one checkout.
set "EXIT_CODE=1"
goto :finish

:missing_record
echo [ERROR] Missing "%RECORD%\metadata.json". Copy RECORD from your working execute.bat.
set "EXIT_CODE=1"
goto :finish

:missing_config
echo [ERROR] Apply the full uniform-training patch and check CONFIG and DIAGNOSTIC_CONFIG.
set "EXIT_CODE=1"
goto :finish

:missing_checkpoint
echo [ERROR] Missing "%OUTPUT%\best.pt". Run training or a probe before plotting.
set "EXIT_CODE=1"
goto :finish

:usage
echo Usage: train_log_uniform.bat [run^|probe^|plot^|check]
echo No argument runs or resumes one fit with automatic full maps after completion.
set "EXIT_CODE=2"
goto :finish

:failed
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo [ERROR] Command failed with exit code %EXIT_CODE%. See the output above.
echo Keep the original configuration for resume; changed experiments need a new OUTPUT.

:finish
popd
echo.
if not "%PHASEFLOW_PAUSE%"=="0" pause
exit /b %EXIT_CODE%
