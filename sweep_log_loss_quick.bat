@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem Requires the previously supplied log-density objective and beta-sweep patches.
rem Copy RECORD from your working execute.bat. Keep the new OUTPUT for this experiment.
rem Four trials: mixed NLL, beta 0.1, beta 1, and pure log regression.
rem Full native PDF maps are deferred to replot; scalar monitoring and angular cuts remain.
set "RECORD=..\rainbow\datasets\drop_a1_i20_700nm_q1800x3600_c90x1800"
set "CONFIG=configs\rainbow_log_quick.json"
set "SWEEP_CONFIG=configs\rainbow_log_quick_sweep.json"
set "OUTPUT=runs\rainbow_log_quick_gpu"
set "DEVICE=cuda:0"
set "PROBE_STEPS=250"

pushd "%~dp0"
if not "%ERRORLEVEL%"=="0" exit /b 1
if not "%~2"=="" goto :usage
set "MODE=%~1"
if "%MODE%"=="" set "MODE=run"
if /i "%MODE%"=="run" goto :environment
if /i "%MODE%"=="probe" goto :environment
if /i "%MODE%"=="first" goto :environment
if /i "%MODE%"=="plan" goto :environment
if /i "%MODE%"=="replot" goto :environment
goto :usage

:environment
if not exist "environment.bat" goto :missing_environment
if not exist "scripts\check_environment.py" goto :missing_environment
if not exist "src\phaseflow\log_sweep.py" goto :missing_log_sweep
call environment.bat
if not "%ERRORLEVEL%"=="0" goto :failed
if /i "%MODE%"=="plan" goto :plan
echo Checking the shared Python and CUDA environment...
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% "scripts\check_environment.py" check --device "%DEVICE%"
if not "%ERRORLEVEL%"=="0" goto :failed
if not exist "%RECORD%\metadata.json" goto :missing_record
if /i "%MODE%"=="replot" goto :replot
if not exist "%CONFIG%" goto :missing_config
if not exist "%SWEEP_CONFIG%" goto :missing_config
set "RUN_LIMITS="
if /i "%MODE%"=="probe" set "RUN_LIMITS=--max-trials-this-run 1 --max-steps-this-trial %PROBE_STEPS%"
if /i "%MODE%"=="first" set "RUN_LIMITS=--max-trials-this-run 1"

echo Quick comparison: L=4, H=64x64, K=64, lr=0.003, 2000 updates per trial.
if /i "%MODE%"=="probe" echo Probe: at most %PROBE_STEPS% additional updates in one unfinished trial.
if /i "%MODE%"=="first" echo First: complete at most one unfinished trial.
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow.log_sweep run ^
    --record "%RECORD%" ^
    --config "%CONFIG%" ^
    --sweep-config "%SWEEP_CONFIG%" ^
    --output "%OUTPUT%" ^
    --device "%DEVICE%" %RUN_LIMITS%
if not "%ERRORLEVEL%"=="0" goto :failed
echo.
echo [DONE] Requested execution finished. A probe or first run may leave pending trials.
echo Summary: "%OUTPUT%\summary.md"
echo Table: "%OUTPUT%\summary.csv"
echo Per-attempt timing in seconds: "%OUTPUT%\attempts"
echo Monitor: call monitor.bat "%OUTPUT%"
echo Continue: call sweep_log_loss_quick.bat
echo Full maps after completed trials: call sweep_log_loss_quick.bat replot
set "EXIT_CODE=0"
goto :finish

:plan
if not exist "%CONFIG%" goto :missing_config
if not exist "%SWEEP_CONFIG%" goto :missing_config
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow.log_sweep plan ^
    --record "%RECORD%" ^
    --config "%CONFIG%" ^
    --sweep-config "%SWEEP_CONFIG%" ^
    --output "%OUTPUT%" ^
    --device "%DEVICE%"
if not "%ERRORLEVEL%"=="0" goto :failed
set "EXIT_CODE=0"
goto :finish

:replot
if not exist "%OUTPUT%\manifest.json" goto :missing_sweep
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow.log_sweep replot ^
    --record "%RECORD%" ^
    --output "%OUTPUT%" ^
    --device "%DEVICE%"
if not "%ERRORLEVEL%"=="0" goto :failed
echo.
echo [DONE] Full native maps and every actual training point were requested for completed trials.
echo New results are under "%OUTPUT%\replots".
set "EXIT_CODE=0"
goto :finish

:missing_environment
echo [ERROR] Keep this batch, environment.bat, and scripts\check_environment.py in one checkout.
set "EXIT_CODE=1"
goto :finish

:missing_log_sweep
echo [ERROR] Apply the log-density objective and beta-sweep patches before this add-on.
set "EXIT_CODE=1"
goto :finish

:missing_record
echo [ERROR] Missing "%RECORD%\metadata.json". Copy RECORD from your working execute.bat.
set "EXIT_CODE=1"
goto :finish

:missing_config
echo [ERROR] Check CONFIG="%CONFIG%" and SWEEP_CONFIG="%SWEEP_CONFIG%".
set "EXIT_CODE=1"
goto :finish

:missing_sweep
echo [ERROR] OUTPUT must contain manifest.json from this quick log-loss sweep.
set "EXIT_CODE=1"
goto :finish

:usage
echo Usage: sweep_log_loss_quick.bat [run^|probe^|first^|plan^|replot]
echo No argument runs or exactly resumes the quick comparison.
set "EXIT_CODE=2"
goto :finish

:failed
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo [ERROR] Command failed with exit code %EXIT_CODE%. See the output above.
echo Use the interpreter selected in environment.bat.

:finish
popd
echo.
if not "%PHASEFLOW_PAUSE%"=="0" pause
exit /b %EXIT_CODE%
