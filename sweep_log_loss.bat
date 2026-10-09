@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem Use exactly the same RECORD and Python environment as the working execute.bat.
rem New output only: keep all prior NLL/capacity/encoding sweeps unchanged.
rem run: target-NLL control plus six shared-pool beta trials.
rem plan: print configuration and update counts without opening the CDF or GPU.
rem replot: redraw completed best.pt weights into a new timestamped subtree.
set "RECORD=..\rainbow\datasets\drop_a1_i20_700nm_q1800x3600_c90x1800"
set "CONFIG=configs\rainbow_log_loss.json"
set "SWEEP_CONFIG=configs\rainbow_log_sweep.json"
set "OUTPUT=runs\rainbow_log_loss_gpu"
set "DEVICE=cuda:0"

pushd "%~dp0"
if not "%ERRORLEVEL%"=="0" exit /b 1
if not "%~2"=="" goto :usage
set "MODE=%~1"
if "%MODE%"=="" set "MODE=run"
if /i "%MODE%"=="run" goto :environment
if /i "%MODE%"=="plan" goto :environment
if /i "%MODE%"=="replot" goto :environment
goto :usage

:environment
if not exist "environment.bat" goto :missing_environment
if not exist "scripts\check_environment.py" goto :missing_environment
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

echo Comparing beta using common validation log RMSE and independent training runs.
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow.log_sweep run ^
    --record "%RECORD%" ^
    --config "%CONFIG%" ^
    --sweep-config "%SWEEP_CONFIG%" ^
    --output "%OUTPUT%" ^
    --device "%DEVICE%"
if not "%ERRORLEVEL%"=="0" goto :failed
echo.
echo [DONE] Summary: "%OUTPUT%\summary.md"
echo Table: "%OUTPUT%\summary.csv"
echo Plot: "%OUTPUT%\summary.png"
echo Selection: "%OUTPUT%\selection.json"
echo Monitor: call monitor.bat "%OUTPUT%"
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
echo [DONE] Revised plots are in a new directory under "%OUTPUT%\replots".
echo Original trial checkpoints, metrics and plots were preserved.
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
echo [ERROR] Check CONFIG="%CONFIG%" and SWEEP_CONFIG="%SWEEP_CONFIG%".
set "EXIT_CODE=1"
goto :finish

:missing_sweep
echo [ERROR] OUTPUT must contain manifest.json from this log-loss sweep.
set "EXIT_CODE=1"
goto :finish

:usage
echo Usage: sweep_log_loss.bat [run^|plan^|replot]
echo No argument runs or exactly resumes the beta comparison.
set "EXIT_CODE=2"
goto :finish

:failed
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo [ERROR] Command failed with exit code %EXIT_CODE%. See the output above.
echo Use the same interpreter selected in environment.bat; no automatic installation occurs.

:finish
popd
echo.
if not "%PHASEFLOW_PAUSE%"=="0" pause
exit /b %EXIT_CODE%
