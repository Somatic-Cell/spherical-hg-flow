@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem Interpreter selection is shared with setup and monitor in environment.bat.
rem Usage: execute.bat [setup^|check^|resume^|precision]
rem No argument starts a fresh run; choose a new OUTPUT for each experiment.
rem resume uses OUTPUT\checkpoint.pt and the original configuration.
rem precision evaluates OUTPUT\best.pt on 4096 points without further training.
set "RECORD=..\rainbow\datasets\drop_a1_i20_700nm_q1800x3600_c90x1800"
set "CONFIG=configs\rainbow_single.json"
set "OUTPUT=runs\rainbow_single_gpu"
set "DEVICE=cuda:0"
set "EVAL_SAMPLES=65536"
set "EVAL_SEED=2026"

pushd "%~dp0"
if not "%ERRORLEVEL%"=="0" exit /b 1
if not exist "environment.bat" goto :missing_launcher
if not exist "scripts\check_environment.py" goto :missing_launcher
call environment.bat
if not "%ERRORLEVEL%"=="0" goto :failed
if not exist "pyproject.toml" goto :missing_launcher
if not "%~2"=="" goto :usage
set "MODE=%~1"
if "%MODE%"=="" set "MODE=run"
if /i "%MODE%"=="setup" goto :setup
if /i "%MODE%"=="check" goto :check
if /i "%MODE%"=="precision" goto :precision
if /i "%MODE%"=="resume" goto :run
if /i "%MODE%"=="run" goto :run
goto :usage

:setup
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% "scripts\check_environment.py" setup --device "%DEVICE%"
if not "%ERRORLEVEL%"=="0" goto :failed
echo.
echo [DONE] Setup completed. Edit RECORD in execute.bat, then run execute.bat.
set "EXIT_CODE=0"
goto :finish

:check
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% "scripts\check_environment.py" check --device "%DEVICE%"
if not "%ERRORLEVEL%"=="0" goto :failed
echo.
echo [DONE] Python, phaseflow, and CUDA checks passed. No CDF was loaded.
set "EXIT_CODE=0"
goto :finish

:precision
echo Checking the shared Python environment before the precision diagnostic...
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% "scripts\check_environment.py" check --device "%DEVICE%"
if not "%ERRORLEVEL%"=="0" goto :failed
if not exist "%RECORD%\metadata.json" goto :missing_record
if not exist "%OUTPUT%\best.pt" goto :missing_best
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow precision-rainbow ^
    --record "%RECORD%" ^
    --checkpoint "%OUTPUT%\best.pt" ^
    --output "%OUTPUT%\precision.json" ^
    --samples 4096 ^
    --device "%DEVICE%"
if not "%ERRORLEVEL%"=="0" goto :failed
echo.
echo [DONE] Precision diagnostic: "%OUTPUT%\precision.json"
set "EXIT_CODE=0"
goto :finish

:run
echo [1/5] Checking the shared Python environment before loading the CDF...
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% "scripts\check_environment.py" check --device "%DEVICE%"
if not "%ERRORLEVEL%"=="0" goto :failed
if not exist "%CONFIG%" goto :missing_config
if not exist "%RECORD%\metadata.json" goto :missing_record
if /i "%MODE%"=="resume" if not exist "%OUTPUT%\checkpoint.pt" goto :missing_checkpoint

echo [2/5] Validating the Rainbow CDF record...
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow inspect-rainbow --record "%RECORD%"
if not "%ERRORLEVEL%"=="0" goto :failed
if /i "%MODE%"=="resume" goto :resume

echo [3/5] Training one condition...
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow train-rainbow ^
    --record "%RECORD%" ^
    --config "%CONFIG%" ^
    --output "%OUTPUT%" ^
    --device "%DEVICE%" ^
    --no-plots
if not "%ERRORLEVEL%"=="0" goto :failed
goto :evaluate

:resume
echo [3/5] Resuming one condition from checkpoint.pt...
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow train-rainbow ^
    --record "%RECORD%" ^
    --config "%CONFIG%" ^
    --output "%OUTPUT%" ^
    --device "%DEVICE%" ^
    --resume "%OUTPUT%\checkpoint.pt" ^
    --no-plots
if not "%ERRORLEVEL%"=="0" goto :failed

:evaluate
echo [4/5] Evaluating the validation-selected checkpoint...
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow evaluate-rainbow ^
    --record "%RECORD%" ^
    --checkpoint "%OUTPUT%\best.pt" ^
    --samples %EVAL_SAMPLES% ^
    --seed %EVAL_SEED% ^
    --device "%DEVICE%" ^
    --output "%OUTPUT%\evaluation.json"
if not "%ERRORLEVEL%"=="0" goto :failed

echo [5/5] Plotting the CDF PDF, every fixed training point, and NF PDF...
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow plot-rainbow ^
    --record "%RECORD%" ^
    --checkpoint "%OUTPUT%\best.pt" ^
    --output "%OUTPUT%\plots" ^
    --scatter training ^
    --device "%DEVICE%"
if not "%ERRORLEVEL%"=="0" goto :failed

set "EXIT_CODE=0"
echo.
echo [DONE] Training, evaluation, and plotting completed.
echo Evaluation: "%OUTPUT%\evaluation.json"
echo Comparison: "%OUTPUT%\plots\comparison.png"
echo Live training curves: run monitor.bat in another window.
goto :finish

:missing_launcher
echo [ERROR] Keep this file, environment.bat, and scripts\check_environment.py in this checkout.
set "EXIT_CODE=1"
goto :finish

:missing_config
echo [ERROR] Missing config: "%CONFIG%"
set "EXIT_CODE=1"
goto :finish

:missing_record
echo [ERROR] Missing "%RECORD%\metadata.json". Edit RECORD in execute.bat.
set "EXIT_CODE=1"
goto :finish

:missing_checkpoint
echo [ERROR] Missing "%OUTPUT%\checkpoint.pt". Check OUTPUT before requesting resume.
set "EXIT_CODE=1"
goto :finish

:missing_best
echo [ERROR] Missing "%OUTPUT%\best.pt". Check OUTPUT before requesting precision diagnostics.
set "EXIT_CODE=1"
goto :finish

:usage
echo Usage: execute.bat [setup^|check^|resume^|precision]
echo No argument starts a fresh run. Existing training results are not deleted.
set "EXIT_CODE=1"
goto :finish

:failed
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo [ERROR] Command failed with exit code %EXIT_CODE%. See the output above.
echo Setup and training both use the interpreter selected in environment.bat.

:finish
popd
echo.
if not "%PHASEFLOW_PAUSE%"=="0" pause
exit /b %EXIT_CODE%
