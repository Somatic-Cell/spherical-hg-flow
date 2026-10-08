@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem Place this file in the spherical-hg-flow repository root.
rem Apply the GPU/plots patch and install the project in .venv first.
rem Edit RECORD to select a directory containing metadata.json and the NPY files.
rem Use a new OUTPUT directory for each fresh training run.
set "RECORD=..\rainbow\datasets\drop_a1_i20_700nm_q1800x3600_c90x1800"
set "CONFIG=configs\rainbow_single.json"
set "OUTPUT=runs\rainbow_single_gpu"
set "DEVICE=cuda:0"
set "EVAL_SAMPLES=65536"
set "EVAL_SEED=2026"
set "PLOT_SAMPLES=32768"
set "PLOT_SEED=2027"

pushd "%~dp0"
if errorlevel 1 exit /b 1
set "PYTHON=.venv\Scripts\python.exe"

if not exist "%PYTHON%" goto :missing_python
if not exist "%CONFIG%" goto :missing_config
if not exist "%RECORD%\metadata.json" goto :missing_record

echo [1/5] Checking the selected CUDA device...
"%PYTHON%" -c "import sys, torch; d = torch.device(sys.argv[1]); assert d.type == 'cuda', 'A CUDA device is required'; assert torch.cuda.is_available(), 'CUDA unavailable: install a CUDA-enabled PyTorch build and check the GPU driver'; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('Device:', d, torch.cuda.get_device_name(d))" "%DEVICE%"
if errorlevel 1 goto :failed

echo [2/5] Validating the Rainbow CDF record...
"%PYTHON%" -m phaseflow inspect-rainbow --record "%RECORD%"
if errorlevel 1 goto :failed

echo [3/5] Training one condition...
rem Plotting is performed explicitly in step 5.
"%PYTHON%" -m phaseflow train-rainbow ^
    --record "%RECORD%" ^
    --config "%CONFIG%" ^
    --output "%OUTPUT%" ^
    --device "%DEVICE%" ^
    --no-plots
if errorlevel 1 goto :failed

echo [4/5] Evaluating the validation-selected checkpoint...
"%PYTHON%" -m phaseflow evaluate-rainbow ^
    --record "%RECORD%" ^
    --checkpoint "%OUTPUT%\best.pt" ^
    --samples %EVAL_SAMPLES% ^
    --seed %EVAL_SEED% ^
    --device "%DEVICE%" ^
    --output "%OUTPUT%\evaluation.json"
if errorlevel 1 goto :failed

echo [5/5] Plotting the CDF PDF, CDF samples, and NF PDF...
"%PYTHON%" -m phaseflow plot-rainbow ^
    --record "%RECORD%" ^
    --checkpoint "%OUTPUT%\best.pt" ^
    --output "%OUTPUT%\plots" ^
    --samples %PLOT_SAMPLES% ^
    --seed %PLOT_SEED% ^
    --device "%DEVICE%"
if errorlevel 1 goto :failed

set "EXIT_CODE=0"
echo.
echo [DONE] Training, evaluation, and plotting completed.
echo Evaluation: "%OUTPUT%\evaluation.json"
echo Comparison: "%OUTPUT%\plots\comparison.png"
goto :finish

:missing_python
echo [ERROR] Missing "%PYTHON%". Install the project in .venv first.
set "EXIT_CODE=1"
goto :finish

:missing_config
echo [ERROR] Missing config: "%CONFIG%"
set "EXIT_CODE=1"
goto :finish

:missing_record
echo [ERROR] Missing "%RECORD%\metadata.json". Edit RECORD at the top of this file.
set "EXIT_CODE=1"
goto :finish

:failed
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo [ERROR] Command failed with exit code %EXIT_CODE%. See the output above.

:finish
popd
echo.
pause
exit /b %EXIT_CODE%
