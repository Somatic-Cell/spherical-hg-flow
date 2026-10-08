@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem Place this file in the spherical-hg-flow repository root.
rem First run: run_rainbow_python314.bat setup
rem Then run:  run_rainbow_python314.bat
rem Select the Python environment containing your existing CUDA PyTorch build.
rem For a venv/conda environment, set PYTHON to its python.exe and clear PYTHON_ARGS.
set "PYTHON=py"
set "PYTHON_ARGS=-3.14"

set "RECORD=D:\rainbow\output\records\i0000\w0000"
set "CONFIG=configs\rainbow_single.json"
set "OUTPUT=runs\rainbow_single_py314_cu130"
set "DEVICE=cuda:0"
set "EVAL_SAMPLES=65536"
set "EVAL_SEED=2026"
set "PLOT_SAMPLES=32768"
set "PLOT_SEED=2027"

pushd "%~dp0"
if errorlevel 1 exit /b 1
if not exist "pyproject.toml" goto :wrong_directory
if not exist "src\phaseflow\__main__.py" goto :wrong_directory
if not "%~1"=="" if /i not "%~1"=="setup" goto :usage

echo Checking the existing Python and CUDA PyTorch environment...
"%PYTHON%" %PYTHON_ARGS% -c "import struct, sys; print('Python:', sys.version); print('Executable:', sys.executable); sys.exit('Python 3.14, 64-bit, is required by this launcher. Check PYTHON and PYTHON_ARGS.') if sys.version_info[:2] != (3, 14) or struct.calcsize('P') != 8 else None"
if errorlevel 1 goto :failed
"%PYTHON%" %PYTHON_ARGS% -c "import sys, torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); sys.exit('This launcher expects existing PyTorch 2.14.1 built for CUDA 13.0. Check the selected Python environment.') if str(torch.__version__).split('+')[0] != '2.14.1' or torch.version.cuda != '13.0' else None; d = torch.device(sys.argv[1]); sys.exit('A CUDA device is required.') if d.type != 'cuda' else None; print('GPU:', torch.cuda.get_device_name(d)); x = torch.ones(1, device=d, requires_grad=True); x.square().sum().backward(); print('GPU gradient check:', x.grad.item())" "%DEVICE%"
if errorlevel 1 goto :failed
if /i "%~1"=="setup" goto :setup

if not exist "%CONFIG%" goto :missing_config
if not exist "%RECORD%\metadata.json" goto :missing_record

echo [1/4] Validating the Rainbow CDF record...
"%PYTHON%" %PYTHON_ARGS% -m phaseflow inspect-rainbow --record "%RECORD%"
if errorlevel 1 goto :failed

echo [2/4] Training one condition...
"%PYTHON%" %PYTHON_ARGS% -m phaseflow train-rainbow ^
    --record "%RECORD%" ^
    --config "%CONFIG%" ^
    --output "%OUTPUT%" ^
    --device "%DEVICE%" ^
    --no-plots
if errorlevel 1 goto :failed

echo [3/4] Evaluating the validation-selected checkpoint...
"%PYTHON%" %PYTHON_ARGS% -m phaseflow evaluate-rainbow ^
    --record "%RECORD%" ^
    --checkpoint "%OUTPUT%\best.pt" ^
    --samples %EVAL_SAMPLES% ^
    --seed %EVAL_SEED% ^
    --device "%DEVICE%" ^
    --output "%OUTPUT%\evaluation.json"
if errorlevel 1 goto :failed

echo [4/4] Plotting the CDF PDF, CDF samples, and NF PDF...
"%PYTHON%" %PYTHON_ARGS% -m phaseflow plot-rainbow ^
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

:setup
echo Installing the project while retaining the exact installed torch version...
rem Pin the installed version, including its CUDA build suffix. Do not force reinstall.
"%PYTHON%" %PYTHON_ARGS% -c "import importlib.metadata as md, subprocess, sys; v = md.version('torch'); print('Keeping installed torch==' + v, flush=True); sys.exit(subprocess.call([sys.executable, '-m', 'pip', 'install', '-e', '.[dev]', 'torch==' + v]))"
if errorlevel 1 goto :failed
"%PYTHON%" %PYTHON_ARGS% -m phaseflow --help
if errorlevel 1 goto :failed
set "EXIT_CODE=0"
echo.
echo [DONE] Setup completed. Edit RECORD, then run this file without setup.
goto :finish

:wrong_directory
echo [ERROR] Place this file in the spherical-hg-flow repository root.
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

:usage
echo Usage: run_rainbow_python314.bat [setup]
set "EXIT_CODE=1"
goto :finish

:failed
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo [ERROR] Command failed with exit code %EXIT_CODE%. See the output above.
echo Installation and training must use the same Python environment.

:finish
popd
echo.
pause
exit /b %EXIT_CODE%
