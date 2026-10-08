@echo off
rem Shared interpreter selection for execute.bat, setup, and monitor.bat.
rem Use the Python 3.14 environment that contains your CUDA PyTorch build.
rem For an existing venv/Conda environment, set PHASEFLOW_PYTHON to the full
rem path of its python.exe and set PHASEFLOW_PYTHON_ARGS to an empty value.
rem Example: set "PHASEFLOW_PYTHON=C:\path with spaces\venv\Scripts\python.exe"
rem          set "PHASEFLOW_PYTHON_ARGS="
set "PHASEFLOW_PYTHON=py"
set "PHASEFLOW_PYTHON_ARGS=-3.14"

rem Set PHASEFLOW_PAUSE=0 before calling a launcher for unattended execution.
if not defined PHASEFLOW_PAUSE set "PHASEFLOW_PAUSE=1"
exit /b 0
