@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem Keep these separate. PARENT is read-only; OUTPUT holds the two child runs.
set "PARENT=runs\rainbow_log_uniform_gpu"
set "OUTPUT=runs\rainbow_log_uniform_continue_gpu"
set "DEVICE=cuda:0"
rem Leave RECORD empty to reuse the CDF directory saved in the parent checkpoint.
rem If the dataset moved, set it to the RECORD used in train_log_uniform.bat.
set "RECORD="
set "MODE=%~1"
if not defined MODE set "MODE=run"

pushd "%~dp0"
if errorlevel 1 exit /b 1
if not exist "environment.bat" goto :missing
call environment.bat
if errorlevel 1 goto :failed
if not defined PHASEFLOW_PYTHON goto :missing
set "PYTHONUNBUFFERED=1"
if not defined CUBLAS_WORKSPACE_CONFIG set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
if defined RECORD goto :with_record

"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow.uniform_branch_sweep "%MODE%" ^
  --parent "%PARENT%" --output "%OUTPUT%" --device "%DEVICE%" ^
  --parent-step 2000 --steps 5000 --milestones 3000 5000 --learning-rates 0.003 0.001
set "RESULT=%ERRORLEVEL%"
goto :done

:with_record
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow.uniform_branch_sweep "%MODE%" ^
  --parent "%PARENT%" --output "%OUTPUT%" --record "%RECORD%" --device "%DEVICE%" ^
  --parent-step 2000 --steps 5000 --milestones 3000 5000 --learning-rates 0.003 0.001
set "RESULT=%ERRORLEVEL%"
goto :done

:missing
echo [ERROR] Place the complete add-on at the repository root next to environment.bat.
set "RESULT=1"
goto :done

:failed
set "RESULT=%ERRORLEVEL%"
if "%RESULT%"=="0" set "RESULT=1"

:done
if not "%RESULT%"=="0" echo [ERROR] Continuation did not complete. See the message above.
popd
exit /b %RESULT%
