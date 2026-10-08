@echo off
rem Compatibility entry point. Edit environment.bat and execute.bat only.
rem Installation and training intentionally use the same experiment settings.
rem Control transfers to execute.bat; no second interpreter/configuration is kept here.
"%~dp0execute.bat" %*
