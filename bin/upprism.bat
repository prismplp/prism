@echo off
if not "%PRISM_MINGW%"=="1" goto prism_original
setlocal
if not defined PRISM_MINGW_BITS set PRISM_MINGW_BITS=64
if not "%PRISM_MINGW_BITS%"=="32" if not "%PRISM_MINGW_BITS%"=="64" (
    echo PRISM_MINGW_BITS must be 32 or 64 1>&2
    exit /b 1
)
if not defined PRISM_PAREA set PRISM_PAREA=8000000
if not defined PRISM_STACK set PRISM_STACK=5000000
if not defined PRISM_TRAIL set PRISM_TRAIL=2000000
if not defined PRISM_TABLE set PRISM_TABLE=10000000
"%~dp0prism_up_mingw%PRISM_MINGW_BITS%.exe" -c -p %PRISM_PAREA% -s %PRISM_STACK% -b %PRISM_TRAIL% -t %PRISM_TABLE% "%~dp0bp.out" "%~dp0prism.out" "%~dp0foc.out" "%~dp0batch.out" %*
exit /b %ERRORLEVEL%

:prism_original

rem - the directory where you installed the system
rem - (not required if you're using Windows 2000 or higher)

set PRISM=

rem - default size parameters passed to the system

set PAREA=4000000
set STACK=5000000
set TRAIL=2000000
set TABLE=10000000


rem ------------------------------------------------------------------------
rem ------[[  Don't edit below unless you know what you're doing.  ]]-------
rem ------------------------------------------------------------------------

if "%PROCESSOR_ARCHITECTURE%" EQU "x86" (
	set PRISM_BIN=prism_win32.exe
)
if "%PROCESSOR_ARCHITECTURE%" NEQ "x86" (
	set PRISM_BIN=prism_win64.exe
)


if not "%CMDEXTVERSION%" == "" goto newer

:older

if "%PRISM%" == "" set PRISM=C:\prism
set PRISM=%PRISM%\bin

%PRISM%\%PRISM_BIN% -c -p %PAREA% -s %STACK% -b %TRAIL% -t %TABLE% %PRISM%\bp.out %PRISM%\prism.out %PRISM%\foc.out %PRISM%\batch.out %1 %2 %3 %4 %5 %6 %7 %8 %9

goto end

:newer

if "%PRISM%" == "" (set PRISM=%~dp0) else (set PRISM=%PRISM%\bin)

%PRISM%\%PRISM_BIN% -c -p %PAREA% -s %STACK% -b %TRAIL% -t %TABLE% %PRISM%\bp.out %PRISM%\prism.out %PRISM%\foc.out %PRISM%\batch.out %*

:end
