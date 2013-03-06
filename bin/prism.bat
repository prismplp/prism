@echo off

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

if not "%CMDEXTVERSION%" == "" goto newer

:older

if "%PRISM%" == "" set PRISM=C:\prism
set PRISM=%PRISM%\bin

%PRISM%\prism_win32.exe -c -p %PAREA% -s %STACK% -b %TRAIL% -t %TABLE% %PRISM%\bp.out %PRISM%\prism.out %PRISM%\fo_all.out %1 %2 %3 %4 %5 %6 %7 %8 %9

goto end

:newer

if "%PRISM%" == "" (set PRISM=%~dp0) else (set PRISM=%PRISM%\bin)

%PRISM%\prism_win32.exe -c -p %PAREA% -s %STACK% -b %TRAIL% -t %TABLE% %PRISM%\bp.out %PRISM%\prism.out %PRISM%\foc.out %*

:end
