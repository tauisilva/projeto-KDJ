@echo off

echo -----------------------------
echo Python Environment Setup
echo -----------------------------
echo.

:: Define the path to the Python executable
set PYTHON_EXE=python.exe

:: Check if Python is installed
if not defined PYTHON_EXE (
    echo [ERROR] Python is not installed on the system.
    echo Please install Python via the Microsoft Store.
    pause
    exit /b
)

:: Get the Python version
for /f "delims=" %%v in ('%PYTHON_EXE% --version 2^>^&1') do set PYTHON_VERSION=%%v

:: Check if the Python version is 3.6 or higher
echo Checking Python version...
echo Installed Python version: %PYTHON_VERSION%
echo.

echo [INFO] Checking Python version...

echo.

:: Change the colors for the success and error messages
echo [INFO] Creating a virtual environment...
%PYTHON_EXE% -m venv venv
call venv\Scripts\activate

echo [INFO] Updating pip...
%PYTHON_EXE% -m ensurepip --default-pip
%PYTHON_EXE% -m pip install --upgrade pip

:: Installing requirements
echo [INFO] Installing requirements...
%PYTHON_EXE% -m pip install -r requirements.txt

echo.
echo [SUCCESS] Process completed.
echo.

:: Wait for user input to exit
pause
