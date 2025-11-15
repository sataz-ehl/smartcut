@echo off
REM ============================================================
REM Smartcut - Windows Setup Script
REM ============================================================
REM This script sets up a Python virtual environment for Smartcut
REM using Python 3.10 located at C:\Python310\python.exe
REM ============================================================

echo.
echo ============================================================
echo Smartcut - Windows Setup Script
echo ============================================================
echo.

REM Check if Python exists at the specified location
set PYTHON_PATH=C:\Python310\python.exe

if not exist "%PYTHON_PATH%" (
    echo ERROR: Python not found at %PYTHON_PATH%
    echo.
    echo Please ensure Python 3.10 is installed at this location, or
    echo edit this script to point to your Python installation.
    echo.
    echo You can download Python from: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo Found Python at: %PYTHON_PATH%
echo.

REM Display Python version
echo Checking Python version...
"%PYTHON_PATH%" --version
echo.

REM Create virtual environment
echo Creating Python virtual environment...
"%PYTHON_PATH%" -m venv venv

if errorlevel 1 (
    echo.
    echo ERROR: Failed to create virtual environment
    echo.
    pause
    exit /b 1
)

echo ✓ Virtual environment created successfully
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

if errorlevel 1 (
    echo.
    echo ERROR: Failed to activate virtual environment
    echo.
    pause
    exit /b 1
)

echo ✓ Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

if errorlevel 1 (
    echo.
    echo WARNING: Failed to upgrade pip (continuing anyway...)
    echo.
)

echo.
echo ============================================================
echo Installation Options
echo ============================================================
echo.
echo Please choose an installation method:
echo.
echo 1. Install from requirements.txt (manual dependency installation)
echo 2. Install smartcut package with all dependencies (recommended)
echo 3. Skip installation (set up environment only)
echo.
choice /C 123 /N /M "Enter your choice (1, 2, or 3): "

if errorlevel 3 goto skip_install
if errorlevel 2 goto install_package
if errorlevel 1 goto install_requirements

:install_requirements
echo.
echo Installing dependencies from requirements.txt...
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies
    echo.
    pause
    exit /b 1
)

echo ✓ Dependencies installed successfully
goto install_complete

:install_package
echo.
echo Installing smartcut package...
python -m pip install -e .

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install smartcut package
    echo.
    pause
    exit /b 1
)

echo ✓ Smartcut package installed successfully
goto install_complete

:skip_install
echo.
echo Skipping installation...
goto install_complete

:install_complete
echo.
echo ============================================================
echo Setup Complete!
echo ============================================================
echo.
echo Your Python virtual environment is ready to use.
echo.
echo To use smartcut:
echo   1. Activate the virtual environment:
echo      venv\Scripts\activate.bat
echo.
echo   2. Run smartcut:
echo      python -m smartcut input.mp4 output.mp4 --keep 10,20
echo.
echo   Or if you installed the package:
echo      smartcut input.mp4 output.mp4 --keep 10,20
echo.
echo Examples with fade effects:
echo   - Fade-in only:
echo     smartcut input.mp4 output.mp4 --keep 30:fadein,40
echo.
echo   - Fade-out only:
echo     smartcut input.mp4 output.mp4 --keep 50,60:fadeout
echo.
echo   - Both fades with custom durations:
echo     smartcut input.mp4 output.mp4 --keep 70:fadein:1.5,80:fadeout:2.0
echo.
echo   - Multiple segments with different fades:
echo     smartcut input.mp4 output.mp4 --keep 30,40 --keep 60:fadein,70:fadeout
echo.
echo For more information, see README.md or run:
echo   smartcut --help
echo.
echo ============================================================
echo.
pause
