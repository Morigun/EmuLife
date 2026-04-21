@echo off
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv venv --without-pip
    if errorlevel 1 (
        echo Failed to create venv.
        pause
        exit /b 1
    )

    echo Installing pip...
    venv\Scripts\python.exe -m ensurepip --upgrade 2>nul
    if errorlevel 1 (
        echo Bootstrapping pip via get-pip.py...
        curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        venv\Scripts\python.exe get-pip.py
        del get-pip.py
    )
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Starting EmuLife...
python main.py
pause
