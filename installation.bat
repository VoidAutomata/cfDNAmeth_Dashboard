@echo off
SET ENV_NAME=rf_env

:: Check if rf_env folder exists
IF NOT EXIST %ENV_NAME%\ (
    echo Creating virtual environment: %ENV_NAME%
    python -m venv %ENV_NAME%
)

:: Activate the virtual environment
call %ENV_NAME%\Scripts\activate.bat

:: Install packages from requirements.txt
pip install -r requirements.txt

echo.
echo Setup complete!


cd rf_env
call Scripts\activate.bat
cd ..\backend 
pip install -r requirements.txt
streamlit run home.py