@echo off
REM Navigate to the directory where the batch file is located
cd %~dp0

REM Use the full path to the Anaconda condabin folder to activate the 'automation_env' environment
call "C:\ProgramData\anaconda3\condabin\conda.bat" activate automation_env

REM Run the Streamlit app using the Python from the Anaconda environment
streamlit run gpu-version/app.py

REM Keep the command prompt open to view any output or errors
pause
