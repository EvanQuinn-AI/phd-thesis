@echo off
REM Navigate to the directory where the batch file is located
cd %~dp0

REM Use the full path to the Anaconda condabin folder to activate the 'cpu_automation' environment
call "C:\ProgramData\anaconda3\condabin\conda.bat" activate cpu_automation

REM Run the Streamlit app using the Python from the Anaconda environment
streamlit run cpu-version/app.py

REM Keep the command prompt open to view any output or errors
pause
