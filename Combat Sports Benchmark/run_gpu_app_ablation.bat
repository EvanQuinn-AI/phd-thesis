@echo off

REM Navigate to the directory where the batch file is located
cd /d "%~dp0"

REM Activate conda environment
call "C:\ProgramData\anaconda3\condabin\conda.bat" activate automation_env

REM -------------------------------
REM Run Ablation on Real Videos
REM -------------------------------

python gpu-version/app_with_ablation.py --video "data/Sample 1.mp4" --frames 300
python gpu-version/app_with_ablation.py --video "data/Sample 2.mp4" --frames 300
python gpu-version/app_with_ablation.py --video "data/Sample 3.mp4" --frames 300

REM -------------------------------
REM Synthetic Timing Benchmark
REM -------------------------------

python gpu-version/app_with_ablation.py --synthetic --frames 200

REM -------------------------------
REM Regenerate Chapter 6 Text Only
REM -------------------------------

python gpu-version/app_with_ablation.py --text-only

REM Keep window open
pause