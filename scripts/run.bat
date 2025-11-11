@echo off
setlocal

REM Create venv
if not exist .venv (
  py -3 -m venv .venv
)

call .venv\Scripts\activate

python -m pip install -U pip
pip install -r requirements.txt

REM Ensure ffmpeg is installed and on PATH

python src\doodle_bodmas_autorender.py

echo Done. See build\final_doodle_1080x1080.mp4
endlocal
