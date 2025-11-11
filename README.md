# Doodle BODMAS Video (1080×1080)

A fully automated whiteboard-doodle video generator for the expression:
9 + 2 × 18 – 2, following BODMAS. Renders a portrait 1080×1080 MP4 with:
- Friendly African female teacher (smart-casual blouse, short natural hair, glasses, warm smile)
- Kenyan English female voiceover
- Soft classroom background music (synthesized in-script)
- Hand-drawn (write/draw-on) animations

Output
- build/final_doodle_1080x1080.mp4 (~1:25–1:30)
- build/audio_master.wav
- build/captions.srt

Requirements
- Python 3.9+ (3.10+ recommended)
- FFmpeg on PATH
- Internet access for TTS (edge-tts)
- System libs needed by Manim (Cairo, Pango)

Quick start (macOS/Linux)
1) Install FFmpeg
   - macOS: brew install ffmpeg
   - Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y ffmpeg
2) Run:
   - bash scripts/run.sh

Quick start (Windows)
1) Install FFmpeg:
   - Download from https://ffmpeg.org/download.html and add to PATH
2) Run:
   - scripts\\run.bat

GitHub Actions (no local setup)
- Push this repo to GitHub.
- Go to Actions → “Render Doodle Video” → Run workflow.
- After it finishes, download the “rendered-video” artifact (MP4 + SRT).

Customization
- Voice: Edit VOICE in src/doodle_bodmas_autorender.py (default: en-KE-AsiliaNeural).
- Music: Synthesized on the fly; tune sound in synth_soft_classroom_music().
- Scene timings and captions: Edit SCENES in src/doodle_bodmas_autorender.py.
- Resolution: Change config.pixel_width/height in the Manim code generator within src/doodle_bodmas_autorender.py.

Troubleshooting
- FFmpeg not found: Install and ensure it’s on PATH.
- TTS errors: Network issues may block edge-tts. Re-run later or replace with an offline TTS.
- Manim font issues: This project uses DejaVu Sans; install system fonts if needed (Ubuntu: sudo apt-get install -y fonts-dejavu).
- Linux Manim deps: You may need Cairo/Pango libs:
  sudo apt-get install -y libcairo2-dev libpango1.0-dev libgdk-pixbuf-2.0-0 pkg-config

License
- MIT (see LICENSE)
