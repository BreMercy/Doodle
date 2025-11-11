# Renders a 1080x1080 whiteboard-doodle video with Kenyan English VO and soft classroom music.
# Outputs: build/final_doodle_1080x1080.mp4

import time
import os
import sys
import math
import json
import asyncio
import subprocess
from pathlib import Path

from pydub import AudioSegment, effects
import numpy as np
import edge_tts
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip

BUILD = Path("build")
ASSETS = BUILD / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

SCENES = [
    {
        "name": "Scene1_TitleIntro",
        "min_duration": 10.0,
        "vo": "Hello everyone! Today, we’re going to solve this question together — nine plus two times eighteen, minus two. Let’s find out the correct answer!",
        "caption": "Solving 9 + 2 × 18 – 2 Step by Step!"
    },
    {
        "name": "Scene2_ReadQuestion",
        "min_duration": 10.0,
        "vo": "Hmm… it looks simple, but we have to be careful. If we just go from left to right, we might get it wrong.",
        "caption": "Let’s read it carefully!"
    },
    {
        "name": "Scene3_BODMAS",
        "min_duration": 15.0,
        "vo": "That’s why we use BODMAS — it helps us know which part to do first! B for brackets, O for orders, D for division, M for multiplication, A for addition, and S for subtraction.",
        "caption": "Remember BODMAS!"
    },
    {
        "name": "Scene4_MultiplyFirst",
        "min_duration": 15.0,
        "vo": "Now, let’s look at the question again. We see two times eighteen — that comes first. Two times eighteen equals thirty-six.",
        "caption": "Step 1: Do the multiplication — 2 × 18!"
    },
    {
        "name": "Scene5_Simplify",
        "min_duration": 10.0,
        "vo": "Great! So now our expression becomes nine plus thirty-six minus two.",
        "caption": "Now the expression becomes 9 + 36 – 2"
    },
    {
        "name": "Scene6_AddSubtract",
        "min_duration": 15.0,
        "vo": "Next, we add first: nine plus thirty-six equals forty-five. Then we subtract two. Forty-five minus two equals… forty-three!",
        "caption": "Step 2: Add, then subtract!"
    },
    {
        "name": "Scene7_FinalAnswer",
        "min_duration": 10.0,
        "vo": "And that’s our final answer — forty-three! Always remember to follow BODMAS when you solve math problems. Great job, everyone!",
        "caption": "The correct answer is 43!"
    }
]

VOICE = "en-KE-AsiliaNeural"  # Kenyan English female
SAMPLE_RATE = 44100

def ensure_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception:
        print("FFmpeg not found. Please install FFmpeg and ensure it's on your PATH.")
        sys.exit(1)
async def tts_edge_async(text, outfile, voice="en-KE-AsiliaNeural"):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(outfile))

def tts_with_fallback(text: str, out_mp3: Path, label: str):
    # Try edge-tts with retries
    last_err = None
    for attempt in range(5):
        try:
            print(f"[TTS] edge-tts {label} attempt {attempt+1}/5")
            asyncio.run(tts_edge_async(text, out_mp3))
            seg = AudioSegment.from_file(out_mp3)
            print(f"[TTS] edge-tts {label} OK, {len(seg)/1000:.2f}s")
            return seg
        except Exception as e:
            last_err = e
            time.sleep(1 + attempt)
    print(f"[TTS] edge-tts failed for {label}: {last_err}")
    # Fallback: espeak-ng (offline)
    wav_path = out_mp3.with_suffix(".wav")
    cmd = ["espeak-ng", "-v", "en+f3", "-s", "160", "-p", "40", "-w", str(wav_path), text]
    subprocess.run(cmd, check=True)
    seg = AudioSegment.from_file(wav_path).set_frame_rate(44100)
    seg.export(out_mp3, format="mp3")
    print(f"[TTS] espeak-ng {label} OK, {len(seg)/1000:.2f}s")
    return seg
    
async def tts_gen_async(text, outfile, voice=VOICE):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(outfile))

def tts_generate_all():
    print("Generating voiceover with fallback…")
    vo_durations = []
    for idx, sc in enumerate(SCENES, start=1):
        out = ASSETS / f"vo_{idx}.mp3"
        # Uses the fallback-capable TTS function we added earlier
        seg = tts_with_fallback(sc["vo"], out, f"scene {idx}")
        vo_durations.append(len(seg) / 1000.0)
    return vo_durations
    
    for idx, sc in enumerate(SCENES, start=1):
        out = ASSETS / f"vo_{idx}.mp3"
        try:
            loop.run_until_complete(tts_gen_async(sc["vo"], out))
        except RuntimeError:
            asyncio.run(tts_gen_async(sc["vo"], out))
        seg = AudioSegment.from_file(out)
        dur = len(seg) / 1000.0
        vo_durations.append(dur)
    return vo_durations

def synth_soft_classroom_music(total_seconds, out_wav):
    """
    Generates a gentle ‘classroom’ soft pad with simple bell accents.
    Royalty-free by construction.
    """
    print("Generating soft classroom background music…")
    sr = SAMPLE_RATE
    t = np.linspace(0, total_seconds, int(sr * total_seconds), endpoint=False)
    music = np.zeros_like(t)

    chords = [
        [261.63, 329.63, 392.00],  # C major: C E G
        [196.00, 246.94, 392.00],  # G: G B D (simplified)
        [220.00, 261.63, 329.63],  # Am: A C E
        [174.61, 220.00, 349.23],  # F: F A C
    ]
    bpm = 84
    beat = 60.0 / bpm
    bar = beat * 4

    def env(x, period):
        phase = (x % period) / period
        return 0.3 + 0.6 * (1 - abs(2 * phase - 1))

    for i, tval in enumerate(t):
        bar_index = int((tval // bar) % len(chords))
        freqs = chords[bar_index]
        e = env(tval, bar) * 0.12
        value = 0.0
        for f in freqs:
            value += np.sin(2 * np.pi * f * tval) + 0.4 * np.sin(2 * np.pi * 2 * f * tval)
        music[i] = e * value / len(freqs)

    rng = np.random.default_rng(42)
    accents = np.zeros_like(music)
    accent_count = int(total_seconds // (2 * beat))
    for _ in range(accent_count):
        start = rng.integers(0, len(t) - int(0.4 * sr))
        dur = int(0.25 * sr)
        f = rng.choice([523.25, 659.25, 783.99])  # C5, E5, G5
        a = np.linspace(0, 1, dur)
        bell = 0.08 * np.sin(2 * np.pi * f * np.arange(dur) / sr) * (1 - (a ** 2))
        accents[start:start + dur] += bell[:max(0, len(accents) - start)]

    noise = (rng.normal(0, 1, len(t)) * 0.008).astype(np.float64)

    final = music + accents + noise
    final = final / (np.max(np.abs(final)) + 1e-9) * 0.8

    audio = (final * 32767).astype(np.int16).tobytes()
    out = AudioSegment(audio, frame_rate=sr, sample_width=2, channels=1)
    out.export(out_wav, format="wav")

def build_manim_file(durations, captions):
    print("Composing Manim drawing script…")
    manim_py = BUILD / "manim_scenes.py"
    code = f'''from manim import *
config.pixel_width = 1080
config.pixel_height = 1080
config.background_color = WHITE
TITLE_FONT = "DejaVu Sans"
TEXT_FONT = "DejaVu Sans"

SCENE_DURS = {json.dumps(durations)}
CAPTIONS = {json.dumps(captions)}

def draw_teacher():
    # Friendly African female teacher (smart-casual blouse, short natural hair, glasses, warm smile)
    head_center = LEFT*4 + UP*1
    head = Circle(radius=0.8, color=BLACK, stroke_width=4).move_to(head_center)
    hairs = VGroup(*[
        Arc(radius=0.85, start_angle=PI, angle=0.6, color=BLACK, stroke_width=4).move_to(head_center + RIGHT*(i*0.2-0.6) + UP*0.5)
        for i in range(7)
    ])
    g1 = Circle(radius=0.22, color=BLACK, stroke_width=4).move_to(head_center + LEFT*0.28 + UP*0.1)
    g2 = Circle(radius=0.22, color=BLACK, stroke_width=4).move_to(head_center + RIGHT*0.28 + UP*0.1)
    bridge = Line(g1.get_center()+RIGHT*0.22, g2.get_center()+LEFT*0.22, stroke_width=4, color=BLACK)
    e1 = Dot(g1.get_center(), color=BLACK, radius=0.04)
    e2 = Dot(g2.get_center(), color=BLACK, radius=0.04)
    smile = Arc(radius=0.28, start_angle=-PI/4, angle=PI/2, color=BLACK, stroke_width=4).move_to(head_center+DOWN*0.2)

    torso = RoundedRectangle(corner_radius=0.15, width=1.9, height=2.4, stroke_width=4, color=BLACK).move_to(LEFT*4 + DOWN*0.8)
    vneck = VGroup(Line(LEFT*4+UP*0.1, LEFT*4+DOWN*0.35),
                   Line(LEFT*4+DOWN*0.35, LEFT*4+RIGHT*0.25)).set_color(BLACK).set_stroke(width=4)

    arm_left = Line(LEFT*4 + DOWN*0.2 + LEFT*0.95, LEFT*4 + DOWN*0.2 + LEFT*2, stroke_width=4, color=BLACK)
    arm_right = Line(LEFT*3.05 + DOWN*0.2, LEFT*2 + DOWN*0.2, stroke_width=4, color=BLACK)

    leg1 = Line(LEFT*4 + DOWN*2, LEFT*4 + DOWN*3, stroke_width=4, color=BLACK)
    leg2 = Line(LEFT*3.5 + DOWN*2, LEFT*3.5 + DOWN*3, stroke_width=4, color=BLACK)

    teacher = VGroup(head, hairs, g1, g2, bridge, e1, e2, smile, torso, vneck, arm_left, arm_right, leg1, leg2)
    teacher.set_z_index(2)
    return teacher

def whiteboard():
    board = RoundedRectangle(corner_radius=0.1, width=7.6, height=4.5, color=BLACK, stroke_width=4)
    board.move_to(RIGHT*1.5 + UP*0.3)
    board.set_z_index(1)
    return board

def caption_box(text):
    cap = Text(text, font=TEXT_FONT, weight="MEDIUM", color=BLACK).scale(0.5)
    cap.to_edge(DOWN).shift(DOWN*0.2)
    cap.set_z_index(3)
    return cap

def equation_text(s):
    t = Text(s, font=TEXT_FONT, color=BLACK).scale(0.9)
    t.move_to(RIGHT*1.3 + UP*0.5)
    t.set_z_index(3)
    return t

def small_text(s):
    t = Text(s, font=TEXT_FONT, color=BLACK).scale(0.6)
    t.set_z_index(3)
    return t

class Scene1_TitleIntro(Scene):
    def construct(self):
        dur = SCENE_DURS[0]
        teacher = draw_teacher()
        board = whiteboard()
        eq = equation_text("9 + 2 × 18 – 2")
        cap = caption_box(CAPTIONS[0])
        self.play(Create(teacher, run_time=2.5))
        self.play(Create(board, run_time=1.2))
        self.play(Write(eq, run_time=2.2))
        self.play(Write(cap, run_time=1.0))
        self.wait(max(0, dur - (2.5+1.2+2.2+1.0)))

class Scene2_ReadQuestion(Scene):
    def construct(self):
        dur = SCENE_DURS[1]
        teacher = draw_teacher()
        board = whiteboard()
        eq = equation_text("9 + 2 × 18 – 2")
        cap = caption_box(CAPTIONS[1])
        self.play(Create(teacher, run_time=1.0))
        self.play(Create(board, run_time=0.8))
        self.play(Write(eq, run_time=1.0))
        highlight = Rectangle(width=eq.width+0.3, height=eq.height+0.15, fill_color=YELLOW, fill_opacity=0.35, stroke_opacity=0)
        highlight.move_to(eq.get_center())
        highlight.set_z_index(2)
        self.play(FadeIn(highlight, run_time=1.0))
        self.play(Write(cap, run_time=0.8))
        self.wait(max(0, dur - (1.0+0.8+1.0+1.0+0.8)))

class Scene3_BODMAS(Scene):
    def construct(self):
        dur = SCENE_DURS[2]
        teacher = draw_teacher()
        board = whiteboard()
        chart = RoundedRectangle(corner_radius=0.1, width=4.5, height=3.8, color=BLACK, stroke_width=4).move_to(RIGHT*2 + UP*0.2)
        title = small_text("BODMAS").move_to(chart.get_top()+DOWN*0.4)
        items = VGroup(
            small_text("B — Brackets"),
            small_text("O — Orders"),
            small_text("D — Division"),
            small_text("M — Multiplication"),
            small_text("A — Addition"),
            small_text("S — Subtraction"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to(chart.get_center()+DOWN*0.2+LEFT*0.7).scale(0.9)
        cap = caption_box(CAPTIONS[2])
        self.play(Create(teacher, run_time=1.0))
        self.play(Create(board, run_time=0.6))
        self.play(Create(chart, run_time=0.8))
        self.play(Write(title, run_time=0.6))
        for it in items:
            self.play(Write(it, run_time=0.6))
        self.play(Write(cap, run_time=0.8))
        total_rt = 1.0+0.6+0.8+0.6+6*0.6+0.8
        self.wait(max(0, dur - total_rt))

class Scene4_MultiplyFirst(Scene):
    def construct(self):
        dur = SCENE_DURS[3]
        teacher = draw_teacher()
        board = whiteboard()
        eq = equation_text("9 + 2 × 18 – 2")
        cap = caption_box(CAPTIONS[3])
        circ_target = Text("2 × 18", font=TEXT_FONT, color=BLACK).scale(0.9).move_to(RIGHT*2.1 + UP*0.5)
        circ = SurroundingRectangle(circ_target, color=RED, buff=0.2, stroke_width=6)
        thought = VGroup(
            Circle(radius=0.9, color=BLACK, stroke_width=3),
            Circle(radius=0.15, color=BLACK, stroke_width=3).shift(DOWN*0.9+LEFT*0.6),
            Circle(radius=0.09, color=BLACK, stroke_width=3).shift(DOWN*1.3+LEFT*0.9),
        )
        thought.move_to(RIGHT*3.1 + UP*1.7)
        calc = small_text("2 × 18 = 36").move_to(thought[0].get_center())
        self.play(Create(teacher, run_time=1.0))
        self.play(Create(board, run_time=0.6))
        self.play(Write(eq, run_time=1.2))
        self.play(Create(circ, run_time=0.8))
        self.play(Create(thought, run_time=0.8))
        self.play(Write(calc, run_time=0.8))
        self.play(Write(cap, run_time=0.8))
        total_rt = 1.0+0.6+1.2+0.8+0.8+0.8+0.8
        self.wait(max(0, dur - total_rt))

class Scene5_Simplify(Scene):
    def construct(self):
        dur = SCENE_DURS[4]
        teacher = draw_teacher()
        board = whiteboard()
        eq1 = equation_text("9 + 2 × 18 – 2")
        eq2 = equation_text("9 + 36 – 2")
        cap = caption_box(CAPTIONS[4])
        self.play(Create(teacher, run_time=0.8))
        self.play(Create(board, run_time=0.6))
        self.play(Write(eq1, run_time=1.0))
        self.play(Transform(eq1, eq2, run_time=1.2))
        self.play(Write(cap, run_time=0.8))
        total_rt = 0.8+0.6+1.0+1.2+0.8
        self.wait(max(0, dur - total_rt))

class Scene6_AddSubtract(Scene):
    def construct(self):
        dur = SCENE_DURS[5]
        teacher = draw_teacher()
        board = whiteboard()
        eq = equation_text("9 + 36 – 2")
        step1 = small_text("9 + 36 = 45").move_to(RIGHT*1.3 + DOWN*0.2)
        step2 = small_text("45 – 2 = 43").move_to(RIGHT*1.3 + DOWN*0.8)
        arrow1 = CurvedArrow(RIGHT*0.0 + UP*0.4, RIGHT*0.9 + DOWN*0.2, color=BLACK, stroke_width=4)
        arrow2 = CurvedArrow(RIGHT*1.2 + DOWN*0.2, RIGHT*1.4 + DOWN*0.8, color=BLACK, stroke_width=4)
        cap = caption_box(CAPTIONS[5])
        self.play(Create(teacher, run_time=0.8))
        self.play(Create(board, run_time=0.6))
        self.play(Write(eq, run_time=0.8))
        self.play(Create(arrow1, run_time=0.6))
        self.play(Write(step1, run_time=1.0))
        self.play(Create(arrow2, run_time=0.6))
        self.play(Write(step2, run_time=1.0))
        forty_three = Text("43", font=TEXT_FONT, color=BLACK).scale(1.2).move_to(RIGHT*2.3 + DOWN*0.8)
        self.play(Transform(step2, forty_three, run_time=0.6))
        self.play(Write(cap, run_time=0.8))
        total_rt = 0.8+0.6+0.8+0.6+1.0+0.6+1.0+0.6+0.8
        self.wait(max(0, dur - total_rt))

class Scene7_FinalAnswer(Scene):
    def construct(self):
        dur = SCENE_DURS[6]
        teacher = draw_teacher()
        teacher_center = teacher.copy().move_to(ORIGIN + UP*0.2 + LEFT*0.2)
        answer_board = RoundedRectangle(corner_radius=0.1, width=3.5, height=2.0, color=BLACK, stroke_width=4).move_to(UP*0.8 + RIGHT*0.5)
        answer = Text("43", font=TEXT_FONT, color=BLACK).scale(1.6).move_to(answer_board.get_center())
        cap = caption_box(CAPTIONS[6])

        confetti = VGroup(*[
            Square(side_length=0.08, color=color, fill_color=color, fill_opacity=1.0).move_to(np.array([
                np.random.uniform(-5,5), np.random.uniform(0.5,3.5), 0
            ]))
            for color in [RED, BLUE, GREEN, YELLOW, PURPLE, ORANGE] for _ in range(15)
        ])
        students = VGroup(*[
            VGroup(
                Circle(radius=0.18, color=BLACK, stroke_width=3),
                Arc(radius=0.12, start_angle=0, angle=PI, color=BLACK, stroke_width=3).shift(DOWN*0.05),
                Dot(radius=0.02).shift(UP*0.04+LEFT*0.05),
                Dot(radius=0.02).shift(UP*0.04+RIGHT*0.05),
            ).move_to(np.array([x, -3.5 + np.random.uniform(0,0.2), 0]))
            for x in np.linspace(-5, 5, 8)
        ]).shift(DOWN*0.2)

        self.play(Create(teacher_center, run_time=1.2))
        self.play(Create(answer_board, run_time=0.8))
        self.play(Write(answer, run_time=0.8))
        self.add(confetti)
        self.play(*[confetti[i].animate.shift(DOWN*3.8) for i in range(len(confetti))], run_time=1.2, rate_func=linear)
        self.play(Create(students, run_time=0.8))
        self.play(Write(cap, run_time=0.8))
        total_rt = 1.2+0.8+0.8+1.2+0.8+0.8
        self.wait(max(0, dur - total_rt))
'''
    manim_py.write_text(code, encoding="utf-8")
    return manim_py

def run_manim(scene_file, scene_class, out_name):
    print(f"Rendering {scene_class}…")
    cmd = [
        sys.executable, "-m", "manim",
        "-qk", "-r", "1080,1080", str(scene_file), scene_class,
        "-o", out_name
    ]
    subprocess.run(cmd, check=True)

def concat_videos(video_paths, out_path):
    clips = [VideoFileClip(str(p)) for p in video_paths]
    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(str(out_path), fps=30, codec="libx264", audio=False, threads=4)
    for c in clips:
        c.close()

def merge_intervals(intervals):
    if not intervals:
        return []
    xs = sorted(intervals)
    merged = [list(xs[0])]
    for s, e in xs[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(a, b) for a, b in merged]

def scene_durations_sum(durs):
    return sum(durs)

def mix_audio(vo_durations, scene_durations, out_wav):
    segments = []
    vo_presence = []
    cursor_ms = 0
    for i, sd in enumerate(scene_durations):
        vo_file = ASSETS / f"vo_{i+1}.mp3"
        vo_seg = AudioSegment.from_file(vo_file).set_frame_rate(SAMPLE_RATE)
        segments.append((vo_seg, cursor_ms))
        vo_presence.append((cursor_ms, cursor_ms + len(vo_seg)))
        cursor_ms += int(sd * 1000)

    total_len_ms = int(scene_durations_sum(scene_durations) * 1000)
    master_vo = AudioSegment.silent(duration=total_len_ms, frame_rate=SAMPLE_RATE)
    for seg, pos in segments:
        master_vo = master_vo.overlay(seg, position=pos)

    music_wav = ASSETS / "music.wav"
    synth_soft_classroom_music(total_len_ms / 1000.0 + 1.0, music_wav)
    music = AudioSegment.from_file(music_wav).set_frame_rate(SAMPLE_RATE)
    music = music[:total_len_ms]

    ducked = AudioSegment.silent(duration=0, frame_rate=SAMPLE_RATE)
    p = 0
    duck_regions = merge_intervals(vo_presence)
    for start, end in duck_regions:
        if start > p:
            pre = music[p:start].apply_gain(-20)
            ducked += pre
        region = music[start:end].apply_gain(-28).fade_in(100).fade_out(120)
        ducked += region
        p = end
    if p < len(music):
        ducked += music[p:].apply_gain(-20)

    master = ducked.overlay(master_vo)
    master = effects.normalize(master, headroom=1.0)
    master.export(out_wav, format="wav")

def main():
    ensure_ffmpeg()
    vo_durations = tts_generate_all()
    scene_durations = []
    for sc, vo_dur in zip(SCENES, vo_durations):
        scene_durations.append(max(sc["min_duration"], vo_dur + 1.0))

    manim_file = build_manim_file(scene_durations, [s["caption"] for s in SCENES])

    out_videos = []
    for i, sc in enumerate(SCENES, start=1):
        out_name = f"s{i:02d}_{sc['name']}"
        run_manim(manim_file, sc["name"], out_name)
        found = list(Path(".").rglob(out_name + ".mp4"))
        if not found:
            found = list(Path("media").rglob(out_name + ".mp4"))
        if not found:
            raise FileNotFoundError(f"Could not find rendered video for {out_name}")
        out_videos.append(found[0])

    silent_video = BUILD / "video_silent.mp4"
    concat_videos(out_videos, silent_video)

    audio_master = BUILD / "audio_master.wav"
    mix_audio(vo_durations, scene_durations, audio_master)

    print("Muxing final video with audio…")
    v = VideoFileClip(str(silent_video))
    a = AudioFileClip(str(audio_master))
    v = v.set_audio(a)
    final_out = BUILD / "final_doodle_1080x1080.mp4"
    v.write_videofile(str(final_out), fps=30, codec="libx264", audio_codec="aac", bitrate="6000k", threads=4)
    v.close()
    a.close()

    print("Writing SRT captions…")
    srt_path = BUILD / "captions.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        t_acc = 0.0
        for idx, (sc, sd) in enumerate(zip(SCENES, scene_durations), start=1):
            start = t_acc
            end = t_acc + sd
            f.write(f"{idx}\n")
            f.write(f"{format_ts(start)} --> {format_ts(end)}\n")
            f.write(sc["vo"] + "\n\n")
            t_acc = end

    print("\nDone!")
    print(f"- Video: {final_out}")
    print(f"- Audio: {audio_master}")
    print(f"- Captions: {srt_path}")

def format_ts(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

if __name__ == "__main__":
    main()
