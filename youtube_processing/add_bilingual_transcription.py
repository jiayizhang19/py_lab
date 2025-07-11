import whisper
import ffmpeg
from googletrans import Translator
import os
import subprocess
from pathlib import Path

video_path = "01_Morning_Routine.mp4"


def process_subtitles(video_path):
    """Full subtitle pipeline for a single video."""
    # Step 1: Extract audio
    audio_path = extract_audio(video_path)
    
    # Step 2: Transcribe English
    en_srt = transcribe_english(audio_path)
    
    # Step 3: Translate to Chinese
    zh_srt = translate_to_chinese(en_srt)
    
    # Step 4: Merge subtitles (English + Chinese)
    dual_srt = merge_subtitles(en_srt, zh_srt)
    
    # Step 5: Burn subtitles into video
    burn_subtitles(video_path, dual_srt)

def extract_audio(video_path, output_audio="temp_audio.wav"):
    """Extract audio using FFmpeg."""
    ffmpeg.input(video_path).output(output_audio, acodec="pcm_s16le", ar="16000").run(overwrite_output=True)
    return output_audio

def transcribe_english(audio_path):
    """Transcribe audio to English .srt using Whisper."""
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, task="transcribe")
    srt_path = audio_path.replace(".wav", "_en.srt")
    with open(srt_path, "w") as f:
        for i, segment in enumerate(result["segments"]):
            f.write(f"{i+1}\n")
            f.write(f"{segment['start']:.3f} --> {segment['end']:.3f}\n")
            f.write(f"{segment['text'].strip()}\n\n")
    return srt_path

def translate_to_chinese(srt_path):
    """Translate English .srt to Chinese."""
    translator = Translator()
    zh_srt = srt_path.replace("_en.srt", "_zh.srt")
    with open(srt_path, "r") as f:
        lines = f.readlines()
    
    with open(zh_srt, "w") as f:
        for line in lines:
            if line.strip() and not (line.startswith(("0", "1", "2")) or "-->" in line):
                translated = translator.translate(line.strip(), src="en", dest="zh-cn").text
                f.write(translated + "\n")
            else:
                f.write(line)
    return zh_srt

def merge_subtitles(en_srt, zh_srt):
    """Merge English + Chinese into one .srt file."""
    dual_srt = en_srt.replace("_en.srt", "_dual.srt")
    with open(en_srt, "r") as f_en, open(zh_srt, "r") as f_zh, open(dual_srt, "w") as f_out:
        en_lines = f_en.readlines()
        zh_lines = f_zh.readlines()
        for i in range(len(en_lines)):
            if en_lines[i].strip() and not (en_lines[i].startswith(("0", "1", "2")) or "-->" in en_lines[i]):
                f_out.write(en_lines[i])  # English
                f_out.write(zh_lines[i])   # Chinese
            else:
                f_out.write(en_lines[i])
    return dual_srt

def burn_subtitles(video_path, subtitle_path):
    """Bulletproof subtitle burning with proper Windows path handling"""
    # Convert to absolute paths
    video_path = os.path.abspath(video_path)
    subtitle_path = os.path.abspath(subtitle_path)
    
    # Prepare subtitle path for FFmpeg (works on all platforms)
    # Step 1: Normalize path separators
    sub_path = subtitle_path.replace('\\', '/')
    # Step 2: Escape special characters
    sub_path = sub_path.replace(':', '\\:')
    # Step 3: Wrap in single quotes
    sub_path = f"'{sub_path}'"
    
    output_path = video_path.replace('.mp4', '_subtitled.mp4')
    
    cmd = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-vf', f'subtitles={sub_path}',
        '-c:a', 'copy',
        output_path
    ]
    
    # Print the command for debugging
    print('Running:', ' '.join(cmd))
    
    try:
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE, text=True)
        os.replace(output_path, video_path)
        return True
    except subprocess.CalledProcessError as e:
        print('FFmpeg failed:')
        print(e.stderr)
        return False


process_subtitles(video_path)
