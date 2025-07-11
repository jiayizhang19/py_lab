import yt_dlp
import subprocess
from datetime import datetime
import os
from add_bilingual_transcription import *

url = "https://www.youtube.com/watch?v=J1f47XwvQYA&t=311s"


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(os.getcwd(), "video", f"{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Download full video
    video_path = download_video(url, output_dir)

    chapters = get_chapters(url)
    if not chapters:
        print("❌ No chapters found.")
        return
    print(f"📁 Saving video and clips to: {output_dir}")

    # Split into chapters
    split_chapters(video_path, chapters, output_dir)
    print("✅ Done splitting!")

    # Add bilingual transcription to each chapter
    for file in os.listdir(output_dir):
        if file.endswith(".mp4") and file != "full_video.mp4":
            chapter_path = os.path.join(output_dir, file)
            process_subtitles(chapter_path)

    print("✅ Done! Subtitles added to all chapters.")


def download_video(url, output_dir):
    video_path = os.path.join(output_dir, "full_video.mp4")
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
        'merge_output_format': 'mp4',
        'outtmpl': video_path, 
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return video_path


def get_chapters(url):
    with yt_dlp.YoutubeDL({}) as ydl:
        info_dict = ydl.extract_info(url, download=False) 
        return info_dict.get("chapters", [])


def split_chapters(video_file, chapters, output_dir):
    for i, chapter in enumerate(chapters, start=1):
        start = format_time(chapter["start_time"])
        end = format_time(chapter["end_time"])
        title = chapter["title"].replace(" ", "_").replace("'", "").replace("?", "").replace("/", "-")
        output_file = os.path.join(output_dir, f"{i:02d}_{title}.mp4")
        
        print(f"▶ Splitting: {output_file}")
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_file,
            "-ss", start,
            "-to", end,
            "-c", "copy",
            output_file
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def format_time(seconds):
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"


if __name__ == "__main__":
    main()

