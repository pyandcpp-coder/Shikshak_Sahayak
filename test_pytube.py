import subprocess

def download_audio_ytdlp(url, output_path):
    command = [
        'yt-dlp',
        '-x',
        '--audio-format', 'mp3',
        '-o', f'{output_path}/%(title)s.%(ext)s',
        url
    ]
    subprocess.run(command, check=True)

download_audio_ytdlp("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "audio_output")