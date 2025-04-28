import json
import os
import re
import shutil
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Thread

import requests
import yt_dlp
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL


def extract_files(input_path):
    # Define the pattern to match the file names
    pattern = re.compile(r"(.*)_(\d+)_video_part_(\d+)\.(mp4|srt|vtt)")
    # Dictionary to store grouped files
    grouped_files = {}

    # Scan directory for matching files
    for root, _, files in os.walk(input_path):
        for file in files:
            match = pattern.match(file)
            if match:
                name, duration, part, extension = match.groups()
                key = f"{name}_{duration}_video_part_{part}"

                if key not in grouped_files:
                    grouped_files[key] = {
                        "path": os.path.join(root, file),
                        "name": name,
                        "duration": int(duration),
                        "full_name": key
                    }

    # Sort files by name and duration
    sorted_files = sorted(grouped_files.values(), key=lambda x: (x['name'], x['duration']))

    # Convert to JSON
    output_data = json.dumps(sorted_files, indent=4)

    # Save JSON to file
    output_file = os.path.join(input_path, "output.json")
    with open(output_file, "w") as f:
        f.write(output_data)

    return sorted_files


def download_subtitle(url, output_path):
    """
    Download YouTube video subtitles and save in SRT format.

    Args:
        url (str): YouTube video URL
        output_path (str): Path where to save the SRT file

    Returns:
        str: Path to the saved SRT file
    """
    # Extract video ID from URL
    try:
        if "youtu.be" in url:
            video_id = url.split("/")[-1]
        else:
            from urllib.parse import parse_qs, urlparse
            parsed_url = urlparse(url)
            video_id = parse_qs(parsed_url.query)['v'][0]
    except Exception as e:
        raise ValueError(f"Could not extract video ID from URL: {str(e)}")

    try:
        # Get transcript from YouTube
        srt = YouTubeTranscriptApi.get_transcript(video_id, preserve_formatting=True)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Convert transcript to SRT format and save
        with open(output_path, 'w', encoding='utf-8') as srt_file:
            for index, subtitle in enumerate(srt, start=1):
                # Calculate start and end times in HH:MM:SS,mmm format
                start_time = format_time(subtitle['start'])
                end_time = format_time(subtitle['start'] + subtitle['duration'])

                # Write subtitle block in SRT format
                srt_file.write(f"{index}\n")
                srt_file.write(f"{start_time} --> {end_time}\n")
                srt_file.write(f"{subtitle['text']}\n\n")

        return output_path

    except Exception as e:
        raise Exception(f"Error downloading subtitles: {str(e)}")


def format_time_srt(seconds_float):
    """
    Convert seconds to SRT time format (HH:MM:SS,mmm) with proper handling of milliseconds.
    """
    hours = int(seconds_float // 3600)
    minutes = int((seconds_float % 3600) // 60)
    seconds = int(seconds_float % 60)
    milliseconds = int((seconds_float - int(seconds_float)) * 1000)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def format_time(seconds):
    """
    Convert seconds to SRT time format (HH:MM:SS,mmm)
    """
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def find_sentence_boundary(words, word_limit):
    """
    Find the nearest sentence boundary (period) within or before the word limit.
    Returns the index where to split and whether a valid boundary was found.
    """
    extended_limit = min(len(words), int(word_limit * 1.2))

    for i in range(min(extended_limit, len(words)) - 1, -1, -1):
        word = words[i]
        if word.endswith('.') or word.endswith('?') or word.endswith('!'):
            return i + 1, True

    if len(words) > word_limit:
        return word_limit, False

    return len(words), False


def parse_timestamp(timestamp):
    """Convert subtitle timestamp to seconds."""
    h, m, s = re.match(r"(\d+):(\d+):([\d.,]+)", timestamp).groups()
    return int(h) * 3600 + int(m) * 60 + float(s.replace(",", "."))


def embed_subtitles(video_file: str, subtitle_file: str):
    """
    Embeds subtitles into a video file using FFmpeg and overwrites the original file safely.
    Added better error handling and subtitle encoding.
    """
    if not os.path.exists(subtitle_file) or os.path.getsize(subtitle_file) == 0:
        print(f"Warning: Subtitle file {subtitle_file} is empty or missing. Skipping embedding.")
        return

    temp_output = f"{video_file}.temp.mp4"  # Temporary output file

    command = [
        'ffmpeg',
        '-i', video_file,  # Input video file
        '-i', subtitle_file,  # Subtitle file (e.g., .srt)
        '-c:v', 'copy',  # Copy video codec
        '-c:a', 'copy',  # Copy audio codec
        '-c:s', 'mov_text',  # Set subtitle codec for MP4
        '-metadata:s:s:0', 'language=eng',  # Set subtitle language
        '-y',  # Overwrite without prompt
        temp_output  # Temporary output file
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Replace original file with the new one
        shutil.move(temp_output, video_file)
        print(f"Subtitles embedded successfully: {video_file}")

    except subprocess.CalledProcessError as e:
        error_output = e.stderr.decode().strip() if e.stderr else "Unknown error"
        print(f"Error embedding subtitles: {error_output}")
        # Cleanup temp file if failed
        if os.path.exists(temp_output):
            os.remove(temp_output)
        raise


def batch_process_videos(video_urls, output_dir, words_per_segment=120,
                         cut_start=None, cut_end=None, max_concurrent=2):
    """
    Process multiple YouTube videos in parallel.

    Args:
        video_urls (list): List of YouTube video URLs
        output_dir (str): Directory to save downloaded files
        words_per_segment (int): Number of words per video segment
        cut_start (str, optional): Timestamp to start from (HH:MM:SS)
        cut_end (str, optional): Timestamp to end at (HH:MM:SS)
        max_concurrent (int): Maximum number of concurrent downloads
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = []
        for url in video_urls:
            future = executor.submit(
                start_thread, url, output_dir, words_per_segment, cut_start, cut_end
            )
            futures.append(future)

        # Wait for all tasks to complete
        for future in futures:
            future.result()

    print(f"All videos processed and saved to: {output_dir}")


def save_video_clip(start_time, end_time, video_file, output_file, cut_start=None, cut_end=None):
    """
    Save a video clip with proper time handling.
    Fixed to handle timestamp conversion consistently.
    """
    try:
        # Convert string timestamps to seconds if needed
        if isinstance(cut_start, str):
            cut_start_time = parse_timestamp(cut_start)
        else:
            cut_start_time = cut_start

        if isinstance(cut_end, str):
            cut_end_time = parse_timestamp(cut_end)
        else:
            cut_end_time = cut_end

        # Adjust start and end times based on optional cut_start and cut_end
        if cut_start_time is not None:
            start_time = max(start_time, cut_start_time)

        if cut_end_time is not None:
            end_time = min(end_time, cut_end_time)

        if start_time >= end_time:
            print(f"Skipping clip due to invalid timing: start={start_time}, end={end_time}")
            return

        # Calculate total duration for percentage calculation
        total_duration = end_time - start_time

        # Initialize the progress bar
        with tqdm(total=total_duration, unit="s", desc="Processing", colour="green") as progress_bar:
            # Build the FFmpeg command
            command = [
                "ffmpeg", "-i", video_file,
                "-ss", str(start_time),
                "-to", str(end_time),
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-c:a", "aac",
                "-strict", "experimental",
                "-progress", "pipe:1",  # Enable progress output
                "-y",  # Add -y to overwrite without asking
                output_file
            ]

            # Run the command and capture its output
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
            )

            for line in process.stdout:
                # Parse progress lines to extract time
                if "out_time=" in line:
                    match = re.search(r"out_time=(\d+):(\d+):(\d+)", line)
                    if match:
                        h, m, s = map(int, match.groups())
                        elapsed = h * 3600 + m * 60 + s
                        progress_bar.n = min(elapsed, total_duration)  # Cap at total duration
                        progress_bar.refresh()

            process.wait()
            if process.returncode == 0:
                print(f"\nVideo clip saved to {output_file}")
            else:
                print(f"\nFFmpeg error with return code {process.returncode}")

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
    except Exception as e:
        print(f"Error saving video clip: {e}")


def parse_timestamp_own(timestamp):
    # Example timestamp parsing function
    h, m, s = map(int, timestamp.split(":"))
    return h * 3600 + m * 60 + s


def adjust_timestamp(original: str, offset: float, is_double_line: bool = False) -> str:
    """
    Adjust the timestamp by subtracting the offset and ensuring it aligns properly with subtitles and audio.
    Properly handles timing adjustments with millisecond precision.
    """
    # Parse hours, minutes, seconds, and milliseconds
    h, m, s = original.split(':')
    seconds, milliseconds = s.split(',')

    # Convert to total seconds for easier manipulation
    total_seconds = int(h) * 3600 + int(m) * 60 + int(seconds) + int(milliseconds) / 1000.0 - offset

    # Ensure total time is never less than 1 second
    total_seconds = max(1, total_seconds)

    # Recalculate hours, minutes, seconds, and milliseconds
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds - int(total_seconds)) * 1000)

    # Format with leading zeros where necessary
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def parse_timestamp_s(timestamp: str) -> float:
    """Parse SRT timestamp format (HH:MM:SS,mmm) to seconds with better precision."""
    parts = timestamp.split(':')
    h, m = float(parts[0]), float(parts[1])
    s_parts = parts[2].split(',')
    s = float(s_parts[0]) + float(s_parts[1]) / 1000
    return h * 3600 + m * 60 + s


def save_section(section_id, blocks, output_dir, word_limit, sub_name):
    """
    Save subtitle section and corresponding video clip with adjusted timestamps.
    Fixed to handle subtitle timing correctly.
    """
    if not blocks:
        return None

    # Use the first subtitle's start time as the offset
    original_start_time = parse_timestamp_s(blocks[0][1].split(" --> ")[0])
    offset = original_start_time

    # Get the overall start and end times for the video segment
    start_time = original_start_time
    end_time = parse_timestamp_s(blocks[-1][1].split(" --> ")[1])

    content = ""
    new_subtitle_id = 1

    # Keep track of previous subtitle end time to prevent overlaps
    prev_end_time = 0

    for index, timestamp, text in blocks:
        start, end = timestamp.split(" --> ")
        is_double_line = len(text.split("\n")) > 1

        # Calculate adjusted timestamps
        adjusted_start_time = parse_timestamp_s(start) - offset
        adjusted_end_time = parse_timestamp_s(end) - offset

        # Ensure no negative times and maintain proper sequence
        adjusted_start_time = max(prev_end_time + 0.01, max(0.1, adjusted_start_time))
        adjusted_end_time = max(adjusted_start_time + 0.5, adjusted_end_time)

        # Format timestamps back to SRT format
        adjusted_start = format_time_srt(adjusted_start_time)
        adjusted_end = format_time_srt(adjusted_end_time)

        # Update for next iteration to prevent overlapping
        prev_end_time = adjusted_end_time

        content += f"{new_subtitle_id}\n{adjusted_start} --> {adjusted_end}\n{text}\n\n"
        new_subtitle_id += 1

    subtitle_output = os.path.join(output_dir, f"{sub_name}_{word_limit}_video_part_{section_id}.srt")
    with open(subtitle_output, "w", encoding="utf-8") as out_file:
        out_file.write(content.strip())

    print(f"Saved subtitles: {subtitle_output}")

    return start_time, end_time, os.path.join(output_dir, f"{sub_name}_{word_limit}_video_part_{section_id}.mp4")


def split_subtitles(input_file, video_file, output_dir, word_limit, sub_name, cut_start=None, cut_end=None):
    """
    Split subtitles and video into segments based on word count.
    Fixed to handle subtitle timing and splits correctly.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the subtitle file
    try:
        with open(input_file, "r", encoding="utf-8") as file:
            subtitles = file.read()
    except UnicodeDecodeError:
        # Try alternate encodings if UTF-8 fails
        encodings = ['latin-1', 'iso-8859-1', 'windows-1252']
        for encoding in encodings:
            try:
                with open(input_file, "r", encoding=encoding) as file:
                    subtitles = file.read()
                break
            except UnicodeDecodeError:
                continue
        else:
            print(f"Error: Could not decode subtitle file {input_file} with any known encoding.")
            return

    subtitle_blocks = re.split(r"\n\n", subtitles.strip())
    parsed_blocks = parse_subtitle_blocks(subtitle_blocks)

    video_tasks = []

    cut_start_seconds = parse_timestamp(cut_start) if cut_start else None
    cut_end_seconds = parse_timestamp(cut_end) if cut_end else None

    with ThreadPoolExecutor(max_workers=2) as executor:
        current_section = []
        accumulated_words = []
        section_id = 1

        for block in parsed_blocks:
            _, timestamp, text = block
            try:
                start_time_sec = parse_timestamp(timestamp.split(" --> ")[0])
            except Exception as e:
                print(f"Warning: Could not parse timestamp '{timestamp}': {e}")
                continue

            end_time_sec = parse_timestamp(timestamp.split(" --> ")[1])

            # Skip if outside cut range
            if (cut_start_seconds and end_time_sec < cut_start_seconds) or \
                    (cut_end_seconds and start_time_sec > cut_end_seconds):
                continue

            # Apply cut range adjustments if needed
            if cut_start_seconds and start_time_sec < cut_start_seconds:
                continue
            if cut_end_seconds and end_time_sec > cut_end_seconds:
                continue

            words = text.split()
            accumulated_words.extend(words)
            current_section.append(block)

            # Check if we have enough words to consider splitting
            if len(accumulated_words) >= word_limit:
                split_index, found_boundary = find_sentence_boundary(accumulated_words, word_limit)

                if split_index > 0:
                    blocks_to_keep, remaining_blocks = process_current_section(
                        current_section, accumulated_words, split_index
                    )

                    result = save_section(section_id, blocks_to_keep, output_dir, word_limit, sub_name)
                    if result:
                        start_time, end_time, video_output = result
                        video_tasks.append(
                            executor.submit(save_video_clip, start_time, end_time, video_file, video_output,
                                            cut_start_seconds, cut_end_seconds)
                        )

                    section_id += 1
                    current_section = remaining_blocks
                    accumulated_words = accumulated_words[split_index:]

        # Save any remaining content
        if current_section:
            result = save_section(section_id, current_section, output_dir, word_limit, sub_name)
            if result:
                start_time, end_time, video_output = result
                video_tasks.append(
                    executor.submit(save_video_clip, start_time, end_time, video_file, video_output,
                                    cut_start, cut_end)
                )

        # Wait for all video tasks to complete
        for task in video_tasks:
            try:
                task.result()
            except Exception as e:
                print(f"Error in video processing task: {e}")


def parse_subtitle_blocks(subtitle_blocks):
    parsed_blocks = []
    for block in subtitle_blocks:
        lines = block.split("\n")
        if len(lines) >= 3:
            subtitle_id = lines[0].strip()
            timestamp = lines[1].strip()
            text = " ".join(lines[2:]).strip()
            parsed_blocks.append((subtitle_id, timestamp, text))
    return parsed_blocks


def process_current_section(current_section, accumulated_words, split_index):
    blocks_to_keep = []
    remaining_blocks = []
    current_block_words = []

    for b in current_section:
        block_words = b[2].split()
        current_block_words.extend(block_words)

        if len(current_block_words) <= split_index:
            blocks_to_keep.append(b)
        else:
            remaining_words = len(current_block_words) - split_index
            if remaining_words > 0:
                new_text = " ".join(block_words[-remaining_words:])
                remaining_blocks.append((b[0], b[1], new_text))

            modified_text = " ".join(block_words[:-remaining_words])
            if modified_text:
                blocks_to_keep.append((b[0], b[1], modified_text))

    return blocks_to_keep, remaining_blocks


def to_snake_case(text):
    # Replace spaces with underscores, remove special characters, and convert to lowercase
    return ''.join([('_' if ch.isspace() else ch.lower()) for ch in text if ch.isalnum() or ch.isspace()])


def create_session_with_retries():
    """Create requests session with retry strategy"""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504]
    )
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session


def download_chunk(url, start_byte, end_byte, chunk_path, progress_dict, chunk_id):
    """Download a chunk with progress tracking and retry logic."""
    session = create_session_with_retries()
    headers = {'Range': f'bytes={start_byte}-{end_byte}'}
    progress_dict[chunk_id] = 0

    max_retries = 3
    retry_count = 0
    chunk_size = 1024 * 1024  # 1MB

    while retry_count < max_retries:
        try:
            response = session.get(url, headers=headers, stream=True)
            response.raise_for_status()

            with open(chunk_path, 'wb') as chunk_file:
                for data in response.iter_content(chunk_size):
                    if not data:
                        break
                    chunk_file.write(data)
                    progress_dict[chunk_id] += len(data)
            return True

        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError) as e:
            retry_count += 1
            if retry_count == max_retries:
                raise Exception(f"Failed to download chunk after {max_retries} retries: {str(e)}")

            # Calculate remaining bytes
            if os.path.exists(chunk_path):
                current_size = os.path.getsize(chunk_path)
                new_start_byte = start_byte + current_size
                headers = {'Range': f'bytes={new_start_byte}-{end_byte}'}

            threading.Event().wait(1)  # Wait before retry


def merge_chunks(output_path, chunk_paths):
    """Merge chunks into single file with validation."""
    try:
        with open(output_path, 'wb') as output_file:
            for chunk_path in chunk_paths:
                if os.path.exists(chunk_path):
                    with open(chunk_path, 'rb') as chunk_file:
                        output_file.write(chunk_file.read())
                    os.remove(chunk_path)
    except Exception as e:
        # Cleanup any partially written output
        if os.path.exists(output_path):
            os.remove(output_path)
        raise Exception(f"Failed to merge chunks: {str(e)}")


def calculate_optimal_threads(file_size):
    """Calculate optimal thread count based on file size."""
    MIN_CHUNK_SIZE = 1024 * 10 * 10
    MAX_THREADS = 40
    optimal_threads = min(file_size // MIN_CHUNK_SIZE, MAX_THREADS)
    return max(1, int(optimal_threads))


def download_media(url, output_path, progress_dict, media_type="video"):
    """Download video or audio with progress tracking."""
    response = requests.head(url)
    file_size = int(response.headers.get('content-length', 0))
    num_threads = calculate_optimal_threads(file_size)
    chunk_size = file_size // num_threads

    chunk_paths = []
    threads = []

    for i in range(num_threads):
        start_byte = i * chunk_size
        end_byte = file_size - 1 if i == num_threads - 1 else (start_byte + chunk_size - 1)
        chunk_path = f"{output_path}_{media_type}_chunk_{i}.tmp"
        chunk_paths.append(chunk_path)

        thread = threading.Thread(
            target=download_chunk,
            args=(url, start_byte, end_byte, chunk_path, progress_dict, f"{media_type}_{i}")
        )
        threads.append(thread)
        thread.start()

    return threads, chunk_paths, file_size


def download_from_youtube(url, download_path):
    """
    Enhanced YouTube downloader that downloads high-quality video and audio,
    merges them efficiently, and handles subtitles properly.

    Args:
        url (str): YouTube video URL
        download_path (str): Path to store downloaded files

    Returns:
        tuple: (video_path, subtitle_path, video_title, snake_case_title)
    """
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    # Generate a unique snake_case filename for fallback
    temp_id = str(uuid.uuid4()).replace("-", "_")
    video_title = temp_id
    snake_case_title = temp_id
    video_filename = os.path.join(download_path, f"{temp_id}.mp4")
    subtitle_filename = os.path.join(download_path, f"{temp_id}.en.srt")

    # Step 1: Try to get video info first without downloading
    try:
        ydl_info = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
            'forcejson': True,
            'ignoreerrors': True,  # Continue despite errors
        }

        with YoutubeDL(ydl_info) as ydl:
            info_dict = ydl.extract_info(url, download=False)

            if info_dict:
                video_title = info_dict.get('title', temp_id)
                snake_case_title = to_snake_case(video_title)

                # Define file paths
                video_filename = os.path.join(download_path, f"{snake_case_title}.mp4")
                subtitle_filename = os.path.join(download_path, f"{snake_case_title}.en.srt")

                # Get available formats
                available_formats = info_dict.get('formats', [])

                # Skip if already downloaded
                if os.path.exists(video_filename) and os.path.getsize(video_filename) > 0:
                    print(f"Video already exists: {video_filename}")
                    if os.path.exists(subtitle_filename) and os.path.getsize(subtitle_filename) > 0:
                        print(f"Subtitles already exist: {subtitle_filename}")
                    else:
                        try:
                            download_subtitle(url, subtitle_filename)
                            print(f"Downloaded subtitles separately: {subtitle_filename}")
                        except Exception as sub_e:
                            print(f"Subtitle download failed: {str(sub_e)}")

                    return video_filename, subtitle_filename, video_title, snake_case_title
    except Exception as e:
        print(f"Warning: Error extracting video info: {str(e)}. Using fallback method.")
        # Continue with fallback values set earlier

    # Step 2: Prepare download options with multiple format fallbacks
    format_options = [
        'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',  # First try best quality
        'best[ext=mp4]/best',  # Then try any mp4
        'best'  # Finally try any format
    ]

    subtitle_options = [
        {'format': 'srt', 'language': 'en'},
        {'format': 'vtt', 'language': 'en'},
        {'format': 'ass', 'language': 'en'},
        {'format': 'srt', 'language': 'en-US'},
        {'format': 'vtt', 'language': 'en-US'}
    ]

    # Try each format option until one works
    for format_choice in format_options:
        try:
            print(f"Trying format: {format_choice}")

            ydl_opts = {
                'format': format_choice,
                'outtmpl': video_filename,
                'subtitleslangs': ['en', 'en-US'],  # Try both English variants
                'writesubtitles': True,
                'writeautomaticsub': True,
                'ignoreerrors': True,  # Continue despite errors
                'nooverwrites': False,  # Allow overwriting
                'noplaylist': True,  # Download single video only
                'postprocessors': [
                    {
                        'key': 'FFmpegVideoConvertor',
                        'preferedformat': 'mp4',
                    }
                ],
                'concurrent_fragment_downloads': 8,  # Multiple fragment downloads
                'retries': 15,  # More retries for better reliability
                'file_access_retries': 10,
                'fragment_retries': 15,
                'skip_unavailable_fragments': True,  # Skip unavailable fragments
                'keepvideo': True,  # Keep video file after post-processing
                'verbose': False,
            }

            with tqdm(desc=f"Downloading video ({format_choice})", unit="B", unit_scale=True,
                      miniters=1) as progress_bar:
                def progress_hook(d):
                    if d['status'] == 'downloading':
                        if 'total_bytes' in d and 'downloaded_bytes' in d:
                            progress_bar.total = d['total_bytes']
                            progress_bar.n = d['downloaded_bytes']
                            progress_bar.refresh()

                ydl_opts['progress_hooks'] = [progress_hook]
                with YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])

            # If we get here without exception, the download succeeded
            if os.path.exists(video_filename) and os.path.getsize(video_filename) > 0:
                print(f"Successfully downloaded video with format: {format_choice}")
                break
            else:
                print(f"Download appeared to succeed but file is missing or empty. Trying next format.")
        except Exception as e:
            print(f"Error with format {format_choice}: {str(e)}")
            # Continue to next format option

    # Step 3: Verify download and try to get subtitles if needed
    if not os.path.exists(video_filename) or os.path.getsize(video_filename) == 0:
        raise Exception("Failed to download video with any available format")

    # Try each subtitle option until one works
    subtitle_downloaded = False
    for sub_option in subtitle_options:
        if subtitle_downloaded:
            break

        try:
            sub_format = sub_option['format']
            sub_lang = sub_option['language']
            temp_subtitle = os.path.join(download_path, f"{snake_case_title}.{sub_lang}.{sub_format}")

            sub_opts = {
                'skip_download': True,
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': [sub_lang],
                'subtitlesformat': sub_format,
                'outtmpl': os.path.join(download_path, f"{snake_case_title}"),
                'ignoreerrors': True,
            }

            with YoutubeDL(sub_opts) as ydl:
                ydl.download([url])

            # Check if subtitle was downloaded
            if os.path.exists(temp_subtitle) and os.path.getsize(temp_subtitle) > 0:
                # Convert to SRT if needed
                if sub_format != 'srt':
                    try:
                        convert_subtitle_to_srt(temp_subtitle, subtitle_filename)
                        print(f"Converted {sub_format} subtitle to SRT format")
                    except Exception as conv_e:
                        print(f"Failed to convert subtitle: {str(conv_e)}")
                        # Use the subtitle we have if conversion fails
                        subtitle_filename = temp_subtitle
                else:
                    subtitle_filename = temp_subtitle

                subtitle_downloaded = True
                print(f"Successfully downloaded subtitles: {subtitle_filename}")
        except Exception as sub_e:
            print(f"Failed to download subtitles with {sub_option}: {str(sub_e)}")

    # Step 4: Try to embed subtitles if both files exist
    if subtitle_downloaded and os.path.exists(video_filename) and os.path.exists(subtitle_filename):
        try:
            embed_subtitles(video_filename, subtitle_filename)
            print(f"Embedded subtitles into video: {video_filename}")
        except Exception as embed_e:
            print(f"Failed to embed subtitles: {str(embed_e)}")

    # Step 5: Return paths even if subtitle download failed
    return video_filename, subtitle_filename, video_title, snake_case_title


def convert_subtitle_to_srt(input_path, output_path):
    """
    Convert subtitle files to SRT format using FFmpeg

    Args:
        input_path (str): Path to input subtitle file
        output_path (str): Path to output SRT file
    """
    try:
        subprocess.run([
            'ffmpeg', '-i', input_path,
            '-f', 'srt', output_path
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception as e:
        print(f"Subtitle conversion error: {str(e)}")
        return False


def download_subtitle(url, output_path):
    """
    Download subtitles separately

    Args:
        url (str): YouTube video URL
        output_path (str): Path to save subtitle file
    """
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en', 'en-US'],
        'subtitlesformat': 'srt/vtt/ass/best',
        'outtmpl': output_path.replace('.en.srt', ''),
        'ignoreerrors': True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Check if subtitle was downloaded with various possible extensions
    possible_paths = [
        output_path,
        output_path.replace('.en.srt', '.en.vtt'),
        output_path.replace('.en.srt', '.en-US.srt'),
        output_path.replace('.en.srt', '.en-US.vtt'),
        output_path.replace('.en.srt', '.en.ass'),
    ]

    for path in possible_paths:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            # Convert to SRT if it's not already
            if not path.endswith('.srt'):
                try:
                    convert_subtitle_to_srt(path, output_path)
                    os.remove(path)  # Remove original after conversion
                except Exception:
                    # If conversion fails, just copy the file
                    shutil.copy(path, output_path)
            elif path != output_path:
                shutil.copy(path, output_path)

            return True

    return False


def embed_subtitles(video_path, subtitle_path):
    """
    Embed subtitles into video file using FFmpeg

    Args:
        video_path (str): Path to video file
        subtitle_path (str): Path to subtitle file
    """
    output_path = video_path + ".temp.mp4"

    try:
        subprocess.run([
            'ffmpeg', '-i', video_path,
            '-i', subtitle_path,
            '-c:v', 'copy', '-c:a', 'copy',
            '-c:s', 'mov_text', '-metadata:s:s:0', 'language=eng',
            output_path
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Replace original with new file
        os.replace(output_path, video_path)
        return True
    except Exception as e:
        print(f"Error embedding subtitles: {str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def download_from_youtube_single_thread(url, download_path):
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    video_title = uuid.uuid1()
    video_title = str(video_title).replace("-", "_")

    snake_case_title = to_snake_case(video_title)

    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': os.path.join(download_path, f'{snake_case_title}.%(ext)s'),
        'subtitleslangs': ['en'],
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitlesformat': 'vtt',
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    title = ydl.extract_info(url, download=False)['title']

    video_filename = os.path.join(download_path, f"{snake_case_title}.mp4")
    subtitle_filename = os.path.join(download_path, f"{snake_case_title}.en.vtt")

    return video_filename, subtitle_filename, video_title


def download_and_process_video(url, download_path, WORDS_PER_SEGMENT, cut_start=None, cut_end=None,
                               existing_video_path=None, existing_subtitle_path=None, existing_title=None):
    """
    Download and process a YouTube video with improved quality.

    Args:
        url (str): YouTube video URL
        download_path (str): Directory to save downloaded files
        WORDS_PER_SEGMENT (int): Number of words per video segment
        cut_start (str, optional): Timestamp to start from (HH:MM:SS)
        cut_end (str, optional): Timestamp to end at (HH:MM:SS)
    """
    print(f"Processing video: {url}")
    try:
        VIDEO_PATH = None
        SUBTITLE_PATH = None
        video_title = None,
        title = None
        if existing_video_path and existing_subtitle_path and existing_title:
            VIDEO_PATH = existing_video_path
            SUBTITLE_PATH = existing_subtitle_path
            video_title = existing_title
            title = existing_title
        else:
            VIDEO_PATH, SUBTITLE_PATH, video_title, title = download_from_youtube(url, download_path)
        print(f"Video downloaded to: {VIDEO_PATH}")
        print(f"Subtitle path: {SUBTITLE_PATH}")

        # Validate subtitle file
        if not os.path.exists(SUBTITLE_PATH) or os.path.getsize(SUBTITLE_PATH) == 0:
            print("Warning: Subtitle file is empty or missing. Downloading again...")
            download_subtitle(url, SUBTITLE_PATH)

        # Split video into segments
        print(f"Splitting video into segments of {WORDS_PER_SEGMENT} words...")
        split_subtitles(SUBTITLE_PATH, VIDEO_PATH, download_path, WORDS_PER_SEGMENT,
                        sub_name=title, cut_start=cut_start, cut_end=cut_end)

        print(f"Video processing complete: {video_title}")
        return VIDEO_PATH, SUBTITLE_PATH

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return None, None


def find_matching_video_subtitles(directory: str):
    """
    Finds matching .mp4 and .srt files in the given directory and embeds subtitles.

    :param directory: Path to the directory containing video and subtitle files.
    """

    files = os.listdir(directory)
    video_files = {f[:-4]: os.path.join(directory, f) for f in files if f.endswith('.mp4')}
    subtitle_files = {f[:-4]: os.path.join(directory, f) for f in files if f.endswith('.srt')}

    for name in video_files.keys():
        if name in subtitle_files:
            print(f"Processing: {video_files[name]} with {subtitle_files[name]}")
            try:
                embed_subtitles(video_files[name], subtitle_files[name])
            except Exception as e:
                print(f"Failed to process {name}: {e}")


def start_thread(download_path, WORDS_PER_SEGMENT, cut_start=None, cut_end=None,
                 existing_video_path=None, existing_subtitle_path=None, existing_title="", url=None):
    """
    Start a thread to download and process a video.

    Args:
        url (str): YouTube video URL
        download_path (str): Directory to save downloaded files
        WORDS_PER_SEGMENT (int): Number of words per video segment
        cut_start (str, optional): Timestamp to start from (HH:MM:SS)
        cut_end (str, optional): Timestamp to end at (HH:MM:SS)
    """
    # Create unique subdirectory for this download to prevent conflicts
    import hashlib
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    video_download_path = os.path.join(download_path, url_hash)

    # Create a thread for each video
    thread = Thread(target=download_and_process_video,
                    args=(url, video_download_path, WORDS_PER_SEGMENT, cut_start, cut_end, existing_video_path,
                          existing_subtitle_path, existing_title))
    thread.start()
    thread.join()

    # Embed subtitles in split video segments
    print("Embedding subtitles in video segments...")
    find_matching_video_subtitles(video_download_path)

    # Move all files to parent directory
    for file in os.listdir(video_download_path):
        source = os.path.join(video_download_path, file)
        destination = os.path.join(download_path, file)

        # Skip if file already exists in destination
        if os.path.exists(destination):
            continue

        shutil.move(source, destination)

    # Remove empty directory
    try:
        os.rmdir(video_download_path)
    except:
        pass

    print(f"Processing complete for: {url}")


def pronunciation():
    OUTPUT_DIR = "/Users/shohimardonabdurashitov/test/pro/tmp_1/down"

    videos = [
        'https://www.youtube.com/watch?v=cJZnlnT0rPA',
        # 'https://www.youtube.com/watch?v=nsi008avBfo&ab_channel=NetworkChuck'
    ]

    WORDS_PER_SEGMENT = 70
    cut_start = "00:00:00"
    cut_end = "00:20:00"

    for url in videos:
        start_thread(url, OUTPUT_DIR, WORDS_PER_SEGMENT, cut_start, cut_end)


def pronunciationExistFile():
    OUTPUT_DIR = "/Users/shohimardonabdurashitov/test/pro/tmp_1/down"

    existing_video_path = "/Users/shohimardonabdurashitov/PycharmProjects/VideoChunkMaster/everyone is putting AI in schools.......mkv"
    existing_subtitle_path = "/Users/shohimardonabdurashitov/PycharmProjects/VideoChunkMaster/[English (auto-generated)] everyone is putting AI in schools...... [DownSub.com].srt"
    existing_title = "everyone is putting AI in schools......."
    WORDS_PER_SEGMENT = 70
    cut_start = "00:00:00"
    cut_end = "00:20:00"
    start_thread(OUTPUT_DIR, WORDS_PER_SEGMENT, cut_start, cut_end, existing_video_path, existing_subtitle_path,
                 existing_title, url="https://www.youtube.com/watch?v=cJZnlnT0rPA")


def podcast():
    _OUTPUT_DIR = "/Users/shohimardonabdurashitov/Documents/Podcast/Feb_18/new_v_1_0"

    videos = [
        'https://www.youtube.com/watch?v=N5DAW8mkJ6Y&ab_channel=AndrewHuberman',
    ]

    _WORDS_PER_SEGMENT = 1000
    cut_start = None
    cut_end = None

    for url in videos:
        start_thread(_OUTPUT_DIR, _WORDS_PER_SEGMENT, cut_start, cut_end, url=url)


def listenAndWrite():
    _OUTPUT_DIR = "/Users/shohimardonabdurashitov/Documents/ListenAndWrite/Feb_19/high"

    videos = [
        'https://www.youtube.com/watch?v=u7pu1cQBqtQ&t=143s&ab_channel=CleoAbram',
    ]

    _WORDS_PER_SEGMENT = 500
    cut_start = "00:01:20"
    cut_end = "00:10:00"

    for url in videos:
        start_thread(_OUTPUT_DIR, _WORDS_PER_SEGMENT, cut_start, cut_end, url=url)


if __name__ == '__main__':
    pronunciationExistFile()
