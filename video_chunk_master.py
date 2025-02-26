import json
from concurrent.futures import ThreadPoolExecutor
from threading import Thread

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading
import uuid
from yt_dlp import YoutubeDL
from tqdm import tqdm
import subprocess
import re

import shutil


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


def embed_subtitles(video_file: str, subtitle_file: str, overwrite: bool = True):
    """
    Embeds subtitles into a video file using FFmpeg and overwrites the original file safely.

    :param video_file: Path to the input video file.
    :param subtitle_file: Path to the subtitle file (.srt).
    :param overwrite: Whether to overwrite the original file (default: True).
    :raises subprocess.CalledProcessError: If FFmpeg command fails.
    """

    temp_output = f"{video_file}.temp.mp4"  # Temporary output file

    command = [
        'ffmpeg',
        '-i', video_file,  # Input video file
        '-i', subtitle_file,  # Subtitle file (e.g., .srt)
        '-c:v', 'copy',  # Copy video codec
        '-c:a', 'copy',  # Copy audio codec
        '-c:s', 'mov_text',  # Set subtitle codec for MP4
        '-y',  # Overwrite without prompt
        temp_output  # Temporary output file
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Replace original file with the new one
        shutil.move(temp_output, video_file)
        print(f"Subtitles embedded successfully: {video_file}")

    except subprocess.CalledProcessError as e:
        print(f"Error embedding subtitles: {e.stderr.decode().strip()}")
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
    try:
        # Adjust start and end times based on optional cut_start and cut_end
        if cut_start:
            cut_start_time = parse_timestamp_own(cut_start)
            start_time = max(start_time, cut_start_time)

        if cut_end:
            cut_end_time = parse_timestamp_own(cut_end)
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
                        progress_bar.n = elapsed
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
    """Adjust the timestamp by subtracting offset and ensuring a minimum start time of 00:00:02.000."""
    # Parse the timestamp more carefully to preserve millisecond precision
    parts = original.split(':')
    h, m = float(parts[0]), float(parts[1])
    s_parts = parts[2].split(',')
    s = float(s_parts[0]) + float(s_parts[1]) / 1000

    total_seconds = max(2.0, h * 3600 + m * 60 + s - offset)

    # Format with proper handling of milliseconds
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    if float(seconds) > 1.5 and is_double_line == False:
        seconds -= 2
        # print()
    milliseconds = int((total_seconds - int(total_seconds)) * 1000)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def parse_timestamp_s(timestamp: str) -> float:
    """Parse SRT timestamp format (HH:MM:SS,mmm) to seconds."""
    parts = timestamp.split(':')
    h, m = float(parts[0]), float(parts[1])
    s_parts = parts[2].split(',')
    s = float(s_parts[0]) + float(s_parts[1]) / 1000
    return h * 3600 + m * 60 + s


def save_section(section_id, blocks, output_dir, word_limit, sub_name):
    """Save subtitle section and corresponding video clip with adjusted timestamps."""
    if not blocks:
        return None

    original_start_time = parse_timestamp_s(blocks[0][1].split(" --> ")[0])
    offset = original_start_time  # Use first timestamp as the offset

    start_time = parse_timestamp_s(blocks[0][1].split(" --> ")[0])  # Fix: Use the first subtitle's timestamp
    end_time = parse_timestamp_s(blocks[-1][1].split(" --> ")[1])  # Last subtitle's end timestamp

    content = ""
    new_subtitle_id = 1

    # Keep track of the end time of the previous subtitle
    last_end_time = 0

    for index, timestamp, text in blocks:
        start, end = timestamp.split(" --> ")
        is_double_line = len(text.split("\n")) > 1
        adjusted_start = adjust_timestamp(start, offset, is_double_line)
        adjusted_end = adjust_timestamp(end, offset, is_double_line)

        # Convert adjusted timestamps back to seconds for comparison
        start_seconds = parse_timestamp(adjusted_start)
        end_seconds = parse_timestamp(adjusted_end)

        # Ensure this subtitle starts after the previous one ended
        if start_seconds < last_end_time:
            # Adjust the start time to be just after the previous subtitle
            start_seconds = last_end_time + 0.001  # Add 1ms to avoid exact overlap

            # Recalculate the timestamp string
            hours = int(start_seconds // 3600)
            minutes = int((start_seconds % 3600) // 60)
            seconds = int(start_seconds % 60)
            milliseconds = int((start_seconds - int(start_seconds)) * 1000)
            adjusted_start = f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

            # If necessary, also adjust the end time to maintain duration
            if end_seconds < start_seconds:
                # Ensure minimum duration of 0.5 seconds
                end_seconds = start_seconds + 0.5
                hours = int(end_seconds // 3600)
                minutes = int((end_seconds % 3600) // 60)
                seconds = int(end_seconds % 60)
                milliseconds = int((end_seconds - int(end_seconds)) * 1000)
                adjusted_end = f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

        content += f"{new_subtitle_id}\n{adjusted_start} --> {adjusted_end}\n{text}\n\n"
        new_subtitle_id += 1

        # Update last_end_time for the next iteration
        last_end_time = parse_timestamp(adjusted_end)

    subtitle_output = os.path.join(output_dir, f"{sub_name}_{word_limit}_video_part_{section_id}.srt")
    with open(subtitle_output, "w", encoding="utf-8") as out_file:
        out_file.write(content.strip())

    print(f"Saved subtitles: {subtitle_output}")

    return start_time, end_time, os.path.join(output_dir, f"{sub_name}_{word_limit}_video_part_{section_id}.mp4")


def split_subtitles(input_file, video_file, output_dir, word_limit, sub_name, cut_start=None, cut_end=None):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the subtitle file
    with open(input_file, "r", encoding="utf-8") as file:
        subtitles = file.read()

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
                continue
            end_time_sec = parse_timestamp(timestamp.split(" --> ")[1])

            # Skip if outside cut range
            if (cut_start_seconds and start_time_sec < cut_start_seconds) or \
                    (cut_end_seconds and end_time_sec > cut_end_seconds):
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
                            executor.submit(save_video_clip, start_time, end_time, video_file, video_output, cut_start,
                                            cut_end)
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
                    executor.submit(save_video_clip, start_time, end_time, video_file, video_output, cut_start, cut_end)
                )

        # Wait for all video tasks to complete
        for task in video_tasks:
            task.result()


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


def to_snake_case(string):
    """Convert string to snake_case."""
    return string.replace(" ", "_").lower()


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

    # Generate a unique snake_case filename
    temp_id = str(uuid.uuid4()).replace("-", "_")

    # Get video info first without downloading
    ydl_info = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
        'forcejson': True,
    }

    try:
        with YoutubeDL(ydl_info) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            video_title = info_dict.get('title', temp_id)
            snake_case_title = to_snake_case(video_title)

            # Define file paths
            video_filename = os.path.join(download_path, f"{snake_case_title}.mp4")
            subtitle_filename = os.path.join(download_path, f"{snake_case_title}.en.srt")

            # Skip if already downloaded
            if os.path.exists(video_filename) and os.path.exists(subtitle_filename):
                print(f"Video and subtitles already exist: {video_filename}")
                return video_filename, subtitle_filename, video_title, snake_case_title
    except Exception as e:
        print(f"Error extracting video info: {str(e)}")
        # Fallback to temp ID if title extraction fails
        snake_case_title = temp_id
        video_filename = os.path.join(download_path, f"{snake_case_title}.mp4")
        subtitle_filename = os.path.join(download_path, f"{snake_case_title}.en.srt")
        video_title = temp_id

    # Download video with optimized settings
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',  # Prefer MP4 for better compatibility
        'outtmpl': video_filename,
        'subtitleslangs': ['en'],
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitlesformat': 'srt',  # Use SRT format directly
        'postprocessors': [
            {
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            },
            {
                'key': 'FFmpegEmbedSubtitle',  # Embed subtitles in the video file
                'already_have_subtitle': False,
            },
            {
                'key': 'FFmpegMetadata',  # Preserve metadata
                'add_metadata': True,
            }
        ],
        'concurrent_fragment_downloads': 8,  # Multiple fragment downloads
        'retries': 10,  # More retries for better reliability
        'file_access_retries': 5,
        'fragment_retries': 10,
        'skip_unavailable_fragments': False,
        'keepvideo': True,  # Keep video file after post-processing
        'verbose': False,
        'progress_hooks': [lambda d: print(f"Downloading: {d['status']} - {d.get('_percent_str', '0%')}")
        if d['status'] == 'downloading' else None],
    }

    try:
        with tqdm(desc="Downloading video", unit="B", unit_scale=True, miniters=1) as progress_bar:
            def progress_hook(d):
                if d['status'] == 'downloading':
                    if 'total_bytes' in d and 'downloaded_bytes' in d:
                        progress_bar.total = d['total_bytes']
                        progress_bar.n = d['downloaded_bytes']
                        progress_bar.refresh()

            ydl_opts['progress_hooks'] = [progress_hook]
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

        # Try to download subtitles separately in case they weren't properly extracted
        if not os.path.exists(subtitle_filename):
            try:
                download_subtitle(url, subtitle_filename)
                print(f"Downloaded subtitles separately: {subtitle_filename}")
            except Exception as sub_e:
                print(f"Subtitle download failed: {str(sub_e)}")

        # Check if subtitles were downloaded and try to embed them if they weren't embedded
        if os.path.exists(subtitle_filename) and os.path.exists(video_filename):
            try:
                embed_subtitles(video_filename, subtitle_filename)
                print(f"Embedded subtitles into video: {video_filename}")
            except Exception as embed_e:
                print(f"Failed to embed subtitles: {str(embed_e)}")

        return video_filename, subtitle_filename, video_title, snake_case_title

    except Exception as e:
        print(f"Error downloading video: {str(e)}")
        # Clean up any partial downloads
        for path in [video_filename, subtitle_filename]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
        raise e


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
        result = ydl.download([url])

    title = ydl.extract_info(url, download=False)['title']

    video_filename = os.path.join(download_path, f"{snake_case_title}.mp4")
    subtitle_filename = os.path.join(download_path, f"{snake_case_title}.en.vtt")

    return video_filename, subtitle_filename, video_title


def download_and_process_video(url, download_path, WORDS_PER_SEGMENT, cut_start=None, cut_end=None):
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


def start_thread(url, download_path, WORDS_PER_SEGMENT, cut_start=None, cut_end=None):
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
                    args=(url, video_download_path, WORDS_PER_SEGMENT, cut_start, cut_end))
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
    OUTPUT_DIR = "/Users/shohimardonabdurashitov/Documents/English/Pronounstation/FEB_17"

    videos = [
        'https://www.youtube.com/watch?v=Q4qWzbP0q7I&ab_channel=AndrewHuberman',
        'https://www.youtube.com/watch?v=nsi008avBfo&ab_channel=NetworkChuck'
    ]

    WORDS_PER_SEGMENT = 120
    cut_start = None
    cut_end = None

    for url in videos:
        start_thread(url, OUTPUT_DIR, WORDS_PER_SEGMENT, cut_start, cut_end)


def podcast():
    _OUTPUT_DIR = "/Users/shohimardonabdurashitov/Documents/Podcast/Feb_18/tmp"

    videos = [
        'https://www.youtube.com/watch?v=Q4qWzbP0q7I&ab_channel=AndrewHuberman',
    ]

    _WORDS_PER_SEGMENT = 1000
    cut_start = "00:00:00"
    cut_end = "00:30:00"

    for url in videos:
        start_thread(url, _OUTPUT_DIR, _WORDS_PER_SEGMENT, cut_start, cut_end)


def listenAndWrite():
    _OUTPUT_DIR = "/Users/shohimardonabdurashitov/Documents/ListenAndWrite/Feb_19/high"

    videos = [
        'https://www.youtube.com/watch?v=u7pu1cQBqtQ&t=143s&ab_channel=CleoAbram',
    ]

    _WORDS_PER_SEGMENT = 500
    cut_start = "00:01:20"
    cut_end = "00:10:00"

    for url in videos:
        start_thread(url, _OUTPUT_DIR, _WORDS_PER_SEGMENT, cut_start, cut_end)


if __name__ == '__main__':
    listenAndWrite()
