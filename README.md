# VideoChunkMaster

**VideoChunkMaster** is a powerful Python-based tool designed to download, process, and segment YouTube videos into smaller, manageable chunks based on their subtitles. This tool is particularly useful for language learners, podcast enthusiasts, content creators, and anyone who needs to break down long videos into smaller, more digestible segments. With advanced features like multi-threaded downloads, subtitle embedding, and precise video cutting, **VideoChunkMaster** ensures a seamless and efficient video processing experience.

## Key Features

### 1. YouTube Video Downloader
- Downloads high-quality video and audio from YouTube.
- Supports multiple formats and resolutions.
- Concurrent fragment downloads for faster performance.

### 2. Subtitle Handling
- Downloads subtitles in SRT format.
- Automatically embeds subtitles into the video file.
- Handles both manual and automatic subtitles.

### 3. Video Segmentation
- Splits videos into smaller segments based on word count.
- Ensures segments end at natural sentence boundaries.
- Supports custom start and end timestamps for precise cutting.

### 4. Parallel Processing
- Utilizes multi-threading for faster downloads and processing.
- Configurable number of concurrent downloads.

### 5. Progress Tracking
- Real-time progress tracking for downloads and processing.
- Detailed logging for debugging and monitoring.

### 6. File Management
- Organizes downloaded files into structured directories.
- Automatically cleans up temporary files.

## Use Cases
- **Language Learning**: Break down videos into smaller segments for focused listening and repetition.
- **Podcast Preparation**: Extract specific segments from long interviews or discussions.
- **Content Creation**: Create clips from longer videos for social media or presentations.
- **Educational Purposes**: Segment lectures or tutorials for easier consumption.

## Installation

### Prerequisites
- Python 3.8 or higher.
- **FFmpeg** installed on your system. [Download FFmpeg here](https://ffmpeg.org/download.html).

### Steps

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/ShohimardonDev/VideoChunkMaster.git
    cd VideoChunkMaster
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Verify FFmpeg Installation**:
    Ensure FFmpeg is installed and accessible from your command line:
    ```bash
    ffmpeg -version
    ```

## Usage

### Basic Usage
```python
from video_chunk_master import download_and_process_video

url = "https://www.youtube.com/watch?v=example"
download_path = "/path/to/download"
WORDS_PER_SEGMENT = 120

download_and_process_video(url, download_path, WORDS_PER_SEGMENT)
