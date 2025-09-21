# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Smart Media Cutter (smartcut) is a Python-based CLI tool for efficient video cutting that performs "smart cuts" - only recoding around the cutpoints while preserving the majority of the original video quality. It's designed as an open-source companion to the commercial Smart Media Cutter application.

## Key Commands

### Development Setup
```bash
# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Install test dependencies
pip install -r requirements_tests.txt
```

### Running the Application
```bash
# Direct execution
python ./smartcut input.mp4 output.mp4 --keep 10,20,40,50

# After installation via setup.py
python -m smartcut input.mp4 output.mp4 --keep 10,20,40,50
```

### Testing
```bash
# Run all tests, this takes 4mins and will timeout so don't run this directly
python tests/smartcut_tests.py
```

from tests/smartcut_tests.py:
```python
parser.add_argument('--category', choices=['basic', 'h264', 'h265', 'codecs', 'containers', 'audio', 'mixed', 'transforms', 'long', 'external', 'all'],
                  help='Run tests from specific category')
parser.add_argument('--list-categories', action='store_true', help='List available test categories')
```

```bash
# Run x265 tests
python tests/smartcut_tests.py --category h265

# Run tests with specific files
python tests/smartcut_tests.py path/to/video.mp4
```

### Building Executables
```bash
# Linux
./linux_package.sh

# Windows
windows_package.bat
```

## Architecture Overview

### Core Modules

1. **smartcut/__main__.py** - CLI entry point and argument parsing
   - Handles time/frame segment parsing
   - Configures audio/video export settings
   - Provides progress tracking via tqdm

2. **smartcut/media_container.py** - Media file abstraction
   - `MediaContainer`: Main class for handling input media files
   - `AudioTrack`: Represents individual audio tracks
   - Uses PyAV for low-level media access
   - Handles GOP (Group of Pictures) analysis for smart cutting

3. **smartcut/cut_video.py** - Core cutting logic
   - `smart_cut()`: Main function orchestrating the cutting process
   - `VideoCutter`: Handles video stream processing
   - `PassthruAudioCutter`: Manages audio passthrough
   - `SubtitleCutter`: Handles subtitle track processing
   - Implements smart cutting algorithm using keyframe analysis

4. **smartcut/misc_data.py** - Data structures and configuration
   - Dataclasses for export settings and video transformations
   - Audio/video export configuration objects

### Smart Cutting Algorithm

The core innovation is the smart cutting approach:
- Identifies GOP boundaries using keyframe analysis
- Segments requiring recoding are minimized to cutpoint boundaries
- Most content is copied directly without reencoding
- Uses PyAV to access FFmpeg functionality at a low level

### Key Dependencies

- **PyAV (av)**: Python binding for FFmpeg, provides low-level media processing
- **numpy**: Used for frame time calculations and GOP analysis
- **tqdm**: Progress bar functionality
- **ffmpeg-python**: Higher-level FFmpeg operations (tests only)
- **soundfile/scipy**: Audio analysis for testing

### Testing Framework

Tests are comprehensive and format-specific:
- Tests multiple video codecs (H.264, H.265, VP9, AV1, MPEG-2)
- Validates pixel-perfect accuracy for video
- Audio tests use correlation analysis since exact matching is difficult
- Some tests depend on proprietary GUI components and are disabled in open-source version

### File Structure

```
smartcut/
├── __main__.py          # CLI entry point
├── cut_video.py         # Core cutting algorithm
├── media_container.py   # Media file handling
└── misc_data.py         # Data structures
tests/
└── smartcut_tests.py    # Comprehensive test suite
```

### Important Design Considerations

- Requires Python 3.11+
- Designed to work with both open-source CLI and proprietary GUI versions
- Audio tracks default to passthrough with lossless quality
- Frame-accurate cutting supported via `--frames` flag
- Supports both "keep" and "cut" operations on time segments
- always use venv `./venv`