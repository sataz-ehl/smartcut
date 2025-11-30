# SmartCut Video Format Support

This document details the video formats and codecs supported by the smartcut library, with specific focus on the minimal re-encoding feature during cutting and fading operations.

## Supported Video Container Formats

SmartCut supports all major video container formats:

### Video Containers
- **MP4** (.mp4) - Most common format, maximum compatibility
- **Matroska** (.mkv) - Supports attachments (e.g., subtitles, fonts)
- **MOV** (.mov) - Apple QuickTime format
- **AVI** (.avi)
- **FLV** (.flv) - Flash Video
- **WMV** (.wmv) - Windows Media Video
- **MPEG/MPG** (.mpg)
- **MPEG-TS** (.ts) - MPEG Transport Stream
- **M2TS** (.m2ts) - Blu-ray MPEG-2 TS format
- **WebM** (.webm) - For VP9/AV1 codecs

### Audio-Only Containers
(`smartcut/media_utils.py:103-109`)
- **MP3** - MPEG Audio
- **FLAC** - Free Lossless Audio Codec
- **OGG** - Ogg Vorbis/Opus container
- **WAV** - PCM audio
- **M4A/iPod** - AAC audio format

## Supported Video Codecs

### Source Codec Support
(`smartcut/media_container.py` and test coverage)

- **H.264** - Full NAL unit type detection (IDR, parameter sets)
  - All profiles: baseline, main, high, high10, high422, high444
- **H.265/HEVC** - Full NAL unit detection with CRA→BLA conversion
- **VP9** - Including profile detection
- **AV1** - With decoder-to-encoder mapping (libdav1d → libaom-av1)
- **MPEG-2** - Specialized time base handling
- **MPEG-4 Visual** - With bitstream filters

### Output Codec Support
(`smartcut/media_utils.py`)

- H.264 with profile preservation
- HEVC/H.265 with x265-params configuration
- VP9 with profile validation
- AV1 with lossless support
- Copy/passthrough codec (default for minimal re-encoding)

## Supported Audio Codecs

(`smartcut/media_utils.py:36-45`)

- **AAC** - passthrough or re-encoding
- **MP3** (libmp3lame) - passthrough or re-encoding
- **Opus** (libopus) - passthrough or re-encoding
- **Vorbis** (libvorbis) - passthrough or re-encoding
- **FLAC** - lossless passthrough
- **PCM** - 16-bit and 32-bit float
- **Passthrough** - No re-encoding (default for all audio)

## Minimal Re-encoding Strategy

The core feature of smartcut is intelligent minimal re-encoding that preserves quality while maximizing performance.

### Standard Cutting (No Fades)

- Only GOPs (Group of Pictures) at cutpoints are re-encoded
- Remaining video data is **passthrough/remuxed** (lossless)
- Performance: Cutting large videos takes seconds instead of minutes
- Uses FFmpeg bitstream filters for format conversion:
  - `h264_mp4toannexb` - MP4 → Annex B format
  - `hevc_mp4toannexb` - MP4 → Annex B format
  - `dump_extra` - MPEG-4 Visual family codec handling

### Video Fades with Minimal Re-encoding

**Video-Only Fades** (`videofadein`/`videofadeout`):
- Selective GOP re-encoding - only re-encode GOPs overlapping fade regions
- Audio uses passthrough (no re-encoding)
- Performance: **~17x faster** than full re-encoding
- Trade-off: Fades may start/end 0.5-1 second early/late due to GOP boundaries
- Example: 20-second segment with 2s fade-in/out:
  - Only 4-6 seconds re-encoded (~20-30%)
  - 14-16 seconds passthrough (~70-80%)

**Audio-Only Fades** (`audiofadein`/`audiofadeout`):
- Re-encode all audio in affected segments
- Video uses passthrough (no re-encoding)
- **~5x faster** than video fades
- Audio sample-level fade application (`smartcut/cut_video.py:193-256`)

**Mixed Fades** (e.g., `videofadein` + `audiofadeout`):
- Each media type follows its own strategy
- Video: Selective GOP re-encoding
- Audio: Full segment re-encoding

**Legacy Fades** (`fadein`/`fadeout`):
- Apply to both video and audio
- Full re-encoding of affected GOPs for both media types
- Backward compatible with existing scripts

## Format-Specific Restrictions

(`smartcut/media_utils.py:138-189`)

### Container Compatibility Restrictions

**H.264 codec:**
- ❌ NOT supported in OGG containers
- ✅ Supported in MP4, MKV, TS, MOV

**H.265/HEVC codec:**
- ❌ NOT supported in MP3 containers
- ❌ NOT supported in OGG containers
- ✅ Supported in MP4, MKV, WebM, MOV

**Audio-Only Format Limitations:**
- MP3, FLAC, OGG, WAV, M4A: **Maximum 1 audio track**
- Audio mixing would cause re-encoding errors
- Codec mapping is strict (`.mp3` → libmp3lame, `.flac` → FLAC, etc.)

### Codec Tag Normalization

(`smartcut/cut_video.py:574-608`)

**MP4/MOV containers** normalize codec tags for compatibility:
- H.264: Tag `27` (MPEG-TS) → `avc1` (MP4 standard)
- H.265: Tag `36` or `HEVC` (MPEG-TS) → `hvc1` (MP4 standard)
- Ensures proper playback across players

### Format Detection

(`smartcut/cut_video.py:1299-1315`)

- Output format automatically determined from file extension
- Audio-only formats detected: 'ogg', 'mp3', 'm4a', 'ipod', 'flac', 'wav'
- Video excluded from output if audio-only format detected
- Matroska/WebM containers auto-detect attachment support

## Special Format Handling

### MPEG-2 Video
(`smartcut/cut_video.py:820-822`)
- Special time base handling
- Frame duration recalculation for proper packet timing

### H.265 CRA Frame Handling
(`smartcut/cut_video.py:944-975`)
- CRA (Clean Random Access) frames converted to BLA (Broken Link Access) when needed
- Applies intelligent conversion logic
- Only when stream continuity breaks (after cut or skip)

### H.264 NAL Unit Detection
(`smartcut/cut_video.py:550-556`)
- Safe keyframe detection for both MP4 and Annex B formats
- Parameter sets preserved in output

## Testing Coverage

All format combinations are tested in the test suite (see README lines 176-219):

- ✅ H.264 smart cutting with all profiles
- ✅ H.265 large file smart cutting
- ✅ VP9 and VP9 profile handling
- ✅ AV1 codec support
- ✅ Format conversion tests (MP4↔MKV conversions)
- ✅ Container-specific tests (AVI, FLV, MOV, WMV, MPEG, TS, M2TS)
- ✅ All codec/container combinations

## Performance Characteristics

(`FADE_FEATURE_SUMMARY.md:334-371`)

| Scenario | Speed | Notes |
|----------|-------|-------|
| Smart cut (no fades) | Baseline | Only GOPs at cutpoints re-encoded |
| Video-only fades | ~17x faster than full | Audio passthrough |
| Audio-only fades | ~5x faster than full | Video passthrough |
| Full re-encode | 100x slower | Processes entire video |

## Practical Recommendations

1. **For fastest cutting**: Use `videofadein`/`videofadeout` to keep audio as passthrough
2. **For format conversion**: MP4↔MKV conversions fully tested and optimized
3. **For maximum compatibility**: Use MP4 with H.264 codec (most universal)
4. **For archival**: Use MKV with H.265 (better compression, supports attachments)
5. **For streaming**: Use VP9 or AV1 in WebM or MP4
6. **For audio mixing**: Only use multi-track audio in video containers, not audio-only formats

## Key Takeaway

**The minimal re-encoding approach works universally across all supported formats.** It fundamentally preserves the original video quality while achieving near-instantaneous cutting performance by only re-encoding the small GOP regions around cutpoints and fade boundaries. The strategy is format-agnostic - whether you're working with MP4, MKV, MOV, or any other supported container, you get the same intelligent minimal re-encoding benefits.
