import argparse
import av
from fractions import Fraction
from smartcut.media_container import MediaContainer
from smartcut.cut_video import smart_cut, VideoSettings, VideoExportMode, VideoExportQuality, AudioExportSettings, AudioExportInfo
from tqdm import tqdm

def time_to_fraction(time_str_elem):
    if ':' in time_str_elem:
        parts = time_str_elem.split(':')
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = Fraction(parts[2])
            return Fraction(hours * 3600) + Fraction(minutes * 60) + seconds
        elif len(parts) == 2:
            minutes = int(parts[0])
            seconds = Fraction(parts[1])
            return Fraction(minutes * 60) + seconds
        else:
            raise ValueError("Timestamp must be in format HH:MM:SS or MM:SS")

    return Fraction(time_str_elem)

def parse_time_segments(time_str):
    times = list(map(time_to_fraction, time_str.split(',')))
    if len(times) % 2 != 0:
        raise ValueError("You must provide an even number of time points for segments.")
    return list(zip(times[::2], times[1::2]))

def frame_to_time(source, frame_str, end_frame = False):
    frame_num = int(frame_str)
    if frame_num == -1:
        # Special case: frame "-1" means "the final frame of the video"
        # (also intentionally not including the ` - source.start_time` offset, to further ensure the chosen time
        #  is all the way at the end of the file)
        return source.video_frame_times[len(source.video_frame_times) - 1]
    if end_frame:
        # Internal calculations in `smart_cut` function currently *exclude* the final frame if it lands
        # exactly on the specified end time, so we manually offset "end" frames by 1
        frame_num += 1
    return source.video_frame_times[frame_num] - source.start_time

def parse_frame_segments(source, frame_str):
    all_frames = frame_str.split(',')
    if len(all_frames) % 2 != 0:
        raise ValueError("You must provide an even number of frames for segments.")
    start_frames = list(map(lambda f: frame_to_time(source, f), all_frames[::2]))
    end_frames = list(map(lambda f: frame_to_time(source, f, True), all_frames[1::2]))
    return list(zip(start_frames, end_frames))

class Progress:
    def __init__(self):
        self.first_call = True
        self.tqdm = None

    def emit(self, value):
        if self.first_call:
            self.first_call = False
            self.tqdm = tqdm(total=value)
            return
        self.tqdm.update(1)

def main():
    description = (
        "SmartCut - Efficient video cutting with minimal recoding. "
        "Only segments around cutpoints are re-encoded, preserving original quality "
        "for the majority of the video. Supports various formats including MP4, MKV, AVI, MOV."
    )

    epilog = """
examples:
  Keep segments from 10-20s and 40-50s:
    smartcut input.mp4 output.mp4 --keep 10,20,40,50

  Cut out segments from 30-40s and 1m-1m10s:
    smartcut input.mp4 output.mp4 --cut 30,40,01:00,01:10

  Use frame numbers instead of times:
    smartcut input.mp4 output.mp4 --keep 300,600,1200,1500 --frames

  Cut using different time formats (mix seconds and minutes syntax):
    smartcut input.mp4 output.mp4 --keep 90,2:45,3:00,4:00

  Subsecond precision cutting:
    smartcut input.mp4 output.mp4 --keep 00:06:48.799,00:06:50.123

time formats:
  - Seconds: 10, 30.5, 120
  - MM:SS: 01:30, 02:45
  - HH:MM:SS: 01:30:45
  - Subseconds: 48.799, 01:30.123, 01:30:45.678
"""

    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('input', metavar='INPUT', type=str,
                       help="Input media file (MP4, MKV, AVI, MOV, etc.)")
    parser.add_argument('output', metavar='OUTPUT', type=str,
                       help="Output media file path")
    parser.add_argument('--keep', metavar='SEGMENTS', type=str,
                       help="Keep specified time segments. Format: start1,end1,start2,end2,... "
                            "Times can be in seconds (10), MM:SS (01:30), HH:MM:SS (01:30:45), "
                            "or with subseconds (48.799, 01:30.123, 01:30:45.678)")
    parser.add_argument('--cut', metavar='SEGMENTS', type=str,
                       help="Remove specified time segments (opposite of --keep). "
                            "Same time format as --keep option")
    parser.add_argument('--frames', action='store_true',
                       help="Interpret --keep/--cut values as frame numbers instead of times. "
                            "Frames are zero-indexed. Use -1 for the last frame")
    parser.add_argument('--log-level', choices=['warning', 'error', 'fatal'],
                       default='warning', metavar='LEVEL',
                       help="Set logging verbosity level (default: %(default)s)")
    parser.add_argument('--version', action='version', version='Smartcut 1.3.0')

    args = parser.parse_args()

    if args.keep and args.cut or not (args.keep or args.cut):
        raise ValueError("You must specify either --keep or --cut, not both.")

    source = MediaContainer(args.input)

    if args.keep:
        if args.frames:
            segments = parse_frame_segments(source, args.keep)
        else:
            segments = parse_time_segments(args.keep)
    elif args.cut:
        if args.frames:
            cut_segments = parse_frame_segments(source, args.cut)
        else:
            cut_segments = parse_time_segments(args.cut)
        segments = [(Fraction(0), source.duration())]
        for c_start, c_end in cut_segments:
            last_segment = segments.pop()
            if c_start > last_segment[0]:
                segments.append((last_segment[0], c_start))
            if c_end < last_segment[1]:
                segments.append((c_end, last_segment[1]))
    else:
        raise ValueError("You must specify either --keep or --cut.")

    # Default audio settings: no mix, include all tracks with lossless passthru
    audio_settings = [AudioExportSettings(codec='passthru')] * len(source.audio_tracks)
    export_info = AudioExportInfo(output_tracks=audio_settings)

    video_settings = VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.NORMAL, None)

    progress = Progress()

    if args.log_level == 'warning':
        av.logging.set_level(av.logging.WARNING)
    if args.log_level == 'error':
        av.logging.set_level(av.logging.ERROR)
    if args.log_level == 'fatal':
        av.logging.set_level(av.logging.FATAL)

    exception_value = smart_cut(source, segments, args.output,
                                audio_export_info=export_info,
                                video_settings=video_settings,
                                progress=progress, log_level=args.log_level)

    progress.tqdm.close()

    if exception_value is not None:
        raise exception_value

    print(f"Smart cut completed successfully. Output saved to {args.output}")

if __name__ == '__main__':
    main()
