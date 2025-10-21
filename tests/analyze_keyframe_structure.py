#!/usr/bin/env python3
"""
Analyze keyframe structure of video files.
Standalone tool for debugging GOP detection and keyframe analysis.
"""

import os
import sys
from pathlib import Path

import av

from smartcut.nal_tools import get_h264_nal_unit_type, get_h265_nal_unit_type, is_safe_h264_keyframe_nal, is_safe_h265_keyframe_nal

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def analyze_keyframes_structure(input_path):
    """Analyze and display keyframe structure of the input video"""
    try:

        # Open file directly to analyze all keyframes
        av_container = av.open(input_path, 'r', metadata_errors='ignore')
        video_stream = None

        if len(av_container.streams.video) == 0:
            print("No video stream found in file.")
            return

        video_stream = av_container.streams.video[0]
        video_stream.thread_type = "FRAME"

        print(f"Keyframe Analysis for: {input_path}")
        print("=" * 60)

        # Basic video info
        ctx = video_stream.codec_context
        duration = float(av_container.duration / av.time_base)

        print("Video info:")
        print(f"  Codec: {ctx.name}")
        print(f"  Resolution: {ctx.width}x{ctx.height}")
        print(f"  Duration: {duration:.2f} seconds")

        # Collect all keyframes and packets for H.265 analysis
        all_keyframes = []
        valid_gop_keyframes = []
        all_packets = []  # Store all packets for next-packet analysis

        first_keyframe = True
        frame_count = 0

        # First pass: collect all packets
        for packet in av_container.demux(video_stream):
            if packet.pts is None:
                continue

            if packet.stream.type == 'video':
                frame_count += 1

                frame_time = float(packet.pts * packet.time_base)
                nal_type = None
                nal_type_str = "Unknown"

                # Determine NAL type for all packets (not just keyframes)
                if ctx.name == 'hevc':
                    nal_type = get_h265_nal_unit_type(bytes(packet))
                    if nal_type is not None:
                        if nal_type in [16, 17, 18]:
                            nal_type_str = f"BLA({nal_type})"
                        elif nal_type in [19, 20]:
                            nal_type_str = f"IDR({nal_type})"
                        elif nal_type == 21:
                            nal_type_str = "CRA(21)"
                        elif nal_type in [32, 33, 34]:
                            nal_type_str = f"ParamSet({nal_type})"
                        else:
                            nal_type_str = f"Other({nal_type})"
                elif ctx.name == 'h264':
                    nal_type = get_h264_nal_unit_type(bytes(packet))
                    if nal_type is not None:
                        if nal_type == 5:
                            nal_type_str = "IDR(5)"
                        elif nal_type in [7, 8]:
                            nal_type_str = f"ParamSet({nal_type})"
                        else:
                            nal_type_str = f"Other({nal_type})"

                # Store packet info for analysis
                all_packets.append({
                    'time': frame_time,
                    'nal_type': nal_type,
                    'nal_type_str': nal_type_str,
                    'is_keyframe': packet.is_keyframe,
                    'frame_idx': frame_count - 1
                })

                # Add all keyframes (as marked by container)
                if packet.is_keyframe:
                    all_keyframes.append((frame_time, nal_type_str))

                # For GOP analysis, use the same safety checks as media_container
                is_picture_keyframe = False
                if ctx.name == 'hevc':
                    is_picture_keyframe = is_safe_h265_keyframe_nal(nal_type)
                elif ctx.name == 'h264':
                    is_picture_keyframe = is_safe_h264_keyframe_nal(nal_type)

                if is_picture_keyframe:
                    # Check if this would be accepted as a valid GOP keyframe
                    is_valid_gop = True
                    if first_keyframe:
                        first_keyframe = False  # First picture keyframe is always accepted
                    # Use the same safety checks as media_container for consistency
                    elif ctx.name == 'hevc':
                        is_valid_gop = is_safe_h265_keyframe_nal(nal_type)
                    elif ctx.name == 'h264':
                        is_valid_gop = is_safe_h264_keyframe_nal(nal_type)

                    if is_valid_gop:
                        valid_gop_keyframes.append((frame_time, nal_type_str))

        print(f"  Total frames: {frame_count}")
        print(f"  Total keyframes: {len(all_keyframes)}")
        print(f"  Valid GOP keyframes: {len(valid_gop_keyframes)}")

        # Show first 20 keyframes (all types)
        print("\nFirst 20 keyframes (all types):")
        keyframes_to_show = all_keyframes[:20]
        if keyframes_to_show:
            for i, (time, nal_str) in enumerate(keyframes_to_show):
                print(f"  {i+1:2d}: {time:7.3f}s - {nal_str}")
            if len(all_keyframes) > 20:
                print(f"  ... and {len(all_keyframes) - 20} more keyframes")
        else:
            print("  No keyframes found!")

        # For H.265 files, show the NAL types of the 3 next packets after each keyframe
        if ctx.name == 'hevc' and all_packets:
            print("\nH.265 Keyframes with next 3 NAL types:")
            print("-" * 60)
            keyframe_count = 0
            for i, packet in enumerate(all_packets):
                if packet['is_keyframe'] and keyframe_count < 20:  # Limit to first 20 keyframes
                    keyframe_count += 1
                    next_nals = []
                    # Get next 3 packets
                    for j in range(1, 8):
                        if i + j < len(all_packets):
                            next_packet = all_packets[i + j]
                            next_nals.append(next_packet['nal_type_str'])
                        else:
                            next_nals.append("EOF")

                    print(f"  KF{keyframe_count:2d}: {packet['time']:7.3f}s - {packet['nal_type_str']}")
                    print(f"       Next 3: {' → '.join(next_nals)}")

            if keyframe_count == 0:
                print("  No H.265 keyframes found in packet stream!")
            elif sum(1 for p in all_packets if p['is_keyframe']) > 20:
                remaining = sum(1 for p in all_packets if p['is_keyframe']) - 20
                print(f"  ... and {remaining} more keyframes")

        # Show first 20 valid GOP keyframes
        print("\nFirst 20 valid GOP keyframes:")
        gop_keyframes_to_show = valid_gop_keyframes[:20]
        if gop_keyframes_to_show:
            for i, (time, nal_str) in enumerate(gop_keyframes_to_show):
                print(f"  {i+1:2d}: {time:7.3f}s - {nal_str}")
            if len(valid_gop_keyframes) > 20:
                print(f"  ... and {len(valid_gop_keyframes) - 20} more valid GOP keyframes")
        else:
            print("  No valid GOP keyframes found!")

        # GOP analysis (from MediaContainer)
        if len(valid_gop_keyframes) > 1:
            gop_durations = []
            for i in range(len(valid_gop_keyframes)):
                start_time = valid_gop_keyframes[i][0]
                if i + 1 < len(valid_gop_keyframes):
                    end_time = valid_gop_keyframes[i + 1][0]
                else:
                    end_time = duration
                gop_duration = end_time - start_time
                gop_durations.append(gop_duration)

            avg_gop_duration = sum(gop_durations) / len(gop_durations)
            max_gop_duration = max(gop_durations)
            min_gop_duration = min(gop_durations)

            print("\nGOP Statistics:")
            print(f"  Average GOP duration: {avg_gop_duration:.2f} seconds")
            print(f"  Min GOP duration: {min_gop_duration:.2f} seconds")
            print(f"  Max GOP duration: {max_gop_duration:.2f} seconds")

            # Warn about problematic GOP sizes
            if max_gop_duration > 30:
                print(f"  ⚠️  WARNING: Large GOP detected ({max_gop_duration:.1f}s) may cause high memory usage")
            elif max_gop_duration > 10:
                print(f"  ⚠️  NOTICE: Moderately large GOP detected ({max_gop_duration:.1f}s)")

        av_container.close()

    except Exception as e:
        print(f"Error analyzing keyframe structure: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point for the keyframe analysis tool."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze keyframe structure of video files for debugging GOP detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  Analyze keyframes in a video file:
    python analyze_keyframe_structure.py input.mp4

  Analyze HEVC file with GOP issues:
    python analyze_keyframe_structure.py video.mkv
        """
    )

    parser.add_argument('input', metavar='INPUT', type=str,
                       help="Input video file to analyze")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File '{args.input}' not found.")
        return 1

    analyze_keyframes_structure(args.input)
    return 0


if __name__ == "__main__":
    sys.exit(main())
