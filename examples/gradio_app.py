"""
Example Gradio web app for smartcut video processing.

This demonstrates how to use smartcut in a Gradio web interface with:
- Single segment processing with independent video/audio fade controls
- Multiple segment processing
- Progress tracking
- Error handling

Requirements:
    pip install smartcut gradio

Usage:
    python gradio_app.py
"""

import gradio as gr
import smartcut
import os
import tempfile


def process_single_segment(
    input_video,
    start_time,
    end_time,
    video_fadein,
    video_fadeout,
    audio_fadein,
    audio_fadeout,
    quality,
    progress=gr.Progress()
):
    """Process a single video segment with fade effects."""
    if input_video is None:
        return None, "Please upload a video file"

    try:
        # Create output file
        output_path = tempfile.mktemp(suffix=".mp4")

        # Progress callback
        def on_progress(current, total):
            progress((current, total), desc=f"Processing {current}/{total} segments")

        # Process
        success = smartcut.process(
            input_path=input_video,
            output_path=output_path,
            start=start_time,
            end=end_time if end_time > 0 else None,
            video_fadein=video_fadein if video_fadein > 0 else None,
            video_fadeout=video_fadeout if video_fadeout > 0 else None,
            audio_fadein=audio_fadein if audio_fadein > 0 else None,
            audio_fadeout=audio_fadeout if audio_fadeout > 0 else None,
            quality=quality.lower(),
            progress_callback=on_progress,
        )

        if success:
            return output_path, "✓ Processing completed successfully!"
        else:
            return None, "✗ Processing failed - check console for errors"

    except Exception as e:
        return None, f"✗ Error: {str(e)}"


def process_multiple_segments(
    input_video,
    segments_text,
    quality,
    progress=gr.Progress()
):
    """Process multiple video segments from text configuration.

    Segments format (one per line):
        start,end[,video_fadein,video_fadeout,audio_fadein,audio_fadeout]

    Examples:
        10,20,2,3,2,3  # Both fades
        30,40,2,3,0,0  # Video fade only
        50,60,0,0,1,2  # Audio fade only
    """
    if input_video is None:
        return None, "Please upload a video file"

    if not segments_text.strip():
        return None, "Please enter segment configuration"

    try:
        # Parse segments
        segments = []
        for line in segments_text.strip().split('\n'):
            parts = [float(x.strip()) for x in line.split(',')]
            if len(parts) < 2:
                return None, f"Invalid segment format: {line}"

            segment = {
                "start": parts[0],
                "end": parts[1],
            }

            if len(parts) >= 4:
                segment["video_fadein"] = parts[2] if parts[2] > 0 else None
                segment["video_fadeout"] = parts[3] if parts[3] > 0 else None

            if len(parts) >= 6:
                segment["audio_fadein"] = parts[4] if parts[4] > 0 else None
                segment["audio_fadeout"] = parts[5] if parts[5] > 0 else None

            segments.append(segment)

        # Create output file
        output_path = tempfile.mktemp(suffix=".mp4")

        # Progress callback
        def on_progress(current, total):
            progress((current, total), desc=f"Processing {current}/{total} segments")

        # Process
        success = smartcut.process_segments(
            input_path=input_video,
            output_path=output_path,
            segments=segments,
            quality=quality.lower(),
            progress_callback=on_progress,
        )

        if success:
            return output_path, f"✓ Processing completed successfully! ({len(segments)} segments)"
        else:
            return None, "✗ Processing failed - check console for errors"

    except Exception as e:
        return None, f"✗ Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="SmartCut Video Processor") as demo:
    gr.Markdown("# SmartCut Video Processor")
    gr.Markdown(
        "Smart video cutting with minimal re-encoding and independent video/audio fade effects."
    )

    with gr.Tabs():
        # Tab 1: Single Segment
        with gr.Tab("Single Segment"):
            gr.Markdown("### Process a single video segment with optional fade effects")

            with gr.Row():
                with gr.Column():
                    input_video1 = gr.Video(label="Input Video", sources=["upload"])

                    with gr.Row():
                        start_time = gr.Number(label="Start Time (seconds)", value=0, minimum=0)
                        end_time = gr.Number(label="End Time (seconds, 0=end)", value=0, minimum=0)

                    gr.Markdown("**Video Fade Effects**")
                    with gr.Row():
                        video_fadein = gr.Number(label="Video Fade-In (seconds)", value=0, minimum=0)
                        video_fadeout = gr.Number(label="Video Fade-Out (seconds)", value=0, minimum=0)

                    gr.Markdown("**Audio Fade Effects**")
                    with gr.Row():
                        audio_fadein = gr.Number(label="Audio Fade-In (seconds)", value=0, minimum=0)
                        audio_fadeout = gr.Number(label="Audio Fade-Out (seconds)", value=0, minimum=0)

                    quality1 = gr.Dropdown(
                        choices=["Low", "Normal", "High", "Lossless"],
                        value="Normal",
                        label="Video Quality"
                    )

                    process_btn1 = gr.Button("Process Video", variant="primary")

                with gr.Column():
                    output_video1 = gr.Video(label="Output Video")
                    status1 = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("""
            **Tips:**
            - Set fade values to 0 to disable fading for that media type
            - Video-only fades are ~17x faster than audio fades
            - Audio-only fades keep video in passthrough mode (faster)
            """)

            process_btn1.click(
                fn=process_single_segment,
                inputs=[
                    input_video1, start_time, end_time,
                    video_fadein, video_fadeout,
                    audio_fadein, audio_fadeout,
                    quality1
                ],
                outputs=[output_video1, status1]
            )

        # Tab 2: Multiple Segments
        with gr.Tab("Multiple Segments"):
            gr.Markdown("### Process multiple segments with independent fade configurations")

            with gr.Row():
                with gr.Column():
                    input_video2 = gr.Video(label="Input Video", sources=["upload"])

                    segments_text = gr.Textbox(
                        label="Segment Configuration (one per line)",
                        placeholder="start,end,video_fadein,video_fadeout,audio_fadein,audio_fadeout\n"
                                   "10,20,2,3,2,3\n30,40,2,0,0,2\n50,60,0,0,1,1.5",
                        lines=10
                    )

                    quality2 = gr.Dropdown(
                        choices=["Low", "Normal", "High", "Lossless"],
                        value="Normal",
                        label="Video Quality"
                    )

                    process_btn2 = gr.Button("Process Video", variant="primary")

                with gr.Column():
                    output_video2 = gr.Video(label="Output Video")
                    status2 = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("""
            **Segment Format:**
            ```
            start,end[,video_fadein,video_fadeout,audio_fadein,audio_fadeout]
            ```

            **Examples:**
            - `10,20,2,3,2,3` - Both fades (2s in, 3s out)
            - `30,40,2,3,0,0` - Video fade only
            - `50,60,0,0,1,2` - Audio fade only
            - `70,80` - No fades
            """)

            process_btn2.click(
                fn=process_multiple_segments,
                inputs=[input_video2, segments_text, quality2],
                outputs=[output_video2, status2]
            )

    gr.Markdown("""
    ---
    **About SmartCut:**
    - Only re-encodes around cut points and fade regions (fast!)
    - Independent video/audio fade control
    - GOP-aligned fades (may start/end up to ~1s early/late)
    - [GitHub](https://github.com/skeskinen/smartcut) | [PyPI](https://pypi.org/project/smartcut)
    """)


if __name__ == "__main__":
    demo.launch()
