from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List

import cv2
import numpy as np

from src.models.violence_detection import (
    ModelConfig,
    load_violence_model,
    predict_violence_score,
    preprocess_frame,
    resolve_device,
)


VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".mpeg", ".mpg", ".m4v"}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_model = script_dir / "artifacts" / "rwf2000_vgg19_lstm.pt"

    parser = argparse.ArgumentParser(
        description=(
            "Run local violence detection on a video using the trained "
            "PyTorch VGG19 + LSTM model."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=script_dir / "input",
        help="Directory containing either a video file or input.txt with a video path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "output",
        help="Directory where annotated outputs and timestamps will be saved.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=default_model,
        help="Path to a trained PyTorch checkpoint such as .pt or .pth.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device to use, for example cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Violence confidence threshold.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Run the model every N frames once the clip buffer is full.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=40,
        help="Number of frames per clip. This must match training.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=160,
        help="Square resize dimension used by the trained model.",
    )
    parser.add_argument(
        "--merge-gap-seconds",
        type=float,
        default=1.0,
        help="Merge nearby violent detections separated by small gaps.",
    )
    parser.add_argument(
        "--min-segment-seconds",
        type=float,
        default=0.5,
        help="Discard detections shorter than this duration.",
    )
    return parser.parse_args()


def resolve_input_video(input_dir: Path) -> Path:
    """
    Resolve the video path from the input directory.

    Supported input styles:
    - `input/input.txt` containing a path to the real video file
    - any video file placed directly inside the input directory
    """

    input_dir = input_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    input_txt = input_dir / "input.txt"
    if input_txt.exists():
        content = input_txt.read_text(encoding="utf-8").strip()
        lines = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("#")]
        if lines:
            video_path = Path(lines[0])
            if not video_path.is_absolute():
                video_path = (input_dir / video_path).resolve()
            if video_path.exists():
                return video_path
            raise FileNotFoundError(f"Video path listed in input.txt was not found: {video_path}")

    candidates = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "No input video found. Put a video file in the input directory or "
        "write its path inside input/input.txt."
    )


def ensure_frame_scores_capacity(frame_scores: np.ndarray, frame_index: int) -> np.ndarray:
    """Extend frame score storage when a video does not expose a reliable frame count."""

    if frame_index < len(frame_scores):
        return frame_scores

    extra = max(1024, frame_index - len(frame_scores) + 1)
    return np.pad(frame_scores, (0, extra), mode="constant")


def collect_frame_scores(
    video_path: Path,
    model,
    config: ModelConfig,
    threshold: float,
    stride: int,
) -> Dict[str, object]:
    """
    Read the video once, run sliding-window inference, and assign a score to each frame.

    A frame is considered violent if it belongs to at least one window whose score
    crosses the threshold.
    """

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps > 0 else 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    estimated_total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frame_scores = np.zeros(max(estimated_total_frames, 1), dtype=np.float32)
    frame_buffer: Deque[np.ndarray] = deque(maxlen=config.sequence_length)

    frame_index = -1
    windows_evaluated = 0
    violent_windows = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        frame_index += 1
        frame_scores = ensure_frame_scores_capacity(frame_scores, frame_index)
        frame_buffer.append(preprocess_frame(frame, config.image_size))

        if len(frame_buffer) < config.sequence_length:
            continue

        start_frame = frame_index - config.sequence_length + 1
        if start_frame % stride != 0:
            continue

        clip_frames = np.stack(frame_buffer, axis=0)
        score = predict_violence_score(model, clip_frames)
        windows_evaluated += 1

        if score < threshold:
            continue

        violent_windows += 1
        end_frame = frame_index + 1
        frame_scores[start_frame:end_frame] = np.maximum(frame_scores[start_frame:end_frame], score)

    capture.release()

    actual_total_frames = frame_index + 1
    frame_scores = frame_scores[:actual_total_frames]

    return {
        "frame_scores": frame_scores,
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": actual_total_frames,
        "windows_evaluated": windows_evaluated,
        "violent_windows": violent_windows,
    }


def frame_to_seconds(frame_index: int, fps: float) -> float:
    return frame_index / fps if fps > 0 else 0.0


def seconds_to_timestamp(seconds: float) -> str:
    total_milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def build_segments(
    frame_scores: np.ndarray,
    fps: float,
    threshold: float,
    min_segment_seconds: float,
    merge_gap_seconds: float,
) -> List[Dict[str, object]]:
    """Convert frame-level scores into merged time segments."""

    raw_segments: List[Dict[str, object]] = []
    active_start = None

    for frame_index, score in enumerate(frame_scores):
        is_active = score >= threshold

        if is_active and active_start is None:
            active_start = frame_index
        elif not is_active and active_start is not None:
            raw_segments.append(
                {
                    "start_frame": active_start,
                    "end_frame": frame_index - 1,
                }
            )
            active_start = None

    if active_start is not None:
        raw_segments.append(
            {
                "start_frame": active_start,
                "end_frame": len(frame_scores) - 1,
            }
        )

    if not raw_segments:
        return []

    merge_gap_frames = int(round(merge_gap_seconds * fps))
    merged_segments = [raw_segments[0].copy()]

    for segment in raw_segments[1:]:
        previous = merged_segments[-1]
        gap = segment["start_frame"] - previous["end_frame"] - 1
        if gap <= merge_gap_frames:
            previous["end_frame"] = segment["end_frame"]
        else:
            merged_segments.append(segment.copy())

    final_segments: List[Dict[str, object]] = []
    for segment in merged_segments:
        start_frame = int(segment["start_frame"])
        end_frame = int(segment["end_frame"])
        duration_seconds = frame_to_seconds(end_frame - start_frame + 1, fps)
        if duration_seconds < min_segment_seconds:
            continue

        max_score = float(frame_scores[start_frame : end_frame + 1].max())
        start_seconds = frame_to_seconds(start_frame, fps)
        end_seconds = frame_to_seconds(end_frame + 1, fps)

        final_segments.append(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_seconds": round(start_seconds, 3),
                "end_seconds": round(end_seconds, 3),
                "start_timestamp": seconds_to_timestamp(start_seconds),
                "end_timestamp": seconds_to_timestamp(end_seconds),
                "duration_seconds": round(end_seconds - start_seconds, 3),
                "max_score": round(max_score, 4),
            }
        )

    return final_segments


def create_video_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """Create an MP4 writer for the annotated output video."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise ValueError(f"Unable to create output video: {output_path}")
    return writer


def draw_detection_overlay(frame: np.ndarray, score: float, timestamp: str) -> np.ndarray:
    """Draw a simple alert banner on violent frames."""

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (min(frame.shape[1] - 10, 700), 110), (0, 0, 255), thickness=-1)
    frame = cv2.addWeighted(overlay, 0.28, frame, 0.72, 0)

    cv2.putText(
        frame,
        "Violence Detected",
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Score: {score:.2f}",
        (25, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Time: {timestamp}",
        (25, 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def write_annotated_video(
    video_path: Path,
    output_path: Path,
    frame_scores: np.ndarray,
    threshold: float,
    fps: float,
    width: int,
    height: int,
) -> None:
    """Create a copy of the source video with violent frames annotated."""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to reopen video for annotation: {video_path}")

    writer = create_video_writer(output_path, fps, width, height)
    frame_index = -1

    while True:
        success, frame = capture.read()
        if not success:
            break

        frame_index += 1
        if frame_index < len(frame_scores) and frame_scores[frame_index] >= threshold:
            frame = draw_detection_overlay(
                frame,
                score=float(frame_scores[frame_index]),
                timestamp=seconds_to_timestamp(frame_to_seconds(frame_index, fps)),
            )
        writer.write(frame)

    capture.release()
    writer.release()


def save_segments(segments: List[Dict[str, object]], output_dir: Path) -> None:
    """Save human-readable and JSON summaries of the detected segments."""

    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "violence_segments.json"
    json_path.write_text(json.dumps(segments, indent=2), encoding="utf-8")

    text_lines = []
    if not segments:
        text_lines.append("No violence segments were detected.")
    else:
        for index, segment in enumerate(segments, start=1):
            text_lines.append(
                (
                    f"{index}. {segment['start_timestamp']} -> {segment['end_timestamp']} | "
                    f"duration={segment['duration_seconds']:.3f}s | max_score={segment['max_score']:.4f}"
                )
            )

    (output_dir / "violence_segments.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = ModelConfig(
        image_size=(args.image_size, args.image_size),
        sequence_length=args.sequence_length,
    )
    device = resolve_device(args.device)

    input_video = resolve_input_video(args.input_dir)
    model = load_violence_model(args.model_path, config=config, device=device)

    analysis = collect_frame_scores(
        video_path=input_video,
        model=model,
        config=config,
        threshold=args.threshold,
        stride=args.stride,
    )

    frame_scores = analysis["frame_scores"]
    fps = float(analysis["fps"])
    width = int(analysis["width"])
    height = int(analysis["height"])

    if width <= 0 or height <= 0:
        raise ValueError("Unable to determine the input video's width and height.")

    segments = build_segments(
        frame_scores=frame_scores,
        fps=fps,
        threshold=args.threshold,
        min_segment_seconds=args.min_segment_seconds,
        merge_gap_seconds=args.merge_gap_seconds,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_video_path = output_dir / f"{input_video.stem}_annotated.mp4"
    write_annotated_video(
        video_path=input_video,
        output_path=output_video_path,
        frame_scores=frame_scores,
        threshold=args.threshold,
        fps=fps,
        width=width,
        height=height,
    )
    save_segments(segments, output_dir)

    print(f"Input video: {input_video}")
    print(f"Annotated output video: {output_video_path}")
    print(f"Segments summary: {output_dir / 'violence_segments.txt'}")
    print(f"Segments JSON: {output_dir / 'violence_segments.json'}")
    print(f"Device: {device}")
    print(f"Frames analysed: {analysis['total_frames']}")
    print(f"Windows evaluated: {analysis['windows_evaluated']}")
    print(f"Violent windows: {analysis['violent_windows']}")
    print(f"Detected segments: {len(segments)}")


if __name__ == "__main__":
    main()
