#!/usr/bin/env python3
"""
Simple entry point for violence detection.

Usage:
    1. Place a video file in the input/ directory (or specify path in input/input.txt)
    2. Run: python main.py

Results will be saved to output/ directory:
    - violence_segments.json: Detected violence timestamps
    - violence_segments.txt: Human-readable summary
    - {video_name}_annotated.mp4: Video with violence segments highlighted
"""

import sys
import argparse
from pathlib import Path


def main():
    """Run violence detection with default paths: input/ → process → output/"""
    
    # Import here to avoid issues if package not installed
    from src.inference.detect import (
        resolve_input_video,
        collect_frame_scores,
        build_segments,
        write_annotated_video,
        save_segments,
    )
    from src.models.violence_detection import (
        ModelConfig,
        load_violence_model,
        resolve_device,
    )
    
    # Define paths (relative to project root)
    input_dir = Path("input")
    output_dir = Path("output")
    model_path = Path("artifacts/best_vgg19_lstm_kaggle.pt")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("🎥 VIOLENCE DETECTION - Real-Time Analysis Pipeline")
    print("=" * 70)
    
    # Configuration
    config = ModelConfig(image_size=(160, 160), sequence_length=40)
    device = resolve_device()  # Auto-detect GPU if available, else use CPU
    threshold = 0.80
    stride = 5
    
    print(f"\n📁 Input directory:  {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🤖 Device: {device}")
    print(f"📊 Violence threshold: {threshold}")
    
    # Find and load video
    print("\n🎬 Searching for video in input/...")
    try:
        input_video = resolve_input_video(input_dir)
        print(f"✅ Found: {input_video.name}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    
    # Load model
    print(f"\n🧠 Loading trained model...")
    try:
        model = load_violence_model(model_path, config=config, device=device)
        print(f"✅ Model loaded ({model_path.name})")
    except FileNotFoundError:
        print(f"❌ Model not found at {model_path}")
        print("   Train the model first or ensure the path is correct")
        return 1
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1
    
    # Run inference
    print(f"\n⚙️  Running violence detection...")
    print("   (This may take a few minutes depending on video length...)")
    
    try:
        analysis = collect_frame_scores(
            video_path=input_video,
            model=model,
            config=config,
            threshold=threshold,
            stride=stride,
        )
        print(f"✅ Inference complete: {analysis['total_frames']} frames processed")
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return 1
    
    # Process results
    print(f"\n📊 Building violence segments...")
    frame_scores = analysis["frame_scores"]
    fps = float(analysis["fps"])
    width = int(analysis["width"])
    height = int(analysis["height"])
    
    segments = build_segments(
        frame_scores=frame_scores,
        fps=fps,
        threshold=threshold,
        min_segment_seconds=0.5,
        merge_gap_seconds=1.0,
    )
    print(f"✅ Found {len(segments)} violence segment(s)")
    
    # Save annotated video
    print(f"\n🎞️  Creating annotated video...")
    try:
        output_video_path = output_dir / f"{input_video.stem}_annotated.mp4"
        write_annotated_video(
            video_path=input_video,
            output_path=output_video_path,
            frame_scores=frame_scores,
            threshold=threshold,
            fps=fps,
            width=width,
            height=height,
        )
        print(f"✅ Annotated video saved: {output_video_path.name}")
    except Exception as e:
        print(f"⚠️  Warning: Could not create annotated video: {e}")
    
    # Save segment data
    print(f"\n💾 Saving detection results...")
    try:
        save_segments(segments, output_dir)
        print(f"✅ Results saved:")
        print(f"   - violence_segments.json (machine-readable)")
        print(f"   - violence_segments.txt (human-readable)")
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return 1
    
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nOutput files in {output_dir}:")
    for f in sorted(output_dir.glob("*")):
        if f.is_file():
            size = f.stat().st_size
            if size > 1024*1024:
                print(f"  • {f.name} ({size / (1024*1024):.1f} MB)")
            elif size > 1024:
                print(f"  • {f.name} ({size / 1024:.1f} KB)")
            else:
                print(f"  • {f.name} ({size} bytes)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
