# Voilence Detection

This repository contains a refactored local violence-detection workflow. It keeps the original repository's idea of using a `VGG19 + LSTM` video classifier, but removes the API dependency and uses PyTorch for both training and inference.

## Included Files

- `main.py`
  - main entry point for violence detection inference
- `src/models/violence_detection.py`
  - PyTorch model architecture (VGG19 + LSTM) and utilities
- `src/inference/detect.py`
  - video processing, sliding-window inference, and output generation
- `notebooks/`
  - Jupyter notebooks for training on RWF-2000 dataset with PyTorch
- `artifacts/best_vgg19_lstm_kaggle.pt`
  - pre-trained model checkpoint

## Input Workflow

The detection script supports two ways of telling it which video to process:

1. Put a video directly inside `input/`
2. Write the video path inside `input/input.txt`

The first non-comment line in `input/input.txt` is treated as the source video path.

## Outputs

Running the detector creates:

- `output/<video_name>_annotated.mp4`
  - the source video with `Violence Detected` drawn on violent frames
- `output/violence_segments.txt`
  - plain-text timeframe summary
- `output/violence_segments.json`
  - machine-readable timeframe summary

## Example Run

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Prepare input:**
   - Place a video file in `input/` directory, OR
   - Specify the path in `input/input.txt`

3. **Run detection:**
```bash
python main.py
```

The script will automatically:
- ✅ Detect and use GPU (CUDA) if available for fast inference
- ✅ Fall back to CPU if GPU is not available
- ✅ Process the video and generate results in `output/`

## Notes

- The model is pre-trained on RWF-2000 (Real World Fighting 2000) dataset
- **GPU acceleration:** The code automatically detects CUDA availability. For faster processing, ensure your PyTorch installation includes CUDA support:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```
- The model input shape must match training. Current config: `image_size=(160, 160)`, `sequence_length=40`
- Sliding-window inference generates a sequence of detected violence time ranges rather than per-frame bounding boxes
- The repository name uses `Voilence` (misspelled) to match the original project naming
