# Voilence Detection

This repository contains a refactored local violence-detection workflow. It keeps the original repository's idea of using a `VGG19 + LSTM` video classifier, but removes the API dependency and uses PyTorch for both training and inference.

## Included Files

- `rwf2000_training_vgg19_lstm_pytorch.ipynb`
  - trains the violence classifier on the `RWF-2000` dataset with PyTorch
- `violence_model.py`
  - shared PyTorch model architecture and model-loading utilities
- `detect_violence.py`
  - reads a local video, runs sliding-window inference, saves an annotated output video, and stores detected time ranges

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

Run the detector after you have a trained model:

```bash
python3 detect_violence.py --model-path artifacts/rwf2000_vgg19_lstm.pt
```

If you saved the best checkpoint instead, you can point to that file:

```bash
python3 detect_violence.py --model-path artifacts/best_vgg19_lstm.pt
```

## Notes

- The model input shape must match training. If you change sequence length or image size during training, use the same values during detection.
- The shared inference code now expects a PyTorch checkpoint, not a TensorFlow `.keras` artifact.
- The script uses sliding-window inference, so the final output is a sequence of detected time ranges rather than bounding boxes.
- The repository name still uses `Voilence` to match the current project naming, even though the standard spelling would be `Violence`.
