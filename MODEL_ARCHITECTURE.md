# Violence Detection Model Architecture

## Overview

This project implements a **deep learning-based video violence detection system** using a **VGG19 + LSTM** hybrid architecture. The model processes video frames to identify violent actions in real-time and generates:
- Annotated video with violence overlays
- JSON and TXT files with precise timestamps of detected violence
- Confidence scores for each detection

---

## What We Do

### Input
- A video file (MP4, AVI, MOV, MKV, etc.)

### Processing
1. Extract frames from the video
2. Use VGG19 CNN to extract spatial features from each frame
3. Use LSTM to model temporal relationships across sequences of frames
4. Classify each sequence as violent or non-violent
5. Merge overlapping detections and filter by minimum duration

### Output
Three files saved to `output/`:
- **`{video_name}_annotated.mp4`** - Video with "Violence Detected" text overlay on violent frames
- **`violence_segments.json`** - Machine-readable violence timestamps and confidence scores
- **`violence_segments.txt`** - Human-readable summary of detected violence time ranges

---

## Model Architecture

### High-Level Design

```
Input Video
    ↓
[Frame Extraction] → 40 consecutive frames per clip
    ↓
[VGG19 Feature Extractor] → Extract spatial features per frame
    ↓
[LSTM Module] → Model temporal patterns across frames
    ↓
[Dense Classifier] → Classify as Violence or Non-Violence
    ↓
[Post-processing] → Generate time segments and merge nearby detections
    ↓
Output: Violence Timestamps + Confidence Scores
```

### Detailed Architecture

#### 1. **Input Specification**
```python
Input shape: (batch_size, sequence_length, 3, height, width)
            = (1, 40, 3, 160, 160)

- batch_size: 1 (process one clip at a time during inference)
- sequence_length: 40 frames (temporal context window)
- 3: RGB color channels
- height, width: 160×160 pixels (default)
```

#### 2. **Feature Extraction: VGG19 CNN**

The model uses a **pre-trained VGG19** (ImageNet weights) as the spatial feature extractor.

```
Input: (batch_size × sequence_length, 3, 160, 160)
       = Individual frames reshaped for parallel processing

VGG19 Architecture:
├── Conv blocks 1-5 (Convolutional layers)
│   ├── Multiple 3×3 convolutions
│   ├── ReLU activations
│   └── MaxPooling layers
│
└── Output: Flattened feature vector per frame
    Size: ~102,400 dimensions (varies with input size)
```

**Key points:**
- All VGG19 parameters are **frozen** (not trainable) during training
- Operates on each frame independently
- Extracts rich spatial information (edges, textures, objects)

#### 3. **Temporal Modeling: LSTM**

The LSTM processes the sequence of extracted frame features to capture temporal dynamics.

```
Input: (batch_size, sequence_length, feature_dim)
       = (1, 40, 102400)

LSTM Configuration:
├── Input size: 102,400 (VGG19 output dimension)
├── Hidden size: 40 (internal state dimension)
├── Batch first: True
│
└── Output: (batch_size, sequence_length, hidden_size)
            = (1, 40, 40)
```

**What LSTM learns:**
- Motion patterns between consecutive frames
- Temporal context of actions
- Whether observed patterns are consistent with violence

#### 4. **Classification Head: Dense Layers**

```
Frame Dense Layer:
├── Input: (batch_size, sequence_length, 40)
├── Output: (batch_size, sequence_length, 160)
└── Activation: ReLU

Temporal Pooling:
├── Average pool across time dimension
├── Input: (batch_size, sequence_length, 160)
└── Output: (batch_size, 160)

Classifier Hidden Layer:
├── Input: (batch_size, 160)
├── Output: (batch_size, 512)
├── Activation: ReLU
└── Dropout: 0.3 (30% during training)

Final Classifier:
├── Input: (batch_size, 512)
├── Output: (batch_size, 2) → [non-violence_logit, violence_logit]
└── Post-processing: Softmax → Probability scores
```

### Model Parameters Summary

```python
ModelConfig(
    image_size=(160, 160),           # Frame resolution
    sequence_length=40,               # Frames per clip
    num_classes=2,                    # Binary classification
    lstm_hidden_size=40,              # LSTM hidden state
    temporal_dense_size=160,          # Feature dimension after frame dense
    classifier_hidden_size=512,       # Classifier hidden layer
    dropout=0.3,                      # Regularization
    learning_rate=5e-4                # Training hyperparameter
)
```

---

## How Violence is Detected

### Step 1: Frame Preprocessing

Each video frame is converted and normalized:

```python
def preprocess_frame(frame_bgr, image_size=(160, 160)):
    1. Convert BGR (OpenCV) → RGB
    2. Resize to 160×160 pixels (match training)
    3. Normalize to [0, 1] range (divide by 255)
    4. Output: np.ndarray of shape (160, 160, 3)
```

### Step 2: Sliding Window Inference

Instead of classifying individual frames, we use **sliding windows** of 40 consecutive frames.

```
Video: [F0, F1, F2, F3, ..., F100, F101, ...]

Window 1: [F0-F39]   → Model → Score₁ (violence probability)
Window 2: [F5-F44]   → Model → Score₂ (stride=5)
Window 3: [F10-F49]  → Model → Score₃
...
```

**Why sliding windows?**
- Single frames contain insufficient context
- 40 frames (~1.6 seconds at 25 FPS) capture complete violent action
- Overlapping windows ensure no violence is missed

### Step 3: Frame Score Assignment

Each frame gets a **maximum confidence score** from all windows it belongs to:

```
Frame 20:
├── Belongs to Window 1: [F0-F39] → Score₁ = 0.85
├── Belongs to Window 2: [F5-F44] → Score₂ = 0.92
└── Final Score = max(0.85, 0.92) = 0.92
```

### Step 4: Threshold-Based Detection

A frame is marked as violent if its score exceeds the threshold (default: **0.80**):

```python
if frame_score >= 0.80:
    Mark frame as VIOLENT
else:
    Mark frame as NON-VIOLENT
```

### Step 5: Segment Building & Merging

Consecutive violent frames are grouped into segments:

```
Violent frames: [F10, F11, F12, F13, F20, F21, F22, F30]
                └─────────────┬──────────┘  └──────────┬──┘
                          Segment 1      Gap    Segment 2
                       (10-13, 4 frames)  (Segment 2 starts at F20)

Merging (if gap < 1 second):
- Segments within 1 second gap are merged
- Result: [F10-F22] merged segment
```

Filtering:
- Minimum segment duration: **0.5 seconds** (filters noise)
- Segments shorter than this are discarded

---

## Complete Inference Pipeline

### Code Flow

```python
# 1. Load video and model
video_path = "input/hospital.mp4"
model = load_violence_model(checkpoint_path)
config = ModelConfig()

# 2. Process entire video with sliding window
analysis = collect_frame_scores(
    video_path=video_path,
    model=model,
    config=config,
    threshold=0.80,      # Violence confidence threshold
    stride=5             # Evaluate every 5 frames
)

# 3. Generate time segments
segments = build_segments(
    frame_scores=analysis['frame_scores'],
    fps=analysis['fps'],
    threshold=0.80,
    merge_gap_seconds=1.0,
    min_segment_seconds=0.5
)

# 4. Create annotated video
write_annotated_video(
    video_path=video_path,
    frame_scores=analysis['frame_scores'],
    segments=segments,
    output_path="output/hospital_annotated.mp4",
    threshold=0.80
)

# 5. Save results
save_segments(segments, "output/violence_segments.json")
```

### Metrics Computed

During inference, the system tracks:

```python
{
    "frame_scores": array of 0-1 scores for each frame,
    "fps": frames per second of video,
    "width": video width in pixels,
    "height": video height in pixels,
    "total_frames": total frames in video,
    "windows_evaluated": number of sliding windows processed,
    "violent_windows": number of windows exceeding threshold
}
```

---

## Example Output

### JSON Format
```json
[
  {
    "start_frame": 240,
    "end_frame": 320,
    "start_seconds": 9.6,
    "end_seconds": 12.8,
    "start_timestamp": "00:00:09.600",
    "end_timestamp": "00:00:12.800",
    "duration_seconds": 3.2,
    "max_score": 0.9512
  },
  {
    "start_frame": 500,
    "end_frame": 620,
    "start_seconds": 20.0,
    "end_seconds": 24.8,
    "start_timestamp": "00:00:20.000",
    "end_timestamp": "00:00:24.800",
    "duration_seconds": 4.8,
    "max_score": 0.8734
  }
]
```

### Text Format
```
Violence Detection Results for: hospital.mp4
=====================================================
Total frames: 1500 | FPS: 25 | Duration: 60.0s
Windows evaluated: 298 | Violent windows: 47
Threshold: 0.80 | Min segment: 0.5s | Merge gap: 1.0s

Detected Violence Segments:
1. 00:00:09.600 - 00:00:12.800 (3.2s) - Confidence: 0.9512
2. 00:00:20.000 - 00:00:24.800 (4.8s) - Confidence: 0.8734
```

---

## Key Design Decisions

### 1. **Why VGG19?**
- Proven architecture for visual recognition
- Captures hierarchical features (edges → textures → objects)
- Frozen backbone ensures consistent feature extraction
- Pre-trained ImageNet weights provide domain knowledge

### 2. **Why LSTM?**
- Captures temporal dependencies between frames
- Can identify sustained actions (violence typically lasts 1-2 seconds)
- Better than simple averaging for understanding action sequences

### 3. **Sliding Windows vs. Single Frame**
- **Single frame:** Cannot distinguish context (punch vs. celebration?)
- **Sliding window:** Provides temporal context (sustained motion pattern)
- **40 frames:** Optimal duration to capture complete action (~1.6s @ 25 FPS)

### 4. **Post-Processing**
- **Merging nearby detections:** Handles temporal gaps in violence (e.g., fast cuts)
- **Minimum segment filtering:** Removes false positives (noise/flicker)
- **Confidence thresholding:** Controls false positive rate

---

## Performance Characteristics

### Inference Speed
- **GPU (CUDA):** 2-10 seconds for ~5-10 minute video
- **CPU:** 5-30+ minutes for same video
- Auto-detection: Uses GPU if available, falls back to CPU

### Memory Requirements
- Model weights: ~140 MB
- Batch processing: Minimal (processes 1 clip at a time)
- Suitable for both cloud and edge deployment

### Accuracy Metrics (on RWF-2000 dataset)
- Trained on **RWF-2000** (Real World Fighting 2000) dataset
- Classes: Violence vs. Non-violence
- Binary classification with confidence scores

---

## Configuration Parameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_size` | 160 | Input frame size (square) |
| `sequence_length` | 40 | Frames per clip |
| `threshold` | 0.80 | Minimum confidence for violence detection |
| `stride` | 5 | Evaluate every Nth frame (5 = skip 4 between evaluations) |
| `merge_gap_seconds` | 1.0 | Merge segments within this time gap |
| `min_segment_seconds` | 0.5 | Discard detections shorter than this |
| `dropout` | 0.3 | Regularization rate (30% during training) |
| `lstm_hidden_size` | 40 | LSTM internal dimension |
| `classifier_hidden_size` | 512 | Classification MLP hidden dimension |

---

## Device Support

The system automatically detects and uses the best available device:

```python
device = resolve_device()
# Returns:
# - torch.device("cuda") if GPU (NVIDIA with CUDA) available
# - torch.device("cpu") otherwise
```

To manually specify device:
```bash
python main.py --device cuda:0  # Use specific GPU
python main.py --device cpu     # Force CPU
```

---

## References

- **VGG19:** [Simonyan & Zisserman (2015)](https://arxiv.org/abs/1409.1556)
- **LSTM:** [Hochreiter & Schmidhuber (1997)](https://www.bioinspired.com/papers/Hochreiter97.pdf)
- **RWF-2000 Dataset:** [Cheng et al. (2020)](https://arxiv.org/abs/2006.14993)
