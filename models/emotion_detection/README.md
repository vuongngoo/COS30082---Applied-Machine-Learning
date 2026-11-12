# Emotion Recognition — MobileNetV3

Six-class facial-expression recognition with a local Windows webcam demo. Labels: **angry, fear, happy, sad, surprise, neutral**. FER2013's `disgust` class is excluded.

## Run the webcam

Install Python and open PowerShell in the repository root. Checked locally on Windows with Python 3.14, PyTorch 2.11.0 CPU, and torchvision 0.26.0.

```powershell
cd models/emotion_detection
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run_emotion_webcam.cmd --check
.\run_emotion_webcam.cmd
```

`--check` verifies model loading, preprocessing, face detection, and inference without opening the camera. Training and dataset downloads are not needed for the webcam demo. Keep the `.pt` and `.onnx` files beside `webcam_emotion.py`.

Press **Q**, **Escape**, or close the window to stop. Frames are processed locally and are not recorded. Close other camera apps if the camera cannot open. Select another camera with:

```powershell
.\run_emotion_webcam.cmd --camera 1
```

The checkpoint provides the default display threshold, **0.95**. Predictions below it remain visible with `?` and an amber box; accepted predictions use a green box. Override the display threshold with:

```powershell
.\run_emotion_webcam.cmd --threshold 0.90
```

Labels, confidence scores, and probability bars update every **0.5 seconds** using averaged predictions per spatially matched face. Video and boxes update every frame. History resets when a face disappears or its crop is unavailable. Matching uses box overlap, so crossing faces can still be confused. Confidence is a model score, not a calibrated probability.

## Files

| File | Purpose |
| --- | --- |
| `emotion_mobilenetv3_colab.ipynb` | Latest Colab notebook: six-block E5 experiment with saved outputs. |
| `emotion_mobilenetv3_large_6class_partial_last3.pt` | Three-block E3 checkpoint used by the webcam by default. |
| `webcam_emotion.py` | Face detection, emotion inference, and smoothed live display. |
| `face_detection_yunet_2023mar.onnx` | OpenCV YuNet face detector. |
| `run_emotion_webcam.cmd` | Windows launcher using this folder's `.venv`. |
| `requirements.txt` | Additional webcam dependencies; install CPU PyTorch as above. |
| `results/` | Exported metadata and evaluation results for the installed E3 checkpoint. |

## Notebook training

Open `emotion_mobilenetv3_colab.ipynb` in Google Colab and select a GPU runtime. Configuration is in **section 2**. The committed notebook uses `RUN_MODE = 'train'`, `UNFREEZE_LAST_BLOCKS = 6`, `COMPARE_SMALL_MODELS = False`, and label smoothing 0.05. It runs MobileNetV3-Large with weighted loss as E5, with three classifier-only warm-up epochs followed by up to 25 partial fine-tuning epochs. Earlier feature parameters and BatchNorm buffers remain frozen.

Use `RUN_MODE = 'smoke'` for a short pipeline check. For training, supply FER2013 through the acquisition options in section 3 and run through section 8 for validation selection. Run sections 9–11 for test evaluation and export. `RUN_MODE = 'evaluate'` reloads matching completed runs from Drive. The default run directory is `MyDrive/emotion_mobilenetv3_6class_partial_last6_ls005/`.

Saved E5 notebook outputs report best epoch 12, test accuracy **64.28%**, macro-F1 **62.58%**, and display threshold **0.80**. The E5 six-block checkpoint is not included here. The webcam uses the separate E3 three-block checkpoint below. Changing notebook settings does not change an existing checkpoint.

To use a newly exported model, place it beside the script and pass its filename:

```powershell
.\run_emotion_webcam.cmd --weights .\emotion_mobilenetv3_large_6class_partial_last6.pt
```

## Installed webcam checkpoint

The supplied checkpoint is **E3, MobileNetV3-Large**, initialized from ImageNet V2 weights. Feature blocks 0–13 remained frozen; blocks 14–16 and the classifier were fine-tuned after warm-up. It uses weighted loss without label smoothing, with best epoch 26.

- Audited folder-based FER2013: 23,810 train / 2,754 validation / 6,957 test images after exclusions and duplicate handling.
- Reported test accuracy: **63.69%**; macro-F1: **62.23%**.
- Validation-selected threshold: **0.95**. Reported test accepted accuracy: **80.87%** at **52.67%** coverage, applying only to accepted predictions.
- Exported file SHA-256: `668ee3d998d7af18df23fc1eec3e2cd3ef7404e4f87d15b864e3b3e9eab81e7e`.

Metrics come from exported training results and were not recomputed locally. They describe individual crops before live display averaging. The hash in `results/test_metrics.json` identifies the evaluated training checkpoint before export metadata was added, so it differs from the exported file hash above.

FER2013 has low-resolution grayscale images and noisy labels; webcam images differ in domain. Person-disjoint splits cannot be verified because FER2013 does not provide person IDs. The labels describe predicted facial expressions. The `.pt` is an inference checkpoint; retain the original Drive run directory for interrupted-training resumption.
