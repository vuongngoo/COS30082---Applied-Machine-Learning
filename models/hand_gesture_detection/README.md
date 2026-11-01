# Hand Gesture Detection — YOLO11n

Standalone hand gesture detection for COS30082 Applied Machine Learning. A COCO-pretrained YOLO11n was fine-tuned on the compact HaGRID dataset to detect **18 gestures and `no_gesture`**. The model returns a label, bounding box, and confidence for each detected hand.

## Architecture and workflow

The diagram covers the YOLO11n architecture, training and evaluation workflow, and local webcam inference.

[![YOLO11n hand gesture architecture, training, evaluation, and webcam workflow](assets/hand_gesture_architecture.png)](assets/hand_gesture_architecture.png)

Click the image to view it at full resolution, or [open the editable FigJam board](https://www.figma.com/board/0fasbqiB98iZlZ4Xhvlc52).

## Files

- `best.pt`: trained gesture checkpoint, ready for inference.
- `webcam_gesture.py`: local Windows webcam demo.
- `hand_gesture_yolo11n_colab.ipynb`: self-contained Colab workflow with retained training, evaluation, and example outputs.

## Run the webcam demo

Tested on **Windows with Python 3.14**, PyTorch 2.11.0 (CPU), and Ultralytics 8.4.193. Open a terminal in this folder; from the repository root, run `cd models/hand_gesture_detection` first.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install ultralytics==8.4.193
.\.venv\Scripts\python.exe webcam_gesture.py
```

Keep `best.pt` beside the script. Press **Q** or **Escape**, or close the window, to stop. Frames are processed locally and are not recorded. No dataset download or training is needed for this demo.

For a faster preview, use a smaller inference size:

```powershell
.\.venv\Scripts\python.exe webcam_gesture.py --imgsz 320
```

Smaller hands may be harder to detect at 320. Default inference size is 640, confidence is 0.60, and NMS IoU is 0.70. If the camera is unavailable, close other camera apps, check Windows camera permissions, or try `--camera 1`.

Check model loading without opening the webcam:

```powershell
.\.venv\Scripts\python.exe webcam_gesture.py --check
```

## Colab notebook

[Open in Google Colab](https://colab.research.google.com/github/vuongngoo/COS30082---Applied-Machine-Learning/blob/main/models/hand_gesture_detection/hand_gesture_yolo11n_colab.ipynb)

Select a GPU runtime and run the cells in order. Google Drive stores training checkpoints and exported results.

- `RUN_MODE = 'smoke'`: short pipeline check using a small subset and 2 epochs. This is the fresh-runtime default.
- `RUN_MODE = 'full'`: train for up to 50 epochs, then evaluate and export.
- `RUN_MODE = 'evaluate'`: load the checkpoint specified by `WEIGHTS` and evaluate without training.

Use `RESUME_FULL = True` only with an interrupted full run's `last.pt`. The notebook retains outputs from the completed full run; its defaults were reset to smoke mode for new users. The retained training log shows epochs 42–50 after resuming, rather than all 50 epochs in one runtime.

## Dataset and results

Dataset: [cj-mills/hagrid-sample-30k-384p](https://huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p), derived from [HaGRID](https://github.com/hukenovs/hagrid). The compact dataset card lists **CC BY-SA 4.0**. Dataset images retained in notebook outputs are attributed to these sources. The dataset archive is downloaded by the notebook and is not stored in this folder.

All **31,833 images** were used with subject-disjoint partitions: **24,997 train / 3,627 validation / 3,209 test**. The selected confidence of 0.60 was chosen on validation only. Test matching uses IoU 0.50 with one-to-one, class-agnostic matching; NMS IoU is separately fixed at 0.70.

| Test metric | All 19 classes | 18 gestures, excluding `no_gesture` |
| --- | ---: | ---: |
| Macro Precision | 98.82% | 99.15% |
| Macro Recall | 98.59% | 98.78% |
| Macro F1 | 98.70% | 98.96% |

Validation mAP50 was **99.4%** and mAP50–95 was **86.7%**. These mAP values are validation results, not test results. The confusion matrix adds a background/no-match category, giving 20 rows and columns; the model itself has 19 classes.

Supported labels:

```text
call, dislike, fist, four, like, mute, no_gesture, ok, one, palm,
peace, peace_inverted, rock, stop, stop_inverted, three, three2,
two_up, two_up_inverted
```

## Limitations

- Back-of-hand views of `three` / `four` may be confused with `peace_inverted` in webcam use.
- Small, occluded, or poorly lit hands can be missed or confused.
- `no_gesture` means a visible natural hand. It is not an empty frame; the fixed benchmark does not include a confirmed no-hand test set.
- Each frame is processed independently. Motion gestures, person identification, and attendance actions are not implemented.
- CPU speed depends on the computer and inference size. Benchmark scores do not guarantee the same performance in a different camera environment.

Model SHA-256:

```text
61577770b0ed2e5517912f447200a33d5f9c9527e626a7ffba02f19a78e258df
```
