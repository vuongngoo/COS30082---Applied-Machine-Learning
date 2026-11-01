"""Run the trained hand gesture detector on a local webcam."""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(__file__).resolve().parent / '.ultralytics'))

import cv2
import numpy as np
from ultralytics import YOLO


CLASS_NAMES = [
    'call', 'dislike', 'fist', 'four', 'like', 'mute', 'no_gesture',
    'ok', 'one', 'palm', 'peace', 'peace_inverted', 'rock', 'stop',
    'stop_inverted', 'three', 'three2', 'two_up', 'two_up_inverted',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=Path(__file__).with_name('best.pt'))
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.60)
    parser.add_argument('--check', action='store_true', help='Check loading and inference without opening the camera.')
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f'Checkpoint not found: {args.weights}')
    if not 0 <= args.conf <= 1 or args.imgsz <= 0:
        parser.error('Confidence must be between 0 and 1; image size must be positive.')

    model = YOLO(str(args.weights))
    if list(model.names.values()) != CLASS_NAMES:
        parser.error('Checkpoint does not have the expected 19 gesture classes.')
    checkpoint = model.ckpt or {}
    print('Checkpoint:', args.weights.resolve())
    print('Saved date:', checkpoint.get('date', 'unknown'))
    epoch = checkpoint.get('epoch', -1)
    print('Completed epochs:', epoch + 1 if epoch >= 0 else 'not retained in the exported checkpoint')
    print('Confidence:', args.conf, '| Image size:', args.imgsz)
    options = dict(conf=args.conf, iou=0.70, imgsz=args.imgsz, verbose=False)
    # Warm up inference before starting the camera or timing its frames.
    sample = np.zeros((480, 640, 3), dtype=np.uint8)
    result = model.predict(sample, **options)[0]
    if args.check:
        assert tuple(result.orig_shape) == sample.shape[:2]
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            assert 0 <= x1 <= x2 <= 640 and 0 <= y1 <= y2 <= 480
            assert 0 <= float(box.conf.item()) <= 1
        print('Model loading, class mapping, and inference checks passed.')
        print('Checkpoint validation metrics:', json.dumps(checkpoint.get('train_metrics', {}), default=str))
        return

    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    try:
        if not camera.isOpened():
            raise RuntimeError('Cannot open camera. Close other camera apps, check Windows camera permissions, or try --camera 1.')
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print('Press Q or Escape, or close the window, to stop. Frames are not recorded.')
        window = 'Hand gesture detection'
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        smoothed_fps = None
        while True:
            started = time.perf_counter()
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError('Camera stopped delivering frames.')
            # OpenCV arrays are BGR; Ultralytics handles their conversion internally.
            result = model.predict(frame, **options)[0]
            annotated = result.plot()
            fps = 1 / max(time.perf_counter() - started, 1e-6)
            smoothed_fps = fps if smoothed_fps is None else .9 * smoothed_fps + .1 * fps
            cv2.putText(annotated, f'{smoothed_fps:.1f} processing FPS | Q: quit', (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 0), 2)
            cv2.imshow(window, annotated)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped.')
