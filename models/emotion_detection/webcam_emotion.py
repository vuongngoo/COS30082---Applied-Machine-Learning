"""Six-class emotion webcam."""

import argparse
import hashlib
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small


CLASS_NAMES = ['angry', 'fear', 'happy', 'sad', 'surprise', 'neutral']
SOURCE_SIZE = 48
INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MIN_CROP_SIDE = 16
PREPROCESSING = dict(
    input='one RGB uint8 face crop, shape (H, W, 3); convert OpenCV BGR to RGB once in the camera adapter',
    grayscale='cv2.COLOR_RGB2GRAY, replicated to three channels',
    square='symmetric edge-replicated padding (cv2.BORDER_REPLICATE)',
    source_size=SOURCE_SIZE, downscale='cv2.INTER_AREA (cv2.INTER_LINEAR when enlarging)',
    input_size=INPUT_SIZE, upscale='bilinear, align_corners=False', center_crop=False,
    scale='[0, 1]', mean=IMAGENET_MEAN, std=IMAGENET_STD)
MEAN = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
STD = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
YUNET_URL = ('https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/'
             'face_detection_yunet_2023mar.onnx')
YUNET_SHA256 = '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'
HERE = Path(__file__).resolve().parent


def face_to_gray48(face):

    assert isinstance(face, np.ndarray) and face.dtype == np.uint8
    assert face.ndim == 2 or (face.ndim == 3 and face.shape[2] == 3), face.shape
    gray = face if face.ndim == 2 else cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    if height != width:
        pad = abs(height - width)
        before, after = pad // 2, pad - pad // 2
        borders = (before, after, 0, 0) if height < width else (0, 0, before, after)
        gray = cv2.copyMakeBorder(gray, *borders, cv2.BORDER_REPLICATE)
    if gray.shape[0] != SOURCE_SIZE:
        enlarge = gray.shape[0] < SOURCE_SIZE
        gray = cv2.resize(gray, (SOURCE_SIZE, SOURCE_SIZE),
                          interpolation=cv2.INTER_LINEAR if enlarge else cv2.INTER_AREA)
    return gray


def gray48_to_input(batch):

    x = F.interpolate(batch.float().div(255), size=(INPUT_SIZE, INPUT_SIZE), mode='bilinear', align_corners=False)
    x = x.expand(-1, 3, -1, -1)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def build_model(arch):
    model = {'small': mobilenet_v3_small, 'large': mobilenet_v3_large}[arch](weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASS_NAMES))
    return model


class EmotionClassifier:


    def __init__(self, checkpoint_path, device='cpu', threshold=None):
        payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        assert payload['format'] == 'emotion-mobilenetv3-v1'
        assert payload['class_names'] == CLASS_NAMES and payload['num_classes'] == len(CLASS_NAMES), (
            'Expected a six-class partial emotion checkpoint without disgust; retrain with the updated notebook.')
        assert payload['preprocessing'] == PREPROCESSING, 'Checkpoint was trained with different preprocessing.'
        self.payload = payload
        self.device = torch.device(device)
        self.model = build_model(payload['arch'])
        self.model.load_state_dict(payload['state_dict'])
        self.model.to(self.device).eval()
        self.threshold = payload.get('display_threshold') if threshold is None else threshold

    @torch.inference_mode()
    def predict_emotions(self, face_crops_rgb):

        results, valid = [None] * len(face_crops_rgb), []
        for i, face in enumerate(face_crops_rgb):
            usable = (isinstance(face, np.ndarray) and face.dtype == np.uint8 and face.ndim == 3
                      and face.shape[2] == 3 and min(face.shape[:2]) >= MIN_CROP_SIDE)
            if usable:
                valid.append(i)
            else:
                results[i] = dict(label=None, confidence=None, probabilities=None, status='unavailable')
        if valid:
            batch = torch.from_numpy(np.stack([face_to_gray48(face_crops_rgb[i]) for i in valid]))[:, None]
            probabilities = self.model(gray48_to_input(batch.to(self.device))).float().softmax(1).cpu().numpy()
            for i, p in zip(valid, probabilities):
                k = int(p.argmax())
                confidence = float(p[k])

                ok = self.threshold is not None and confidence >= self.threshold
                results[i] = dict(label=CLASS_NAMES[k], confidence=confidence,
                                  probabilities=[float(v) for v in p], status='ok' if ok else 'uncertain')
        return results


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_detector(path, download):
    if not path.is_file():
        if not download:
            raise FileNotFoundError(f'Face detector not found: {path}. Rerun with --download-detector '
                                    f'or place face_detection_yunet_2023mar.onnx there ({YUNET_URL}).')
        print('Downloading YuNet face detector from', YUNET_URL)
        temporary = path.with_suffix('.download')
        urllib.request.urlretrieve(YUNET_URL, temporary)
        if sha256_file(temporary) != YUNET_SHA256:
            temporary.unlink()
            raise RuntimeError('Downloaded detector checksum mismatch.')
        temporary.replace(path)
    if sha256_file(path) != YUNET_SHA256:
        print('Warning: detector file differs from the pinned YuNet 2023mar checksum.')
    return cv2.FaceDetectorYN.create(str(path), '', (320, 320), 0.8, 0.3, 5000)


def detect_faces(detector, frame_bgr):

    height, width = frame_bgr.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame_bgr)
    boxes = []
    for face in faces if faces is not None else []:
        x, y, w, h = face[:4]
        x1, y1 = max(0, int(round(x))), max(0, int(round(y)))
        x2, y2 = min(width, int(round(x + w))), min(height, int(round(y + h)))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


class EmotionDisplaySmoother:
    def __init__(self, threshold):
        self.threshold = threshold
        self.tracks = []

    def update(self, boxes, results, now):
        pairs = sorted((box_iou(box, track['box']), i, j)
                       for i, box in enumerate(boxes)
                       for j, track in enumerate(self.tracks)
                       if now - track['last_seen'] <= 0.75)
        matches, used = {}, set()
        for overlap, i, j in reversed(pairs):
            if overlap < 0.3:
                break
            if i not in matches and j not in used:
                matches[i] = j
                used.add(j)

        tracks, displayed = [], []
        for i, (box, result) in enumerate(zip(boxes, results)):
            if result['probabilities'] is None:
                displayed.append(result)
                continue
            track = self.tracks[matches[i]] if i in matches else dict(
                total=np.zeros(len(CLASS_NAMES)), count=0, shown_at=now, result=dict(result))
            track.update(box=box, last_seen=now)
            track['total'] += np.asarray(result['probabilities'])
            track['count'] += 1
            if now - track['shown_at'] >= 0.5:
                probabilities = track['total'] / track['count']
                k = int(probabilities.argmax())
                confidence = float(probabilities[k])
                accepted = self.threshold is not None and confidence >= self.threshold
                track['result'] = dict(label=CLASS_NAMES[k], confidence=confidence,
                                       probabilities=probabilities.tolist(),
                                       status='ok' if accepted else 'uncertain')
                track.update(total=np.zeros(len(CLASS_NAMES)), count=0, shown_at=now)
            tracks.append(track)
            displayed.append(track['result'])
        self.tracks = tracks
        return displayed


def draw(frame_bgr, boxes, results):
    for (x1, y1, x2, y2), result in zip(boxes, results):
        if result['status'] == 'unavailable':
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (128, 128, 128), 1)
            continue
        colour = (0, 200, 0) if result['status'] == 'ok' else (0, 200, 255)
        text = f'{result["label"].capitalize()} {result["confidence"]:.2f}'
        if result['status'] == 'uncertain':
            text += ' ?'
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame_bgr, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)

    if results and results[0]['probabilities'] is not None:
        for row, (name, p) in enumerate(zip(CLASS_NAMES, results[0]['probabilities'])):
            y = 56 + row * 20
            cv2.rectangle(frame_bgr, (12, y - 12), (12 + int(120 * p), y + 2), (255, 200, 0), -1)
            cv2.putText(frame_bgr, f'{name} {p:.2f}', (138, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--weights', type=Path, default=HERE / 'emotion_mobilenetv3_large_6class_partial_last3.pt')
    parser.add_argument('--detector', type=Path, default=HERE / 'face_detection_yunet_2023mar.onnx')
    parser.add_argument('--download-detector', action='store_true', help='Download YuNet if it is missing.')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override the display threshold stored in the checkpoint (0-1).')
    parser.add_argument('--check', action='store_true', help='Check loading and inference without opening the camera.')
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f'Checkpoint not found: {args.weights}')
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        parser.error('Threshold must be between 0 and 1.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classifier = EmotionClassifier(args.weights, device, args.threshold)
    payload = classifier.payload
    print('Checkpoint:', args.weights.resolve())
    print('Model:', payload.get('experiment'), payload['arch'], '| epoch', payload.get('epoch'),
          '| protocol:', payload.get('protocol'))
    print('Display threshold:', classifier.threshold, '(None: every prediction is shown as uncertain)')
    print('Device:', device)
    detector = load_detector(args.detector, args.download_detector)


    sample = np.full((480, 640, 3), 127, np.uint8)
    detect_faces(detector, sample)
    crops = [np.random.default_rng(i).integers(0, 256, (120, 100, 3), dtype=np.uint8) for i in range(3)]
    outputs = classifier.predict_emotions(crops + [np.zeros((8, 8, 3), np.uint8)])
    if args.check:
        assert classifier.predict_emotions([]) == []
        assert [r['status'] == 'unavailable' for r in outputs] == [False, False, False, True]
        for result in outputs[:3]:
            assert result['label'] in CLASS_NAMES and abs(sum(result['probabilities']) - 1) < 1e-4
        started = time.perf_counter()
        for _ in range(20):
            classifier.predict_emotions(crops[:1])
        print(f'Classifier latency, one face: {(time.perf_counter() - started) / 20 * 1000:.1f} ms mean')
        print('Model loading, preprocessing, detector, and inference checks passed.')
        return

    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    try:
        if not camera.isOpened():
            raise RuntimeError('Cannot open camera. Close other camera apps, check Windows camera permissions, or try --camera 1.')
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print('Press Q or Escape, or close the window, to stop. Frames are not recorded.')
        window = 'Emotion recognition'
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        smoothed_fps = None
        display_smoother = EmotionDisplaySmoother(classifier.threshold)
        while True:
            started = time.perf_counter()
            ok, frame_bgr = camera.read()
            if not ok:
                raise RuntimeError('Camera stopped delivering frames.')
            boxes = detect_faces(detector, frame_bgr)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = classifier.predict_emotions([frame_rgb[y1:y2, x1:x2] for x1, y1, x2, y2 in boxes])
            results = display_smoother.update(boxes, results, time.monotonic())
            draw(frame_bgr, boxes, results)
            fps = 1 / max(time.perf_counter() - started, 1e-6)
            smoothed_fps = fps if smoothed_fps is None else .9 * smoothed_fps + .1 * fps
            cv2.putText(frame_bgr, f'{smoothed_fps:.1f} processing FPS | {len(boxes)} face(s) | Q: quit', (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 0), 2)
            cv2.imshow(window, frame_bgr)
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
