# Emotion recognition model card

- **Model:** E3, MobileNetV3-Large initialized from MobileNet_V3_Large_Weights.IMAGENET1K_V2, last classifier layer replaced with 6 outputs.
- **Partial fine-tuning:** last 3 feature blocks and classifier trained after classifier-only warm-up; all earlier feature parameters and BatchNorm buffers remain frozen. Frozen feature blocks: 14; verified maximum frozen change: 0.0.
- **Data:** folder-based FER2013 protocol (audited); source https://www.kaggle.com/datasets/msambare/fer2013; SHA256 f3d407e59594dc0e36c23985b2c4139f83d8bfe8142f91b39be9e7f3fe00b611. Exclusions: {'near_blank': 13, 'duplicate': 1668, 'conflicting_duplicate': 138, 'excluded_disgust': 547}. Kept counts: {'train': {'angry': 3392, 'fear': 3398, 'happy': 6313, 'sad': 4200, 'surprise': 2174, 'neutral': 4333}, 'val': {'angry': 385, 'fear': 400, 'happy': 717, 'sad': 477, 'surprise': 287, 'neutral': 488}, 'test': {'angry': 941, 'fear': 1005, 'happy': 1764, 'sad': 1235, 'surprise': 788, 'neutral': 1224}}.
- **Label policy:** disgust (original FER2013 ID 1) excluded from train/validation/test; source IDs 0, 2, 3, 4, 5, 6 remapped to model IDs 0–5. Scores are for a six-class task and are not directly comparable with seven-class results. Disgust expressions are outside the trained label set; removing the class does not guarantee less flicker in webcam use.
- **Classes (ID order):** 0 angry, 1 fear, 2 happy, 3 sad, 4 surprise, 5 neutral.
- **Preprocessing:** RGB crop → grayscale → edge-replicated square → 48 × 48 (INTER_AREA) → bilinear 224 × 224 → 3 channels → ImageNet normalization. Live crops pass through 48 × 48 so they match the training resolution.
- **Selection:** highest validation macro-F1 among fine-tuning epochs (best epoch 26); E3 if validation macro-F1 >= E2 + 0.01 and CPU p95 <= 50.0 ms; otherwise E2.
- **Confidence:** maximum softmax score, not a calibrated probability. Display threshold 0.95, chosen on validation for at least 80% accepted accuracy (validation coverage 52.3%).

## Test results
- Test macro-F1 0.6223, accuracy 0.6369, balanced accuracy 0.6195 (n=6957).
- angry: precision 0.594, recall 0.546, F1 0.569, support 941
- fear: precision 0.509, recall 0.468, F1 0.487, support 1005
- happy: precision 0.839, recall 0.836, F1 0.837, support 1764
- sad: precision 0.495, recall 0.545, F1 0.519, support 1235
- surprise: precision 0.763, recall 0.732, F1 0.747, support 788
- neutral: precision 0.558, recall 0.590, F1 0.574, support 1224

## Runtime
- cpu, 1 face(s): p50 20.2 ms, p95 22.95 ms (classifier only)
- cpu, 3 face(s): p50 49.06 ms, p95 58.08 ms (classifier only)
- cpu, 5 face(s): p50 77.67 ms, p95 91.02 ms (classifier only)
- cuda, 1 face(s): p50 6.69 ms, p95 9.21 ms (classifier only)
- cuda, 3 face(s): p50 6.94 ms, p95 8.23 ms (classifier only)
- cuda, 5 face(s): p50 7.37 ms, p95 8.11 ms (classifier only)

## Limitations
- FER2013 is 48 × 48 grayscale with noisy labels and class imbalance (six-class subset excludes disgust); webcam crops differ in domain.
- FER2013 has no person IDs, so splits are not guaranteed subject-disjoint; results come from a single seed.
- Outputs are predicted facial-expression categories. They do not measure internal emotion, mental health, or employee sentiment, and they do not replace liveness or decide attendance.
