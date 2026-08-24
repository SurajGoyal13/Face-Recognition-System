import cv2
import numpy as np


MODEL_PATH = "face_detection_yunet.onnx"

FACE_DETECTOR = cv2.FaceDetectorYN.create(
    MODEL_PATH,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)


def detect_faces(image):
    height, width = image.shape[:2]

    FACE_DETECTOR.setInputSize((width, height))

    _, faces = FACE_DETECTOR.detect(image)

    if faces is None:
        return []

    results = []

    for face in faces:
        x, y, w, h = map(int, face[:4])
        results.append((x, y, w, h))

    return results


def extract_face(image, face_box, size=(47, 62)):
    x, y, w, h = face_box

    x = max(0, x)
    y = max(0, y)

    face = image[y:y + h, x:x + w]

    if face.size == 0:
        return None

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray, size)

    normalized = resized.astype(np.float32) / 255.0

    return normalized