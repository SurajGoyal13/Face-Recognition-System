from pathlib import Path

import cv2
import joblib
import numpy as np

from flask import Flask, render_template, request


app = Flask(__name__)

GALLERY_PATH = Path("models/face_gallery.pkl")
SFACE_MODEL = "models/face_recognition_sface_2021dec.onnx"
YUNET_MODEL = "face_detection_yunet.onnx"

gallery = joblib.load(GALLERY_PATH)

detector = cv2.FaceDetectorYN.create(
    YUNET_MODEL,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)

recognizer = cv2.FaceRecognizerSF.create(
    SFACE_MODEL,
    ""
)

# Stricter rejection threshold than before.
COSINE_THRESHOLD = 0.45


def extract_embedding(image, face_box=None):

    if face_box is not None:

        x, y, w, h = face_box

        x = max(0, x)
        y = max(0, y)

        x2 = min(image.shape[1], x + w)
        y2 = min(image.shape[0], y + h)

        image = image[y:y2, x:x2]

    if image.size == 0:
        return None

    image = cv2.resize(
        image,
        (112, 112)
    )

    feature = recognizer.feature(image)

    feature = np.asarray(
        feature,
        dtype=np.float32
    ).flatten()

    norm = np.linalg.norm(feature)

    if norm == 0:
        return None

    return feature / norm


def find_best_match(feature):

    best_person = "Unknown Person"
    best_score = -1.0

    for person, embeddings in gallery.items():

        # Compare against every stored example
        # for this identity.
        scores = embeddings @ feature

        person_score = float(
            np.max(scores)
        )

        if person_score > best_score:
            best_score = person_score
            best_person = person

    if best_score < COSINE_THRESHOLD:
        return "Unknown Person", best_score

    return best_person, best_score


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return render_template(
            "index.html",
            error="Please upload an image."
        )

    file = request.files["image"]

    if not file.filename:
        return render_template(
            "index.html",
            error="Please select an image."
        )

    data = np.frombuffer(
        file.read(),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        data,
        cv2.IMREAD_COLOR
    )

    if image is None:
        return render_template(
            "index.html",
            error="Invalid image."
        )

    height, width = image.shape[:2]

    detector.setInputSize(
        (width, height)
    )

    _, faces = detector.detect(image)

    results = []

    if faces is None or len(faces) == 0:

        if width <= 150 and height <= 150:

            feature = extract_embedding(image)

            if feature is not None:

                person, score = find_best_match(
                    feature
                )

                results.append({
                    "name": person,
                    "confidence": round(
                        max(score, 0.0) * 100,
                        2
                    )
                })

        else:

            return render_template(
                "index.html",
                error="No face detected in the image."
            )

    else:

        for face in faces:

            box = tuple(
                map(
                    int,
                    face[:4]
                )
            )

            feature = extract_embedding(
                image,
                box
            )

            if feature is None:
                continue

            person, score = find_best_match(
                feature
            )

            results.append({
                "name": person,
                "confidence": round(
                    max(score, 0.0) * 100,
                    2
                )
            })

    if not results:
        return render_template(
            "index.html",
            error="Unable to extract facial features."
        )

    return render_template(
        "index.html",
        results=results
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )